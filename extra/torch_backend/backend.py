# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
import functools
import weakref
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import Ops
from tinygrad.helpers import getenv, prod, strides_for_shape
import torch.lib
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, math, operator, inspect
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

# https://pytorch.org/docs/stable/torch.compiler_ir.html

def _from_torch_device(device: torch.device): return f"{Device.DEFAULT}:{device.index or 0}"
def _to_torch_device(device: str): return torch.device("tiny", int(device.partition(":")[2] or 0))

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[str(pathlib.Path(__file__).parent / "wrapped_tensor.cpp")])

def calculate_storage_offset(x: Tensor) -> int:
  offset = 0
  for u in x.uop.toposort():
    if u.op == Ops.SHRINK:
      for i, (start, _) in enumerate(u.marg): offset += start * strides_for_shape(u.src[0].shape)[i]
  return offset

def wrap(x: Tensor) -> torch.Tensor:
  x._strides = strides_for_shape(x.shape)
  if not hasattr(x, '_storage_offset'): x._storage_offset = calculate_storage_offset(x)
  return mod.wrap(x, _to_torch_dtype(x.dtype), _to_torch_device(x.device).index)

def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend:
  def is_initialized(self): return True
  def is_available(self): return True
  def current_device(self): return 0
  def _is_in_bad_fork(self): return False
  def manual_seed_all(self, seed: int): Tensor.manual_seed(seed)
  def device_count(self): return getenv("GPUS", 1) # TODO: device count in tiny?
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend())
torch.utils.generate_methods_for_privateuse1_backend()
aten = torch.ops.aten

# track view relationships for in place operations
def is_view(tensor: Tensor): return hasattr(tensor, "_view_base")
def canonical_base(view: Tensor): return getattr(view, "_view_base", view)
def derived_views(base: Tensor): return [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]

def _get_view_ops(view: Tensor) -> list:
  actual_base_uop = canonical_base(view).uop
  ops, cur = [], view.uop
  while cur is not actual_base_uop and len(cur.src) > 0:
    if cur.op in MOVEMENT_OPS: ops.append(cur)
    cur = cur.src[0]
  return list(reversed(ops))

def _apply_view_ops(target: Tensor, ops: list) -> Tensor:
  for u in ops: target = MOVEMENT_OPS[u.op](target, u)
  return target

def _replay_view_from_base(target: Tensor, view: Tensor) -> Tensor:
  if view is target or view.uop is target.uop: return view
  actual_base = canonical_base(view)
  if target.shape != actual_base.shape: return view
  return _apply_view_ops(target, _get_view_ops(view))

def _view_write(base: Tensor, view: Tensor, value: Tensor) -> None:
  tgt_dtype = base.dtype
  val = value.cast(tgt_dtype) if value.dtype != tgt_dtype else value
  if view.shape == base.shape:
    base.assign(val)
    return
  idx_base = Tensor.arange(base.numel(), device=base.device, dtype=dtypes.int32).reshape(base.shape)
  idx_view = _replay_view_from_base(idx_base, view).reshape(-1)
  flat_base = base.reshape(base.numel()).contiguous()
  flat_base[idx_view] = val.reshape(-1)
  base.assign(flat_base.reshape(base.shape))

def _apply_inplace(target: Tensor, value: Tensor) -> None:
  val = value.cast(target.dtype) if value.dtype != target.dtype else value
  if target.uop.is_realized or not is_view(target):
    target.assign(val)
    return
  base = canonical_base(target)
  if target is base or target.uop is base.uop:
    target.assign(val)
    return
  views = derived_views(base)
  if not views:
    target.assign(val)
    return
  view_ops_map = {v: _get_view_ops(v) for v in views}
  _view_write(base, target, val)
  for v in views:
    v.replace(_apply_view_ops(base, view_ops_map[v]))

def wrap_view_op(fn):
  def _wrap(*args,**kwargs):
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    ret = fn(*args,**kwargs)
    if ret is None: raise NotImplementedError("view operation returned None")
    ret._view_base = base = canonical_base(args[0])
    if not hasattr(base, "_views"): base._views = set()
    base._views.add(weakref.ref(ret))
    return wrap(ret)
  return _wrap

view_ops = {
  "aten.view": Tensor.reshape,
  "aten._unsafe_view": Tensor.reshape,  # when are views unsafe, and do we care?
  "aten.view.dtype": lambda self,dtype: self.bitcast(_from_torch_dtype(dtype)),
  "aten.expand": Tensor.expand,
  "aten.t": Tensor.transpose,
  "aten.transpose.int": Tensor.transpose,
  "aten.squeeze.dim": Tensor.squeeze,
  "aten.unsqueeze": Tensor.unsqueeze,
  "aten.detach": Tensor.detach,
  "aten.select.int": lambda self, dim, idx: self[(slice(None),) * (dim%self.ndim) + (idx,)],
  "aten.permute": Tensor.permute,
  "aten.alias": lambda self: self,
  }

for k,v in view_ops.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_view_op(v))

# TODO do we want Ops.MULTI here?
MOVEMENT_OPS = {
  Ops.RESHAPE: lambda t, u: t.reshape(u.shape),
  Ops.EXPAND: lambda t, u: t.expand(u.shape),
  Ops.SHRINK: lambda t, u: t.shrink(u.marg),
  Ops.PAD: lambda t, u: t.pad(u.marg),
  Ops.PERMUTE: lambda t, u: t.permute(u.marg),
  Ops.FLIP: lambda t, u: t.flip(u.marg),
}


# *** bad functions on CPU ***

@torch.library.impl("aten::_index_put_impl_", "privateuseone")
def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
  # TODO: move to tinygrad
  ret = aten._index_put_impl_(self.cpu(), [x.cpu() if isinstance(x, torch.Tensor) else None for x in indices], values.cpu(), accumulate, unsafe).to(self.device)
  return wrap(unwrap(self).assign(unwrap(ret)))

@torch.library.impl("aten::index_put", "privateuseone")
def index_put(self, indices, values, accumulate=False):
  return aten.index_put(self.cpu(), [z.cpu() if isinstance(z, torch.Tensor) else None for z in indices], values.clone().cpu(), accumulate).tiny()

@torch.library.impl("aten::isin.Tensor_Tensor_out", "privateuseone")
def isin_tensor_tensor_out(x, y, *, assume_unique=False, invert=False, out=None): return out.copy_(aten.isin(x.cpu(), y.cpu(), assume_unique=assume_unique, invert=invert).tiny())

@torch.library.impl("aten::randperm.generator_out", "privateuseone")
def randperm_generator(n, generator=None, out=None):
  return out.copy_(wrap(Tensor.randperm(n, generator=generator, device=unwrap(out).device)))

@torch.library.impl("aten::cummax", "privateuseone")
def cummax(self, dim):
  # TODO: support cummax with indices to match torch
  cummax, indices = aten.cummax(self.cpu(), dim)
  return (cummax.tiny(), indices.tiny())

@torch.library.impl("aten::nonzero", "privateuseone")
# TODO: move to tinygrad
def nonzero(self): return aten.nonzero(self.cpu()).tiny()

@torch.library.impl("aten::_linalg_eigh", "privateuseone")
# TODO: move to tinygrad
def _linalg_eigh(self, UPLO: str = 'U'):
  w, v = torch.linalg.eigh(self.cpu(), UPLO=UPLO)
  return w.tiny(), v.tiny()

@torch.library.impl("aten::_linalg_det", "privateuseone")
# TODO: move to tinygrad
def _linalg_det(self: torch.Tensor):
  result = aten._linalg_det(self.cpu())
  return result[0].tiny(), result[1].tiny(), result[2].tiny()

def upsample_backward(grad_out, output_size, input_size, *args, f=None): return f(grad_out.cpu(), output_size, input_size, *args).tiny()

for i in [
  "upsample_linear1d_backward", "upsample_nearest1d_backward", "_upsample_nearest_exact1d_backward",
  "upsample_nearest2d_backward", "_upsample_nearest_exact2d_backward",
  "upsample_nearest3d_backward", "_upsample_nearest_exact3d_backward",
  "upsample_trilinear3d_backward", "upsample_bilinear2d_backward"
]:
  torch.library.impl(f"aten::{i}", "privateuseone")(functools.partial(upsample_backward, f=getattr(aten, i)))

# *** end bad functions on CPU ***

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y):
  return wrap(unwrap(x)[[unwrap(_y.to(x.device)) if _y is not None else slice(None) for _y in y]])

# Helper for inplace operations that use replace for non-views
def _inplace_op(t, new_value):
  if not is_view(t): t.replace(new_value)
  else: _apply_inplace(t, new_value)
  return t

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor): return unwrap(tensor).item()

def as_strided_view(base:Tensor, size, stride, storage_offset):
  non_broadcast_size = tuple(s for s, st in zip(size, stride) if st != 0)
  if all(st != 0 for st in stride): return base.shrink(((storage_offset, storage_offset + prod(size)),)).reshape(size)
  return base.shrink(((storage_offset, storage_offset + prod(non_broadcast_size)),)).reshape(tuple(s if st != 0 else 1 for s, st in zip(size, stride))).expand(size)

def as_strided_gather(base:Tensor, size, stride, storage_offset):
  indices = Tensor.full(size, storage_offset, dtype=dtypes.int32, device=base.device)
  for dim, (sz, st) in enumerate(zip(size, stride)):
    if st != 0: indices += (Tensor.arange(sz, device=base.device, dtype=dtypes.int32) * st).reshape((1,) * dim + (sz,) + (1,) * (len(size) - dim - 1))
  return base[indices.flatten()].reshape(size)

@wrap_view_op
def _as_strided(tensor:Tensor, size, stride, storage_offset=0):
  base = canonical_base(tensor).flatten()
  non_broadcast_stride = tuple(st for st in stride if st != 0)
  non_broadcast_size = tuple(s for s, st in zip(size, stride) if st != 0)
  is_contiguous = non_broadcast_stride == strides_for_shape(non_broadcast_size) and storage_offset + prod(size) <= base.shape[0]
  result = as_strided_view(base, size, stride, storage_offset) if is_contiguous else as_strided_gather(base, size, stride, storage_offset)
  result._as_strided_base = base
  return result

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
  storage_offset = storage_offset or tensor.storage_offset()
  return _as_strided(tensor, size, stride, storage_offset)

@torch.library.impl("aten::_reshape_alias", "privateuseone")
def _reshape_alias(tensor:torch.Tensor, size, stride):
  return _as_strided(tensor, size, stride)

def _empty_tensor(size, dtype=None, device=None, **kwargs):
  if TORCH_DEBUG: print(f"empty {size=} {dtype=} {device=} {kwargs=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype or torch.get_default_dtype()), device=_from_torch_device(device))
  return wrap(ret)

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout=None, device=None, pin_memory=False):
  # TODO: should return with requested strides
  return _empty_tensor(size, dtype, device, stride=stride, layout=layout, pin_memory=pin_memory)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  return _empty_tensor(size, dtype, device, layout=layout, pin_memory=pin_memory, memory_format=memory_format)

@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_with_indices(self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: supprt stride [] in tinygrad?
  if stride is not None and len(stride) == 0: stride = None
  ret, idx = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode, return_indices=True)
  return (wrap(ret), wrap(idx.cast(dtypes.int64)))

@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_with_indices_backward(grad_out:torch.Tensor, self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False, indices=None):
  return wrap(Tensor.max_unpool2d(unwrap(grad_out), unwrap(indices), output_size=unwrap(self).shape))

@torch.library.impl("aten::max_unpool2d", "privateuseone")
def max_unpool2d(self:torch.Tensor, indices:torch.Tensor, output_size):
  return wrap(unwrap(self).max_unpool2d(unwrap(indices), output_size=output_size))

def _arange_dtype(*args, dtype=None):
  has_float = any(isinstance(x, float) for x in args)
  return _from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))

@torch.library.impl("aten::arange", "privateuseone")
def arange(end, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(0, end, dtype=_arange_dtype(end, dtype=dtype)))

@torch.library.impl("aten::arange.start", "privateuseone")
def arange_start(start, end, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(start, end, dtype=_arange_dtype(start, end, dtype=dtype)))

@torch.library.impl("aten::arange.start_step", "privateuseone")
def arange_start_step(start, end, step, dtype=None, device=None, pin_memory=None):
  return wrap(Tensor.arange(start, end, step, dtype=_arange_dtype(start, end, step, dtype=dtype)))

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  if TORCH_DEBUG >= 1:
    print(f"convolution {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  input, weight, bias = unwrap(input), unwrap(weight), unwrap(bias) if bias is not None else None
  # TODO: fix test_biased_conv2d fails without realize()
  conv_fn = input.conv2d if not transposed else input.conv_transpose2d
  kwargs = dict(groups=groups, stride=stride, dilation=dilation, padding=padding)
  if transposed: kwargs['output_padding'] = output_padding
  return wrap(conv_fn(weight, bias, **kwargs))

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_out, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  
  # Unwrap and detach to avoid building on top of existing graph
  grad_out_t = unwrap(grad_out).detach()
  input_t = unwrap(input).detach()
  weight_t = unwrap(weight).detach()
  
  bias_shape = weight_t.shape[1] * groups if transposed else weight_t.shape[0]
  bias_t = Tensor.zeros(bias_shape, device=input_t.device, dtype=input_t.dtype)
  
  if not transposed: 
    out = input_t.conv2d(weight_t, bias_t if output_mask[2] else None, groups=groups, stride=stride, dilation=dilation, padding=padding)
  else: 
    out = input_t.conv_transpose2d(weight_t, bias_t if output_mask[2] else None, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding)

  targets = [t for t, m in zip([input_t, weight_t, bias_t], output_mask) if m]
  grads = out.gradient(*targets, gradient=grad_out_t)
  
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])
@torch.library.impl("aten::slice.Tensor", "privateuseone")
@wrap_view_op
def slice_tensor(self, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * self.ndim
  slices[dim] = slice(start, end, step)
  return self[slices]

@torch.library.impl("aten::slice_backward", "privateuseone")
def slice_backward(grad_out, input_sizes, dim, start, end, step):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = slice(start, end, step)
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

@torch.library.impl("aten::select_backward", "privateuseone")
def select_backward(grad_out, input_sizes, dim, index):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = index
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

def avg_pool(self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  return wrap(unwrap(self).avg_pool2d(kernel_size, stride if stride != [] else None, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad))

def avg_pool_backward(grad_out, self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.avg_pool2d(self, kernel_size, stride if stride != [] else None, dilation=1, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [2, 3]:
  torch.library.impl(f"aten::avg_pool{dim}d", "privateuseone")(avg_pool)
  torch.library.impl(f"aten::avg_pool{dim}d_backward", "privateuseone")(avg_pool_backward)

def pad_forward(self, padding, mode=None): return wrap(Tensor.pad(unwrap(self), padding, mode=mode))

def pad_backward(grad_out, self, padding, mode):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.pad(self, padding, mode=mode)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [1, 2, 3]:
  for pad_type, mode in [("replication", "replicate"), ("reflection", "reflect")]:
    torch.library.impl(f"aten::{pad_type}_pad{dim}d", "privateuseone")(functools.partial(pad_forward, mode=mode))
    torch.library.impl(f"aten::{pad_type}_pad{dim}d_backward", "privateuseone")(functools.partial(pad_backward, mode=mode))

def upsample(self, size, align_corners=False, mode=None): return wrap(Tensor.interpolate(unwrap(self), size, mode=mode, align_corners=align_corners))
for i,pre in enumerate(["", "bi", "tri"]):
  torch.library.impl(f"aten::upsample_{pre}linear{i+1}d", "privateuseone")(functools.partial(upsample, mode="linear"))
  torch.library.impl(f"aten::upsample_nearest{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest"))
  torch.library.impl(f"aten::_upsample_nearest_exact{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest-exact"))

@torch.library.impl("aten::scatter_add.out", "privateuseone")
def scatter_add(self, dim, index, src, out):
  self, index, src, out = unwrap(self), unwrap(index), unwrap(src), unwrap(out)
  if self.shape == ():
    _apply_inplace(out, src)
    return wrap(out)
  _apply_inplace(out, Tensor.scatter_reduce(self, dim, index, src, reduce='sum'))
  return wrap(out)

def _copy_between_devices(src, dest, cast_dtype, to_device, non_blocking=False):
  if src.is_tiny and dest.is_tiny:
    src_t, dest_t = unwrap(src), unwrap(dest)
    if dest_t.uop.is_contiguous() or dest_t.uop.is_realized: src_t = src_t.contiguous()
    _apply_inplace(dest_t, src_t.cast(cast_dtype).to(to_device))
  elif src.is_tiny and dest.is_cpu:
    dest.resize_(src.numel()).resize_(src.shape)
    dest.copy_(torch.from_numpy(unwrap(src).cast(cast_dtype).numpy()))
  elif src.is_cpu and dest.is_tiny:
    unwrap(dest).assign(Tensor(src.numpy()).cast(cast_dtype).to(to_device))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  cast_dtype = _from_torch_dtype(dest.dtype)
  to_device = _from_torch_device(dest.device)
  _copy_between_devices(src, dest, cast_dtype, to_device, non_blocking)
  return dest

@torch.library.impl("aten::copy_", "privateuseone")
def copy_(self, src, non_blocking=False):
  cast_dtype = _from_torch_dtype(self.dtype)
  to_device = _from_torch_device(self.device)
  if TORCH_DEBUG and src.is_tiny and self.is_tiny:
    dest_t = unwrap(self)
    print("copy_ tiny->tiny", dest_t.shape, "from", unwrap(src).shape, "is_view", is_view(dest_t), "is_realized", dest_t.uop.is_realized)
  _copy_between_devices(src, self, cast_dtype, to_device, non_blocking)
  return self


@torch.library.impl("aten::cat.out", "privateuseone")
def cat_out(tensors, dim=0, out=None):
  out_t = unwrap(out)
  _apply_inplace(out_t, Tensor.cat(*[unwrap(x) for x in tensors], dim=dim))
  return wrap(out_t)

@torch.library.impl("aten::topk.values", "privateuseone")
def topk_values(input, k, dim=None, largest=True, sorted=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).topk(k, dim if dim is not None else -1, largest, sorted)
  val_t, idx_t = unwrap(values), unwrap(indices)
  _apply_inplace(val_t, out_values)
  _apply_inplace(idx_t, out_indices.cast(dtypes.int64))
  return wrap(out_values), wrap(out_indices)

@torch.library.impl("aten::sort.values_stable", "privateuseone")
def sort_values(input, dim=-1, descending=False, stable=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).sort(dim, descending)
  val_t, idx_t = unwrap(values), unwrap(indices)
  _apply_inplace(val_t, out_values)
  _apply_inplace(idx_t, out_indices.cast(dtypes.int64))
  return wrap(out_values), wrap(out_indices)

@torch.library.impl("aten::_linalg_svd", "privateuseone")
def _linalg_svd(self, full_matrices=False):
  U, S, Vh = unwrap(self).svd(full_matrices)
  return wrap(U), wrap(S), wrap(Vh)

# register some decompositions
from torch._decomp import get_decompositions
decomps = [
  aten.native_layer_norm_backward,
  aten.linalg_cross,
  aten.addmm,
  aten.addcmul,
  aten.addcdiv,
  aten._log_softmax_backward_data,
  aten.threshold_backward,
  aten.softplus_backward,
  aten.elu,  # elu has a scale + input_scale param
  aten.elu_backward,
  aten.softplus,
  aten.logaddexp,
  aten.threshold,
  aten.nll_loss_forward,
  aten.nll_loss_backward,
  aten.nll_loss2d_backward,
  # AttributeError: 'int' object has no attribute '_broadcasted'
  aten.sigmoid_backward,
  aten.tanh_backward,
  aten.sinc,
  aten._prelu_kernel,
  aten.softshrink,
  aten.hardshrink,
  aten.log_sigmoid_forward,
  aten.log_sigmoid_backward,
  aten.isneginf,
  aten.isposinf,
  aten.nan_to_num,
  aten.logit,
  aten.rsub,
  aten.index_select,
  aten.native_dropout, aten.native_dropout_backward,
  aten._softmax_backward_data, aten.embedding_dense_backward,
  aten.linalg_vector_norm,
  aten.binary_cross_entropy, aten.binary_cross_entropy_backward,
  aten.upsample_nearest2d.out,
  # activations
  aten.hardswish, aten.hardswish_backward,
  aten.hardtanh, aten.hardtanh_backward,
  aten.gelu, aten.gelu_backward,
  aten.logical_and,
  aten.randint,
  aten.eye,
  aten.hardsigmoid_backward,
  aten.leaky_relu_backward,
  aten.nll_loss2d_forward,
  aten.unfold_backward,
  # NOTE: many of these don't work or cause infinite loops
  #aten.var_mean,
  #aten.var,
  #aten.rsqrt,
  #aten.max_pool2d_with_indices,
  # NOTE: these are prims
  #aten.digamma,
  #aten.erfinv,
  #aten.lgamma,
  # this needs copy_strided
  #aten.lerp,
  aten.norm,
]
for k,v in get_decompositions(decomps).items():
  key = str(k._schema).split("(")[0]
  if TORCH_DEBUG >= 2: print("register decomp for", k)
  torch.library.impl(key, "privateuseone")(v)

# NOTE: we should only implement the "out" form, it should be 0 overhead
# TODO: due to issue with empty / is_realized, it is slow to use assign so we use replace
# the goal is to make as much as we can this
simple_tensor_methods = [
  # unary (ish)
  "log", "log2", "sqrt", "rsqrt", "sign", "silu", "hardsigmoid", "exp", "exp2", "neg", "reciprocal", "bitwise_not",
  "sigmoid", "clamp", "mish", "erf", "leaky_relu",
  # trig
  "acos", "acosh", "cos", "cosh", "asin", "asinh", "sin", "sinh", "atan", "atanh", "tan", "tanh",
  # rounding
  "ceil", "round", "floor", "trunc",
  # binary
  "mul", "div", "maximum", "minimum", "copysign",
  # modify
  "tril", "triu",
  # reduce
  "all", "any", "argmax", "argmin", "cumsum", "cumprod",
  # complex
  "avg_pool2d", "linspace"]

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  "aten.add.out": lambda input,other,alpha=1: input+alpha*other,
  "aten.sub.out": lambda input,other,alpha=1: input-alpha*other, # NOTE: this is also needed to handle reverse
  "aten.div.out_mode": Tensor.div,
  "aten.mul.out": operator.mul,
  "aten.bmm.out": operator.matmul,
  # NOTE: because these methods have a name with "Tensor" in them, they can't go in simple tensor methods
  "aten.remainder.Tensor_out": Tensor.mod,
  "aten.pow.Tensor_Tensor_out": Tensor.pow,
  "aten.pow.Tensor_Scalar_out": Tensor.pow,
  "aten.pow.Scalar_out": lambda input,exponent: input**exponent,
  "aten.bitwise_and.Tensor_out": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor_out": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor_out": Tensor.bitwise_xor,
  "aten.eq.Tensor_out": Tensor.eq, "aten.eq.Scalar_out": Tensor.eq,
  "aten.ne.Tensor_out": Tensor.ne, "aten.ne.Scalar_out": Tensor.ne,
  "aten.ge.Tensor_out": Tensor.__ge__, "aten.ge.Scalar_out": Tensor.__ge__,
  "aten.gt.Tensor_out": Tensor.__gt__, "aten.gt.Scalar_out": Tensor.__gt__,
  "aten.lt.Tensor_out": Tensor.__lt__, "aten.lt.Scalar_out": Tensor.__lt__,
  "aten.le.Tensor_out": Tensor.__le__, "aten.le.Scalar_out": Tensor.__le__,
  "aten.clamp_max.Tensor_out": lambda input,max_: input.clamp(max_=max_),
  "aten.clamp_min.Tensor_out": lambda input,min_: input.clamp(min_=min_),
  "aten.fmod.Tensor_out": lambda input,other: input-input.div(other, rounding_mode="trunc")*other,
  # TODO: this might result in overflow issues
  "aten.round.decimals_out": lambda self,decimals: (self*10**decimals).round()/10**decimals,
  # TODO: support this in tinygrad
  "aten.bitwise_left_shift.Tensor_out": lambda x,y: x*(2**y),
  "aten.bitwise_right_shift.Tensor_out": lambda x,y: x//(2**y),
  # not in tinygrad. are there decomps for these?
  "aten.log10.out": lambda self: self.log2() * (math.log(2) / math.log(10)),
  "aten.log1p.out": lambda self: (self+1).log(),
  "aten.expm1.out": lambda self: self.exp() - 1,
  "aten.fmax.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.maximum(input, other))),
  "aten.fmin.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.minimum(input, other))),
  "aten.amax.out": lambda self,dim=None: self.max(axis=dim),
  "aten.amin.out": lambda self,dim=None: self.min(axis=dim),
  # TODO: this gets the shape wrong
  #"aten.arange.start_out": Tensor.arange,
  "aten.lerp.Scalar_out": Tensor.lerp,
  "aten.scatter.value_out": Tensor.scatter,
  "aten.where.self_out": Tensor.where,
  "aten.prod.int_out": Tensor.prod,
  "aten.scatter.src_out": Tensor.scatter,
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self, dim, keepdim=False, dtype=None: (
      self.sum(
          # torch semantics: dim=None or dim == [] -> reduce over all dims
          None if (dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0)) else dim,
          keepdim=keepdim,
          dtype=_from_torch_dtype(dtype) if dtype is not None else None,
      )
  ),
}}

# we add the "out" here
def wrap_out(f):
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    if getenv("ALLOW_DTYPE_MISMATCH", 1): assigned = assigned.cast(out.dtype)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert out.device == assigned.device, f"device mismatch: {assigned.device} -> {out.device}"
    assert out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    return out.assign(assigned)
  return _wrap_out

def _fill_tensor_tensor(self: Tensor, value: Tensor) -> Tensor:
  if value.numel() != 1: raise RuntimeError("fill_ expects a 0-d tensor value")
  scalar_value = value.reshape(()).item()
  return self.assign(Tensor.full(self.shape, scalar_value, device=self.device, dtype=self.dtype))

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  "aten.remainder.Scalar_Tensor": lambda x,y: x%y,
  "aten.floor_divide": lambda x,y: x//y,
  "aten.floor_divide_.Tensor": lambda x,y: x.assign(x//y),
  # TODO: use tinygrad methods, but they require x to be unsigned
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__ilshift__.Scalar": lambda x,y: x.assign(x*(2**y)),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  "aten.__irshift__.Scalar": lambda x,y: x.assign(x//(2**y)),
  # inplace ops using replace for fusion
  "aten.zero_": lambda x: _inplace_op(x, x.zeros_like()),
  "aten.fill_.Scalar": lambda x, y: _inplace_op(x, x.full_like(y)),
  "aten.add_.Tensor": lambda self, other, alpha=1.0: _inplace_op(self, self + other * alpha),
  "aten.add_.Scalar": lambda self, other, alpha=1.0: _inplace_op(self, self + other * alpha),
  "aten.mul_.Tensor": lambda self, other: _inplace_op(self, self * other),
  "aten.mul_.Scalar": lambda self, other: _inplace_op(self, self * other),
  # relu doesn't have an out form?
  "aten.relu": Tensor.relu,
  "aten.relu_": lambda x: x.assign(x.relu()),
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.max": Tensor.max,
  "aten.mm": Tensor.matmul,
  "aten.mv": Tensor.matmul,
  "aten.dot": Tensor.dot,
  "aten.prod": Tensor.prod,
  "aten.isnan": Tensor.isnan,
  "aten.std.correction": Tensor.std,
  "aten.std_mean.correction": Tensor.std_mean,
  "aten.var.correction": Tensor.var,
  "aten.var_mean.correction": Tensor.var_mean,
  "aten.scatter.value": Tensor.scatter,
  "aten.scatter.value_reduce": Tensor.scatter,
  "aten.gather": lambda self, dim, index: self.gather(dim, index.cast(dtypes.int)),
  "aten.where.self": Tensor.where, # NOTE: this is needed as well as the out type
  "aten.repeat": Tensor.repeat,
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten._log_softmax": lambda self,dim,half_to_float: self.log_softmax(dim),
  "aten.random_": lambda self: self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype)),
  "aten.random_.from": lambda self, from_, to: self.assign(Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype)),
  "aten.uniform_": lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high, dtype=self.dtype)),
  "aten.normal_": lambda self, mean=0, std=1: self.assign(Tensor.normal(*self.shape, mean=mean, std=std, dtype=self.dtype)),
  # these don't work in out form, they have size 0
  "aten.abs": Tensor.abs,
  "aten.logical_not": Tensor.logical_not,
  "aten.logical_or_": lambda x, y: x.assign(x | y),
  "aten.multinomial": Tensor.multinomial,
  "aten.masked_fill_.Scalar": lambda self, mask, value: self.assign(self.masked_fill(mask, value)),
  "aten.masked_fill_.Tensor": lambda self, mask, value: self.assign(self.masked_fill(mask, value)),
  "aten.masked_fill.Scalar": Tensor.masked_fill,
  "aten.masked_fill.Tensor": Tensor.masked_fill,
  "aten.masked_select": Tensor.masked_select,
  "aten.all": Tensor.all,
  "aten.sgn": Tensor.sign,
  "aten.acos": Tensor.acos,
  "aten.any": Tensor.any,
  "aten.bitwise_not": Tensor.bitwise_not,
  "aten.argmax": Tensor.argmax,
  "aten.argmin": Tensor.argmin,
  "aten.asinh": Tensor.asinh,
  "aten.mul": Tensor.mul,
  "aten.atanh": Tensor.atanh,
  "aten.fill_.Tensor": _fill_tensor_tensor,
  "aten.flip": Tensor.flip,
  "aten.scatter_reduce.two": Tensor.scatter_reduce,
  "aten.squeeze_.dim": lambda self, dim: self.replace(self.squeeze(dim), allow_shape_mismatch=True), # TODO: inplace view op, here?
  "aten.add.Tensor": lambda input,other,alpha=1: input+alpha*other,
  "aten.linspace": lambda start, stop, steps, dtype=None, **kwargs:
    Tensor.linspace(start, stop, steps, **({"dtype": _from_torch_dtype(dtype)} if dtype is not None else {})),
  "aten.topk": Tensor.topk,
  "aten.constant_pad_nd": lambda self, padding, value=0.0: self.pad(padding, mode="constant", value=value),
  "aten.cumsum": Tensor.cumsum,
  "aten.logsumexp": lambda self, axis, keepdim=False: self.logsumexp(axis[0], keepdim=keepdim),
  "aten.roll": Tensor.roll,
  "aten.logcumsumexp": Tensor.logcumsumexp,
  "aten.lerp.Tensor": Tensor.lerp,
  "aten.ones_like": lambda self, dtype=None, device=None, **kwargs:
    self.ones_like(**{k: v for k, v in {"dtype": _from_torch_dtype(dtype) if dtype else None,
                                        "device": _from_torch_device(device) if device else None}.items() if v is not None}),
  "aten.max.dim": lambda self, dim, keepdim=False: (self.max(dim, keepdim), self.argmax(dim, keepdim).cast(dtype=dtypes.int64)),
  "aten.unfold": Tensor.unfold,
}}

def _wrap_output(out):
  if isinstance(out, Tensor): return wrap(out)
  elif isinstance(out, tuple): return tuple(wrap(x) for x in out)
  else: raise RuntimeError(f"unknown output type {type(out)}")

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    if TORCH_DEBUG:
      print(k, len(args), [x.shape if isinstance(x, torch.Tensor) else x for x in args],
                          {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()})
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    return _wrap_output(f(*args, **kwargs))
  return nf

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_fxn(k,v))

@torch.library.impl("aten::equal", "privateuseone")
def equal(x: torch.Tensor, y: torch.Tensor): return (x==y).all().item()

if TORCH_DEBUG:
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  (_dispatch_log:=DispatchLog()).__enter__() # NOTE: must be kept alive

# NOTE: patch torch optimizer step to avoid continously growing the computation graph
_torch_modules_with_buffers: weakref.WeakSet[torch.nn.Module] = weakref.WeakSet()
def register_torch_buffer(mod, _name, _buffer): _torch_modules_with_buffers.add(mod)

torch.nn.modules.module.register_module_buffer_registration_hook(register_torch_buffer)

torch.nn.modules.module.register_module_module_registration_hook(lambda module, _name, _submodule: None)

def realize_optimizer_step(optimizer: torch.optim.Optimizer, *args, **kwargs):
  # Don't realize after every optimizer step - let tinygrad schedule and fuse operations
  # The next forward pass or explicit sync point will trigger realization
  pass

_optimizer_init = torch.optim.Optimizer.__init__
def _optimizer_patched_init(self, *args, **kwargs):
  _optimizer_init(self, *args, **kwargs)
  self.register_step_post_hook(realize_optimizer_step)
torch.optim.Optimizer.__init__ = _optimizer_patched_init

@torch.library.impl("aten::native_batch_norm", "privateuseone")
def native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps):
  input_t, weight_t, bias_t = unwrap(input), unwrap(weight) if weight is not None else None, unwrap(bias) if bias is not None else None
  running_mean_t, running_var_t = unwrap(running_mean) if running_mean is not None else None, unwrap(running_var) if running_var is not None else None

  if training:
    batch_var, batch_mean = input_t.var_mean(axis=tuple(x for x in range(input_t.ndim) if x != 1), correction=0)
    batch_invstd = batch_var.add(eps).rsqrt()
    out = input_t.batchnorm(weight_t, bias_t, batch_mean, batch_invstd)
    if running_mean_t is not None and running_var_t is not None:
      numel_ratio = input_t.numel() / (input_t.numel() - input_t.shape[1])
      running_mean_t.assign((1 - momentum) * running_mean_t + momentum * batch_mean.detach())
      running_var_t.assign((1 - momentum) * running_var_t + momentum * numel_ratio * batch_var.detach())
    return wrap(out), wrap(batch_mean), wrap(batch_invstd)
  else:
    out = input_t.batchnorm(weight_t, bias_t, running_mean_t, running_var_t.add(eps).rsqrt())
    return wrap(out), wrap(running_mean_t), wrap(running_var_t.add(eps).rsqrt())

@torch.library.impl("aten::native_batch_norm_backward", "privateuseone")
def native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask):
  grad_out_t, input_t = unwrap(grad_out), unwrap(input)
  weight_t = unwrap(weight) if weight is not None else None
  save_mean_t = unwrap(save_mean)
  save_invstd_t = unwrap(save_invstd)

  # Forward pass to get computation graph for backward
  out = input_t.batchnorm(weight_t, None, save_mean_t, save_invstd_t)

  # Compute gradients
  targets = [t for t, m in zip([input_t, weight_t], output_mask[:2]) if t is not None and m]
  if targets:
    grads = out.gradient(*targets, gradient=grad_out_t)
    grad_input = grads.pop(0) if output_mask[0] else None
    grad_weight = grads.pop(0) if output_mask[1] and weight_t is not None else None
  else:
    grad_input, grad_weight = None, None

  # Grad bias is just sum of grad_out over batch dimensions
  grad_bias = grad_out_t.sum(axis=tuple(x for x in range(grad_out_t.ndim) if x != 1)) if output_mask[2] else None

  return (wrap(grad_input) if grad_input is not None else None,
          wrap(grad_weight) if grad_weight is not None else None,
          wrap(grad_bias) if grad_bias is not None else None)

# still ugly -> is there a better way?
# aten::pad_circular1d does not exist
# aten::_pad_circular used under the hood but registering that doesn't work either
# only place that needs an explicit AutogradPrivateUse1 registration, this can't be the best way
class CircularPad(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, pad):
    ctx.save_for_backward(input)
    ctx.pad = pad
    return wrap(unwrap(input).pad(pad, mode="circular"))
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return wrap(unwrap(input).pad(ctx.pad, mode="circular").gradient(unwrap(input), gradient=unwrap(grad_output))[0]), None

@torch.library.impl("aten::_pad_circular", "privateuseone")
def _pad_circular(self, padding): return CircularPad.apply(self, padding)

@torch.library.impl("aten::_pad_circular", "AutogradPrivateUse1")
def _pad_circular_autograd(self, padding): return CircularPad.apply(self, padding)

# this is Tensor.diagonal, but extended for batches and non-square
@torch.library.impl("aten::diagonal", "privateuseone")
@wrap_view_op
def diagonal(self, offset=0, dim1=0, dim2=1):
  if offset != 0: raise NotImplementedError(f"diagonal with {offset=} not implemented")
  dim1, dim2 = dim1 % self.ndim, dim2 % self.ndim
  if dim1 != self.ndim - 2 or dim2 != self.ndim - 1: raise NotImplementedError(f"diagonal with {dim1=}, {dim2=} not implemented, only last two dims supported")
  # this is Tensor.diagonal, but extended for batches and non-square
  batch_shape, m, n = self.shape[:-2], self.shape[-2], self.shape[-1]
  diag_len = min(m, n)
  return self.reshape(*batch_shape, m*n).pad(tuple((0,0) for _ in batch_shape) + ((0, diag_len),)).reshape(*batch_shape, diag_len, n+1)[..., :, 0]

# @torch.library.impl("aten::diagonal_backward", "privateuseone")
# def diagonal_backward(grad_out, input_sizes, offset, dim1, dim2):
#   # TODO: support batched diagonal_backward for multi-dimensional tensors (currently only works for 2D)
#   if offset != 0 or dim1 != 0 or dim2 != 1: raise NotImplementedError(f"diagonal_backward with {offset=}, {dim1=}, {dim2=} not implemented")
#   n = min(input_sizes[0], input_sizes[1])
#   return wrap(unwrap(grad_out).diag().pad(((0, max(0, input_sizes[0]-n)), (0, max(0, input_sizes[1]-n)))))
