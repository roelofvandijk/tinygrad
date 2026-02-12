from __future__ import annotations
from tinygrad import Tensor, UOp, nn, getenv
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.nn.state import GGML_QUANT_INFO

def custom_q4_0_linear(out:UOp, x:UOp, blocks:UOp) -> UOp:
  # out: (N, O), x: (N, I), blocks: (O, I//32, 18)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(blocks.shape) == 3
  assert all(isinstance(s, int) for s in out.shape+x.shape+blocks.shape), "custom q4_0 kernel requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  bpr = I // 32
  assert x.shape[0] == N and blocks.shape[0] == O and blocks.shape[1] == bpr and blocks.shape[2] == 18

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  r = UOp.range(bpr*16, 2, axis_type=AxisType.REDUCE)
  br, j = r//16, r%16

  acc = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
  acc = acc.after(n, o)[0].set(0.0)
  scale = (blocks[o, br, 0].cast(dtypes.ushort) + (blocks[o, br, 1].cast(dtypes.ushort) << 8)).bitcast(dtypes.half).cast(dtypes.float)
  q = blocks[o, br, j+2]
  q_lo = (q & 0xF).cast(dtypes.float) - 8.0
  q_hi = (q >> 4).cast(dtypes.float) - 8.0
  x_lo = x[n, br*32 + j].cast(dtypes.float)
  x_hi = x[n, br*32 + j + 16].cast(dtypes.float)
  acc = acc[0].set(acc.after(r)[0] + scale * (q_lo * x_lo + q_hi * x_hi), end=r)
  return out[n, o].store(acc[0].cast(out.dtype.base)).end(n, o).sink(arg=KernelInfo(name=f"custom_q4_0_linear_{N}_{O}_{I}", opts_to_apply=()))

def _q4_0_mul_mat_id_opts(n_sel:int, out_features:int, in_features:int):
  if n_sel == 4 and out_features == 3072 and in_features == 2048:
    return (Opt(OptOps.LOCAL, 1, 8), Opt(OptOps.GROUP, 1, 16))
  if n_sel == 4 and out_features == 2048 and in_features == 1536:
    return (Opt(OptOps.LOCAL, 1, 16), Opt(OptOps.GROUP, 1, 16))
  return ()

def _fp16_mul_mat_id_opts(n_sel:int, out_features:int, in_features:int):
  if n_sel == 4 and out_features == 3072 and in_features == 2048:
    return (Opt(OptOps.LOCAL, 1, 8), Opt(OptOps.GROUP, 1, 16))
  if n_sel == 4 and out_features == 2048 and in_features == 1536:
    return (Opt(OptOps.LOCAL, 1, 16), Opt(OptOps.GROUP, 1, 16))
  return ()

def custom_q4_0_mul_mat_id(out:UOp, x:UOp, scale:UOp, packed:UOp, sel:UOp) -> UOp:
  # out: (N, O), x: (N, I), scale: (E, O, I//32, 1), packed: (E, O, I//32, 16), sel: (N,)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(scale.shape) == 4 and len(packed.shape) == 4 and len(sel.shape) == 1
  assert all(isinstance(s, int) for s in out.shape+x.shape+scale.shape+packed.shape+sel.shape), "custom q4_0 kernel requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  bpr = I // 32
  E = scale.shape[0]
  assert x.shape == (N, I) and sel.shape[0] == N
  assert scale.shape == (E, O, bpr, 1) and packed.shape == (E, O, bpr, 16)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  r = UOp.range(bpr*16, 2, axis_type=AxisType.REDUCE)
  br, j = r//16, r%16
  # Clamp dynamic expert id once; avoids expensive wraparound index math in the reduction loop.
  e = sel[n].cast(dtypes.int).maximum(0).minimum(E-1).cast(dtypes.index)
  base = ((e * O) + o) * bpr + br
  scale_flat = scale.reshape(E * O * bpr)
  packed_flat = packed.reshape(E * O * bpr * 16)

  acc = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
  acc = acc.after(n, o)[0].set(0.0)
  s = scale_flat[base].cast(dtypes.float)
  q = packed_flat[base * 16 + j]
  q_lo = (q & 0xF).cast(dtypes.float) - 8.0
  q_hi = (q >> 4).cast(dtypes.float) - 8.0
  x_lo = x[n, br*32 + j].cast(dtypes.float)
  x_hi = x[n, br*32 + j + 16].cast(dtypes.float)
  acc = acc[0].set(acc.after(r)[0] + s * (q_lo * x_lo + q_hi * x_hi), end=r)
  return out[n, o].store(acc[0].cast(out.dtype.base)).end(n, o).sink(
    arg=KernelInfo(name=f"custom_q4_0_mul_mat_id_{N}_{O}_{I}", opts_to_apply=_q4_0_mul_mat_id_opts(N, O, I)))

def custom_fp16_mul_mat_id(out:UOp, x:UOp, weights:UOp, sel:UOp) -> UOp:
  # out: (N, O), x: (N, I), weights: (E, O, I), sel: (N,)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(weights.shape) == 3 and len(sel.shape) == 1
  assert all(isinstance(s, int) for s in out.shape+x.shape+weights.shape+sel.shape), "custom fp16 mul_mat_id requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  E = weights.shape[0]
  assert x.shape == (N, I) and sel.shape[0] == N and weights.shape == (E, O, I)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  r = UOp.range(I, 2, axis_type=AxisType.REDUCE)
  e = sel[n].cast(dtypes.int).maximum(0).minimum(E-1).cast(dtypes.index)

  acc = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
  acc = acc.after(n, o)[0].set(0.0)
  w_flat = weights.reshape(E * O * I)
  w = w_flat[((e * O) + o) * I + r].cast(dtypes.float)
  xv = x[n, r].cast(dtypes.float)
  acc = acc[0].set(acc.after(r)[0] + w * xv, end=r)
  return out[n, o].store(acc[0].cast(out.dtype.base)).end(n, o).sink(
    arg=KernelInfo(name=f"custom_fp16_mul_mat_id_{N}_{O}_{I}", opts_to_apply=_fp16_mul_mat_id_opts(N, O, I)))

class QuantizedLinear:
  __slots__ = ('blocks', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_dequant_fn', '_dequant_cache', '_q4_0_blocks',
               '_q4_0_scale', '_q4_0_packed')
  def __init__(self, blocks:Tensor, shape:tuple[int, int], ggml_type:int):
    self.blocks = blocks
    self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, _, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self._dequant_cache = None
    self._q4_0_blocks = None
    self._q4_0_scale = None
    self._q4_0_packed = None

  def _ensure_dequant_cache(self, device: str|tuple[str, ...]) -> None:
    if self._dequant_cache is not None and self._dequant_cache.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    w = self._dequant_fn(blocks).reshape(self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast('float16')
    self._dequant_cache = w.realize()

  def _ensure_q4_0_blocks(self, device: str|tuple[str, ...]) -> None:
    if self._q4_0_blocks is not None and self._q4_0_blocks.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    bpr = self.in_features // 32
    self._q4_0_blocks = blocks.reshape(self.out_features, bpr, 18)

  def _ensure_q4_0_separated(self, device: str|tuple[str, ...]) -> None:
    """Pre-separate Q4_0 scale and packed data for better memory coalescing."""
    if self._q4_0_scale is not None and self._q4_0_scale.device == device: return
    self._ensure_q4_0_blocks(device)
    self._q4_0_scale = self._q4_0_blocks[:, :, :2].bitcast(dtypes.float16).contiguous().realize()  # (O, bpr, 1)
    self._q4_0_packed = self._q4_0_blocks[:, :, 2:].contiguous().realize()                         # (O, bpr, 16)

  def __call__(self, x:Tensor) -> Tensor:
    use_cache = getenv("QL_CACHE_ALL", 0) == 1
    if not use_cache:
      use_cache = (
        (getenv("QL_CACHE_ATTN_Q_A", 0) == 1 and self.out_features == 1536 and self.in_features == 2048) or
        (getenv("QL_CACHE_ATTN_KV_A", 0) == 1 and self.out_features == 576 and self.in_features == 2048) or
        (getenv("QL_CACHE_ATTN_Q_B", 0) == 1 and self.out_features == 3072 and self.in_features == 1536) or
        (getenv("QL_CACHE_ATTN_OUT", 0) == 1 and self.out_features == 2048 and self.in_features == 2048) or
        (getenv("QL_CACHE_FFN_IN", 0) == 1 and self.out_features == 6144 and self.in_features == 2048) or
        (getenv("QL_CACHE_FFN_DOWN", 0) == 1 and self.out_features == 2048 and self.in_features == 6144)
      )
    # Q4_0 packed-dot: read quantized blocks directly (3.56x less bandwidth than fp16 cache)
    if self.ggml_type == 2 and self.in_features % 32 == 0 and not use_cache:
      if getenv("QL_CUSTOM", 0) == 1:
        self._ensure_q4_0_blocks(x.device)
        x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
        x_flat = x_fp16.reshape(-1, self.in_features)
        out = Tensor.empty(x_flat.shape[0], self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
        out = Tensor.custom_kernel(out, x_flat, self._q4_0_blocks, fxn=custom_q4_0_linear)[0]
        return out.reshape(*x.shape[:-1], self.out_features)
      self._ensure_q4_0_separated(x.device)
      O, bpr = self.out_features, self.in_features // 32
      scale = self._q4_0_scale   # (O, bpr, 1) — contiguous fp16
      packed = self._q4_0_packed  # (O, bpr, 16) — contiguous uint8
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).reshape(*x.shape[:-1], O)
    self._ensure_dequant_cache(x.device)
    return x.linear(self._dequant_cache.T, None)

class QuantizedExpertWeights:
  __slots__ = ('blocks', 'num_experts', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_bytes_per_block', '_dequant_fn',
               'expert_first_in_memory', '_blocks_per_expert', '_expert_blocks', '_q4_0_scale', '_q4_0_packed', '_expert_dequant_cache')
  def __init__(self, blocks:Tensor, shape:tuple[int, int, int], ggml_type:int, expert_first_in_memory:bool=True):
    self.num_experts, self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, self._bytes_per_block, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self.blocks = blocks
    self.expert_first_in_memory = expert_first_in_memory
    self._blocks_per_expert = self.blocks.shape[0] // self.num_experts
    self._expert_blocks = None
    self._q4_0_scale = None
    self._q4_0_packed = None
    self._expert_dequant_cache = None
    assert self.blocks.shape[0] % self.num_experts == 0, f"blocks {self.blocks.shape[0]} not divisible by num_experts {self.num_experts}"

  def _ensure_expert_blocks(self, device: str|tuple[str, ...]) -> None:
    """Reshape blocks to (num_experts, blocks_per_expert, bytes_per_block) - just a view, no copy."""
    if self._expert_blocks is not None and self._expert_blocks.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    if self.expert_first_in_memory:
      self._expert_blocks = blocks.reshape(self.num_experts, self._blocks_per_expert, self._bytes_per_block)
    else:
      blocks_per_row = (self.in_features + self._el_per_block - 1) // self._el_per_block
      reshaped = blocks.reshape(self.out_features, blocks_per_row, self.num_experts, self._bytes_per_block)
      self._expert_blocks = reshaped.permute(2, 0, 1, 3).contiguous().reshape(self.num_experts, self._blocks_per_expert, self._bytes_per_block)

  def _ensure_q4_0_separated(self, device: str|tuple[str, ...]) -> None:
    """Pre-separate Q4_0 scale and packed data for better memory coalescing in matmul kernels."""
    if self._q4_0_scale is not None and self._q4_0_scale.device == device: return
    self._ensure_expert_blocks(device)
    O, bpr = self.out_features, self.in_features // 32
    eb_4d = self._expert_blocks.reshape(self.num_experts, O, bpr, 18)
    self._q4_0_scale = eb_4d[:, :, :, :2].bitcast(dtypes.float16).contiguous().realize()   # (E, O, bpr, 1)
    self._q4_0_packed = eb_4d[:, :, :, 2:].contiguous().realize()                          # (E, O, bpr, 16)

  def _ensure_dequant_expert_cache(self, device: str|tuple[str, ...]) -> None:
    if self._expert_dequant_cache is not None and self._expert_dequant_cache.device == device: return
    self._ensure_expert_blocks(device)
    w = self._dequant_fn(self._expert_blocks.reshape(-1, self._bytes_per_block))
    w = w.reshape(self.num_experts, self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast(dtypes.float16)
    self._expert_dequant_cache = w.contiguous().realize()

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    B, T, K = sel.shape
    n_sel = B * T * K
    if len(x.shape) == 3: x = x.reshape(B, T, 1, x.shape[-1])
    xk = x if x.shape[2] == K else x.expand(B, T, K, x.shape[-1])
    x_flat = xk.reshape(n_sel, self.in_features)

    # Q4_0 expert path via mul_mat_id-style primitive call.
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      self._ensure_q4_0_separated(x.device)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      sel_flat = sel.reshape(-1)
      out = Tensor.empty(n_sel, self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
      if getenv("QL_MOE_MSL", 0) == 1 and isinstance(x_fp16.device, str) and x_fp16.device.startswith("METAL"):
        from tinygrad.apps.q4_moe_msl import custom_q4_0_mul_mat_id_msl
        out = Tensor.mul_mat_id(out, x_fp16, self._q4_0_scale, self._q4_0_packed, sel_flat, fxn=custom_q4_0_mul_mat_id_msl)
      else:
        out = Tensor.mul_mat_id(out, x_fp16, self._q4_0_scale, self._q4_0_packed, sel_flat, fxn=custom_q4_0_mul_mat_id)
      return out.reshape(B, T, K, self.out_features)

    # Q5_K/Q6_K expert path: keep primitive contract (expert-id indexed matmul) with dequantized expert cache.
    if self.ggml_type in (13, 14):
      self._ensure_dequant_expert_cache(x.device)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      sel_flat = sel.reshape(-1)
      out = Tensor.empty(n_sel, self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
      out = Tensor.mul_mat_id(out, x_fp16, self._expert_dequant_cache, sel_flat, fxn=custom_fp16_mul_mat_id)
      return out.reshape(B, T, K, self.out_features)

    # Fallback: gather combined blocks then dequant
    self._ensure_expert_blocks(x.device)
    sel_blocks = self._expert_blocks[sel.reshape(-1)]  # (n_sel, bpe, bpb)
    w = self._dequant_fn(sel_blocks.reshape(-1, self._bytes_per_block)).reshape(n_sel, self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast(dtypes.float16)
    return (x_flat.reshape(n_sel, 1, self.in_features) @ w.transpose(-1, -2)).reshape(B, T, K, self.out_features)

def replace_quantized_modules(model, quantized_tensors: dict, state_dict: dict) -> tuple[int, int, int]:
  """Replace Linear/ExpertWeights modules with quantized versions. Returns (linear_replaced, expert_replaced, dequantized)."""
  q_replaced_linear, q_replaced_expert, q_dequant = 0, 0, 0
  for name in list(quantized_tensors.keys()):
    tensor_data = quantized_tensors[name]
    if len(tensor_data) == 4:
      blocks, shape, ggml_type, expert_first_in_memory = tensor_data
    else:
      blocks, shape, ggml_type = tensor_data
      expert_first_in_memory = True
    if not name.endswith('.weight') or 'attn_kv_b' in name: continue
    parts = name[:-7].split('.')
    obj, parent, attr, found = model, None, None, False
    for i, part in enumerate(parts):
      if part == 'blk': continue
      if parts[i-1] == 'blk' and part.isdigit(): obj = obj.blk[int(part)]
      elif hasattr(obj, part): parent, attr, obj = obj, part, getattr(obj, part)
      else: break
    else: found = True
    if found:
      if isinstance(obj, nn.Linear):
        del quantized_tensors[name]
        setattr(parent, attr, QuantizedLinear(blocks, shape, ggml_type))
        q_replaced_linear += 1
      elif type(obj).__name__ == 'ExpertWeights':
        del quantized_tensors[name]
        setattr(parent, attr, QuantizedExpertWeights(blocks, shape, ggml_type, expert_first_in_memory))
        q_replaced_expert += 1
      else: found = False
    if not found:
      del quantized_tensors[name]
      dequant_fn = GGML_QUANT_INFO[ggml_type][2]
      state_dict[name] = dequant_fn(blocks).reshape(*shape)
      state_dict[name] = state_dict[name].cast('float16') if getenv("HALF", 1) else state_dict[name]
      q_dequant += 1
  return q_replaced_linear, q_replaced_expert, q_dequant
