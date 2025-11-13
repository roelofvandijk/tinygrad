from __future__ import annotations
import weakref
from typing import Any, Callable
from tinygrad import Tensor
from tinygrad.uop.ops import GroupOp, Ops, UOp, sint
from tinygrad.helpers import canonicalize_strides, strides_for_shape, prod
from tinygrad.dtype import _from_torch_dtype

_PASSTHROUGH_OPS = {Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.MULTI}
def _is_view_op(op: Ops) -> bool: return op in GroupOp.Movement or op in _PASSTHROUGH_OPS
def _get_uop(x: UOp|Tensor) -> UOp: return x.uop if isinstance(x, Tensor) else x

def _movement_chain(uop: UOp|Tensor) -> list[tuple[Ops, Any]]:
  cur, chain = _get_uop(uop), []
  while _is_view_op(cur.op):
    if cur.op in GroupOp.Movement and cur.op is not Ops.PAD: chain.append((cur.op, cur.marg))
    if not cur.src: break
    cur = cur.src[0]
  return list(reversed(chain))

def canonical_base(view: Tensor) -> Tensor:
  seen = set()
  while hasattr(view, "_view_base") and id(view) not in seen:
    seen.add(id(view))
    view = view._view_base
  return view

def _apply_chain(base: Tensor, chain: list[tuple[Ops, Any]]) -> Tensor:
  for op, arg in chain:
    if op is Ops.FLIP: base = base.flip(tuple(i for i, f in enumerate(arg) if f))
    else: base = getattr(base, op.name.lower())(arg)
  return base

def update_shrink_region(tt:Tensor, updater:Callable[[Tensor], Tensor]) -> bool:
  if not hasattr(tt, '_view_base'): return False
  base, chain = canonical_base(tt), _movement_chain(tt)
  if chain and not all(op == Ops.SHRINK for op, _ in chain): return False
  if not base.uop.is_contiguous(): base.replace(base.contiguous())
  slices = [slice(None)] * len(base.shape)
  for op, arg in chain:
    for dim, (start, end) in enumerate(arg):
      slices[dim] = slice((slices[dim].start or 0) + start, (slices[dim].start or 0) + end)
  updated_base = base.clone()
  updated = updater(_apply_chain(base, chain))
  updated_base[tuple(slices)] = updated
  base.replace(updated_base)
  tt.replace(updated)
  return True

def _compute_strides(uop: UOp|Tensor) -> tuple[tuple[sint, ...], sint]:
  cur = _get_uop(uop)
  while _is_view_op(cur.op):
    if cur.op in GroupOp.Movement and cur.op is not Ops.PAD: break
    if not cur.src: break
    cur = cur.src[0]
  shape, strides, offset = cur.shape, strides_for_shape(cur.shape), 0
  for op, arg in _movement_chain(uop):
    match op:
      case Ops.RESHAPE: shape, strides = tuple(arg), strides_for_shape(arg)
      case Ops.EXPAND:
        new_strides, old_idx = [], 0
        for ns in arg:
          new_strides.append(strides[old_idx] if old_idx < len(shape) and ns == shape[old_idx] else 0)
          if old_idx < len(shape) and ns == shape[old_idx]: old_idx += 1
        shape, strides = tuple(arg), tuple(new_strides)
      case Ops.SHRINK:
        offset += sum(start * strides[i] for i, (start, _) in enumerate(arg) if i < len(strides))
        shape, strides = tuple(end - start for start, end in arg), tuple(strides[i] for i in range(len(arg)) if i < len(strides))
      case Ops.PERMUTE: shape, strides = tuple(shape[i] for i in arg if i < len(shape)), tuple(strides[i] for i in arg if i < len(strides))
      case Ops.FLIP:
        offset += sum((shape[i] - 1) * strides[i] for i, f in enumerate(arg) if f)
        strides = tuple(-strides[i] if arg[i] else strides[i] for i in range(len(arg)))
  return canonicalize_strides(shape, strides), offset

