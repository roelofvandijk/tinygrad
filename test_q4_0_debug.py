#!/usr/bin/env python3
"""Compare Metal source: tensor DSL vs custom_kernel for Q4_0."""
import os
os.environ["DEBUG"] = "5"
os.environ["METAL_XCODE"] = "0"

from tinygrad import Tensor, UOp
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

O, I = 2048, 5120
bpr = I // 32

blocks = Tensor.randint(O, bpr, 18, high=256, dtype=dtypes.uchar).realize()
x = Tensor.randn(1, I).cast(dtypes.float16).realize()
x_flat = x.reshape(I).realize()

print("="*80)
print("  TENSOR DSL KERNEL")
print("="*80)

# Tensor DSL path
scale = blocks[:, :, :2].bitcast(dtypes.float16)
packed = blocks[:, :, 2:]
x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16)
x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
result_dsl = (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1)
result_dsl.realize()

print("\n" + "="*80)
print("  CUSTOM KERNEL (raw, no opts)")
print("="*80)

def custom_q4_0(out:UOp, x:UOp, blocks:UOp) -> UOp:
  O = out.shape[0]
  I = x.shape[0]
  bpr = I // 32
  o = UOp.range(O, 0)
  r = UOp.range(bpr*16, 1, axis_type=AxisType.REDUCE)
  br, j = r // 16, r % 16
  acc = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
  acc = acc.after(o)[0].set(0.0)
  scale = (blocks[o, br, 0].cast(dtypes.ushort) + (blocks[o, br, 1].cast(dtypes.ushort) << 8)).bitcast(dtypes.half).cast(dtypes.float)
  q = blocks[o, br, j+2]
  q_lo = (q & 0xF).cast(dtypes.float) - 8.0
  q_hi = (q >> 4).cast(dtypes.float) - 8.0
  x_lo = x[br*32 + j].cast(dtypes.float)
  x_hi = x[br*32 + j + 16].cast(dtypes.float)
  acc = acc[0].set(acc.after(r)[0] + scale * (q_lo * x_lo + q_hi * x_hi), end=r)
  return out[o].store(acc[0].cast(out.dtype.base)).end(o).sink(
    arg=KernelInfo(name=f"q4_0_{O}_{I}", opts_to_apply=()))

out = Tensor.empty(O, dtype=dtypes.float16, device=x.device)
result_ck = Tensor.custom_kernel(out, x_flat, blocks, fxn=custom_q4_0)[0]
result_ck.realize()
