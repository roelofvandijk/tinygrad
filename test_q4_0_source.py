#!/usr/bin/env python3
"""Print generated Metal source for Q4_0 tensor DSL matvec."""
import os
os.environ["DEBUG"] = "5"
from tinygrad import Tensor
from tinygrad.dtype import dtypes

O, I = 10240, 2048
bpr = I // 32

blocks = Tensor.randint(O, bpr, 18, high=256, dtype=dtypes.uchar).realize()
x = Tensor.randn(I).cast(dtypes.float16).realize()

print("=" * 80)
print("Q4_0 KERNEL SOURCE")
print("=" * 80)
scale = blocks[:, :, :2].bitcast(dtypes.float16)
packed = blocks[:, :, 2:]
xp = x.reshape(1, bpr, 2, 16)
x_lo, x_hi = xp[:, :, 0, :], xp[:, :, 1, :]
lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
hi = packed.rshift(4).cast(dtypes.float16) - 8.0
result = (scale * (lo * x_lo + hi * x_hi)).reshape(O, bpr * 16).sum(axis=-1)
result.realize()
