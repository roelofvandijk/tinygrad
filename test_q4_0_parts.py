#!/usr/bin/env python3
"""Decompose Q4_0 matvec into atomic parts. Benchmark each. Find the slow one."""
import time
from tinygrad import Tensor
from tinygrad.dtype import dtypes

def bench(fn, name, warmup=5, iters=30):
  for _ in range(warmup): fn().realize()
  times = []
  for _ in range(iters):
    s = time.perf_counter()
    fn().realize()
    times.append(time.perf_counter() - s)
  times.sort()
  avg = sum(times[3:-3]) / (len(times) - 6)
  print(f"  {name:40s}: {avg*1000:7.3f} ms")
  return avg

# GLM shared expert dimensions (the 22% bottleneck)
O, I = 10240, 2048
bpr = I // 32  # 64 blocks per row

print(f"Q4_0 matvec decomposition: O={O}, I={I}, bpr={bpr}")
print(f"Weight size: {O * bpr * 18 / 1e6:.1f} MB")
print(f"Activation size: {I * 2 / 1e3:.1f} KB (fp16)")
print()

# Create data
blocks = Tensor.randint(O, bpr, 18, high=256, dtype=dtypes.uchar).realize()
x = Tensor.randn(I).cast(dtypes.float16).realize()

# Pre-slice the block parts
scale_bytes = blocks[:, :, :2].contiguous().realize()   # (O, bpr, 2) uchar
packed = blocks[:, :, 2:].contiguous().realize()        # (O, bpr, 16) uchar

print("=" * 60)
print("Part 1: Raw memory reads (no compute)")
print("=" * 60)

# Just read the weight bytes and sum (measures memory bandwidth)
bench(lambda: blocks.cast(dtypes.float16).sum(), "read all blocks + sum (fp16)")
bench(lambda: packed.cast(dtypes.float16).sum(), "read packed only + sum (fp16)")

print()
print("=" * 60)
print("Part 2: Nibble extraction only")
print("=" * 60)

# Extract nibbles (the bitwise ops)
bench(lambda: packed.bitwise_and(0xF), "lo nibbles (AND 0xF)")
bench(lambda: packed.rshift(4), "hi nibbles (>> 4)")
bench(lambda: (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0), "lo dequant (AND+cast+sub)")
bench(lambda: (packed.rshift(4).cast(dtypes.float16) - 8.0), "hi dequant (SHR+cast+sub)")
# Both together
bench(lambda: Tensor.stack(packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0,
                           packed.rshift(4).cast(dtypes.float16) - 8.0).sum(),
      "both nibbles dequant + reduce")

print()
print("=" * 60)
print("Part 3: Scale load + bitcast")
print("=" * 60)

bench(lambda: scale_bytes.bitcast(dtypes.float16), "scale bitcast to fp16")
# Reconstruct from bytes (what custom kernel does)
bench(lambda: (scale_bytes[:,:,0].cast(dtypes.ushort) + (scale_bytes[:,:,1].cast(dtypes.ushort) << 8)).bitcast(dtypes.half),
      "scale from 2 bytes (ushort + shift)")

print()
print("=" * 60)
print("Part 4: Dot product components")
print("=" * 60)

# Create pre-dequantized weights (fp16)
lo = (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0).realize()  # (O, bpr, 16)
hi = (packed.rshift(4).cast(dtypes.float16) - 8.0).realize()         # (O, bpr, 16)
scale = scale_bytes.bitcast(dtypes.float16).realize()                 # (O, bpr, 1)

# Activation slices
x_lo = x.reshape(1, bpr, 2, 16)[:, :, 0, :].contiguous().realize()  # (1, bpr, 16)
x_hi = x.reshape(1, bpr, 2, 16)[:, :, 1, :].contiguous().realize()  # (1, bpr, 16)

# Element-wise multiply (no reduce yet)
bench(lambda: lo * x_lo, "lo * x_lo (elementwise)")
bench(lambda: lo * x_lo + hi * x_hi, "lo*x_lo + hi*x_hi (elementwise)")
bench(lambda: scale * (lo * x_lo + hi * x_hi), "scale*(lo*x_lo + hi*x_hi) (elementwise)")

print()
print("=" * 60)
print("Part 5: Reduction")
print("=" * 60)

# Pre-compute the per-block products
products = (scale * (lo * x_lo + hi * x_hi)).realize()  # (O, bpr, 16)

bench(lambda: products.sum(axis=-1), "reduce over 16 (within-block)")
bench(lambda: products.reshape(O, bpr*16).sum(axis=-1), "reduce over bpr*16 (flat)")

print()
print("=" * 60)
print("Part 6: Full Q4_0 matvec (tensor DSL)")
print("=" * 60)

def tensor_dsl():
  s = blocks[:, :, :2].bitcast(dtypes.float16)
  p = blocks[:, :, 2:]
  xp = x.reshape(1, bpr, 2, 16)
  xl, xh = xp[:, :, 0, :], xp[:, :, 1, :]
  l = p.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  h = p.rshift(4).cast(dtypes.float16) - 8.0
  return (s * (l * xl + h * xh)).reshape(O, bpr * 16).sum(axis=-1)

t_full = bench(tensor_dsl, "FULL tensor DSL matvec")
bw_full = O * bpr * 18 / t_full / 1e9
print(f"  {'':40s}  → {bw_full:.1f} GB/s")

print()
print("=" * 60)
print("Part 7: fp16 matvec baseline (dequant cache)")
print("=" * 60)

# Compare: if we had fp16 weights pre-dequantized
w_fp16 = Tensor.randn(O, I).cast(dtypes.float16).realize()
bench(lambda: x.reshape(1, I) @ w_fp16.T, "fp16 matvec (x @ W.T)")
t_fp16 = bench(lambda: x.cast(dtypes.float16).reshape(1, I).linear(w_fp16.T, None), "fp16 linear")
bw_fp16 = O * I * 2 / t_fp16 / 1e9
print(f"  {'':40s}  → {bw_fp16:.1f} GB/s (fp16 weights)")

print()
print("=" * 60)
print("Summary")
print("=" * 60)
