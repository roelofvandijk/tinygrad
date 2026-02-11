#!/usr/bin/env python3
"""Proper GPU timing for Q4_0 matvec vs fp16 matvec with GPU sync."""
import time
from tinygrad import Tensor, TinyJit, Device
from tinygrad.dtype import dtypes

def bench_jit(fn, name, warmup=10, iters=50):
  for _ in range(warmup):
    fn().realize()
    Device["METAL"].synchronize()
  times = []
  for _ in range(iters):
    Device["METAL"].synchronize()
    s = time.perf_counter()
    fn().realize()
    Device["METAL"].synchronize()
    times.append(time.perf_counter() - s)
  times.sort()
  avg = sum(times[5:-5]) / (len(times) - 10)
  return avg

TESTS = [
  ("shexp_gate (22% of GLM)", 10240, 2048),
  ("expert_down (16%)",        2048, 10240),
  ("attn_output",              2048, 5120),
  ("attn_q_b",                 5120, 768),
]

for name, O, I in TESTS:
  bpr = I // 32
  weight_mb = O * bpr * 18 / 1e6

  blocks = Tensor.randint(O, bpr, 18, high=256, dtype=dtypes.uchar).realize()
  x = Tensor.randn(I).cast(dtypes.float16).realize()
  w = Tensor.randn(O, I).cast(dtypes.float16).realize()

  @TinyJit
  def q4_0_mv(blocks=blocks, x=x, bpr=bpr, O=O):
    scale = blocks[:, :, :2].bitcast(dtypes.float16)
    packed = blocks[:, :, 2:]
    xp = x.reshape(1, bpr, 2, 16)
    x_lo, x_hi = xp[:, :, 0, :], xp[:, :, 1, :]
    lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
    hi = packed.rshift(4).cast(dtypes.float16) - 8.0
    return (scale * (lo * x_lo + hi * x_hi)).reshape(O, bpr * 16).sum(axis=-1)

  @TinyJit
  def fp16_mv(x=x, w=w, I=I):
    return x.reshape(1, I) @ w.T

  t_q4 = bench_jit(q4_0_mv, "q4_0")
  t_fp16 = bench_jit(fp16_mv, "fp16")

  q4_bw = weight_mb / t_q4 / 1e3
  fp16_bw = O * I * 2 / t_fp16 / 1e9
  ratio = t_q4 / t_fp16

  print(f"{name}: O={O}, I={I}")
  print(f"  Q4_0:  {t_q4*1000:7.3f} ms  {q4_bw:6.1f} GB/s  ({weight_mb:.1f} MB)")
  print(f"  fp16:  {t_fp16*1000:7.3f} ms  {fp16_bw:6.1f} GB/s  ({O*I*2/1e6:.1f} MB)")
  print(f"  Q4_0 reads {O*I*2/1e6/weight_mb:.1f}x LESS data but takes {ratio:.2f}x the time")
  print(f"  Theoretical Q4_0 @ {fp16_bw:.0f} GB/s: {weight_mb/fp16_bw*1e3/1e3:.3f} ms")
  print()
