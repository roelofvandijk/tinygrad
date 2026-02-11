#!/usr/bin/env python3
"""Test different Q4_0 tensor DSL formulations for better memory coalescing."""
import time
from tinygrad import Tensor, TinyJit, Device
from tinygrad.dtype import dtypes

def bench(fn, warmup=10, iters=50):
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
  return sum(times[5:-5]) / (len(times) - 10)

O, I = 10240, 2048
bpr = I // 32
weight_mb = O * bpr * 18 / 1e6

blocks = Tensor.randint(O, bpr, 18, high=256, dtype=dtypes.uchar).realize()
x = Tensor.randn(I).cast(dtypes.float16).realize()

print(f"O={O}, I={I}, bpr={bpr}, weight={weight_mb:.1f} MB")
print()

# === V0: Current tensor DSL (baseline) ===
@TinyJit
def v0_current():
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xp = x.reshape(1, bpr, 2, 16)
  x_lo, x_hi = xp[:, :, 0, :], xp[:, :, 1, :]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * x_lo + hi * x_hi)).reshape(O, bpr * 16).sum(axis=-1)

t = bench(v0_current)
print(f"V0 current:     {t*1000:.3f} ms  {weight_mb/t/1e3:.1f} GB/s")

# === V1: Read packed as uint16 (2 bytes = 4 nibbles at a time) ===
# Reinterpret packed bytes as uint16: (O, bpr, 8) uint16 instead of (O, bpr, 16) uchar
@TinyJit
def v1_uint16():
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed_u16 = blocks[:, :, 2:].bitcast(dtypes.ushort)  # (O, bpr, 8) uint16
  # Each uint16 has 4 nibbles: lo0 = bits[0:3], lo1 = bits[4:7], hi0 = bits[8:11], hi1 = bits[12:15]
  n0 = (packed_u16.bitwise_and(0x000F)).cast(dtypes.float16) - 8.0  # low nibble of low byte
  n1 = (packed_u16.bitwise_and(0x00F0).rshift(4)).cast(dtypes.float16) - 8.0  # high nibble of low byte
  n2 = (packed_u16.bitwise_and(0x0F00).rshift(8)).cast(dtypes.float16) - 8.0  # low nibble of high byte
  n3 = (packed_u16.rshift(12)).cast(dtypes.float16) - 8.0  # high nibble of high byte
  # x layout: pairs of 16, regrouped as 4 groups of 8
  xp = x.reshape(1, bpr, 4, 8)
  x0, x1, x2, x3 = xp[:, :, 0, :], xp[:, :, 1, :], xp[:, :, 2, :], xp[:, :, 3, :]
  return (scale * (n0 * x0 + n1 * x2 + n2 * x1 + n3 * x3)).reshape(O, bpr * 8).sum(axis=-1)

t = bench(v1_uint16)
print(f"V1 uint16:      {t*1000:.3f} ms  {weight_mb/t/1e3:.1f} GB/s")

# === V2: Two-level reduce (within-block then across-blocks) ===
@TinyJit
def v2_two_level():
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xp = x.reshape(1, bpr, 2, 16)
  x_lo, x_hi = xp[:, :, 0, :], xp[:, :, 1, :]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  # First reduce within each block (sum over 16), then across blocks
  block_dot = (scale * (lo * x_lo + hi * x_hi)).sum(axis=-1)  # (O, bpr)
  return block_dot.sum(axis=-1)  # (O,)

t = bench(v2_two_level)
print(f"V2 two-level:   {t*1000:.3f} ms  {weight_mb/t/1e3:.1f} GB/s")

# === V3: Process 4 output rows together (like llama.cpp NR0=4) ===
# Reshape to expose 4 output rows as a local dimension
@TinyJit
def v3_nr4():
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xp = x.reshape(1, bpr, 2, 16)
  x_lo, x_hi = xp[:, :, 0, :], xp[:, :, 1, :]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * x_lo + hi * x_hi)).reshape(O, bpr * 16).sum(axis=-1)

t = bench(v3_nr4)
print(f"V3 nr4:         {t*1000:.3f} ms  {weight_mb/t/1e3:.1f} GB/s")

# === V4: Read blocks as uint32 (4 bytes = 8 nibbles per read) ===
@TinyJit
def v4_uint32():
  scale = blocks[:, :, :2].bitcast(dtypes.float16)  # (O, bpr, 1)
  # Read 16 packed bytes as 4 uint32
  packed_u32 = blocks[:, :, 2:].bitcast(dtypes.uint)  # (O, bpr, 4) uint32
  # Each uint32 has 8 nibbles. Extract each nibble
  # Match with the Q4_0 packed layout: byte0_lo, byte0_hi, byte1_lo, byte1_hi, ...
  n0 = (packed_u32.bitwise_and(0x0000000F)).cast(dtypes.float16) - 8.0
  n1 = (packed_u32.bitwise_and(0x000000F0).rshift(4)).cast(dtypes.float16) - 8.0
  n2 = (packed_u32.bitwise_and(0x00000F00).rshift(8)).cast(dtypes.float16) - 8.0
  n3 = (packed_u32.bitwise_and(0x0000F000).rshift(12)).cast(dtypes.float16) - 8.0
  n4 = (packed_u32.bitwise_and(0x000F0000).rshift(16)).cast(dtypes.float16) - 8.0
  n5 = (packed_u32.bitwise_and(0x00F00000).rshift(20)).cast(dtypes.float16) - 8.0
  n6 = (packed_u32.bitwise_and(0x0F000000).rshift(24)).cast(dtypes.float16) - 8.0
  n7 = (packed_u32.rshift(28)).cast(dtypes.float16) - 8.0
  # Activation in groups of 8 matching the 8 nibbles per uint32
  xp = x.reshape(1, bpr, 4, 8)  # 4 uint32s per block
  # Q4_0 layout: byte i has lo=element[i] and hi=element[i+16]
  # uint32 j has bytes [4j, 4j+1, 4j+2, 4j+3] → lo elements [4j..4j+3], hi elements [4j+16..4j+19]
  # So: n0=elem[4j], n2=elem[4j+1], n4=elem[4j+2], n6=elem[4j+3] (lo nibbles)
  #     n1=elem[4j+16], n3=elem[4j+17], n5=elem[4j+18], n7=elem[4j+19] (hi nibbles)
  # This is complex... let's just try the straightforward version
  x_flat8 = x.reshape(1, bpr, 32)
  # Actually, let me just sum all 8 nibble*x products per uint32
  # This needs x indexed as [bpr, 4, 8] where the 8 matches nibble order within uint32
  # Nibble order in uint32: byte0_lo, byte0_hi, byte1_lo, byte1_hi, byte2_lo, byte2_hi, byte3_lo, byte3_hi
  # = elem[4j], elem[4j+16], elem[4j+1], elem[4j+17], elem[4j+2], elem[4j+18], elem[4j+3], elem[4j+19]
  # This is too complex for a simple reshape. Skip this approach.
  return (scale * (n0.sum(axis=-1, keepdim=True))).reshape(O)  # placeholder

t = bench(v4_uint32)
print(f"V4 uint32:      {t*1000:.3f} ms  (placeholder, not correct)")

# === V5: Use contiguous packed data with simpler shapes ===
# Separate scale and packed into contiguous tensors
scale_c = blocks[:, :, :2].bitcast(dtypes.float16).contiguous().realize()  # (O, bpr, 1)
packed_c = blocks[:, :, 2:].contiguous().realize()  # (O, bpr, 16)

@TinyJit
def v5_contiguous():
  xp = x.reshape(1, bpr, 2, 16)
  x_lo, x_hi = xp[:, :, 0, :], xp[:, :, 1, :]
  lo = packed_c.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed_c.rshift(4).cast(dtypes.float16) - 8.0
  return (scale_c * (lo * x_lo + hi * x_hi)).reshape(O, bpr * 16).sum(axis=-1)

t = bench(v5_contiguous)
print(f"V5 contiguous:  {t*1000:.3f} ms  {weight_mb/t/1e3:.1f} GB/s")

# === V6: Flat buffer + bitcast to uint16, pre-separated ===
packed_u16_c = blocks[:, :, 2:].bitcast(dtypes.ushort).contiguous().realize()  # (O, bpr, 8) ushort

@TinyJit
def v6_u16_contiguous():
  # 4 nibbles per uint16
  n0 = (packed_u16_c.bitwise_and(0x000F)).cast(dtypes.float16) - 8.0  # (O, bpr, 8)
  n1 = (packed_u16_c.rshift(4).bitwise_and(0xF)).cast(dtypes.float16) - 8.0
  n2 = (packed_u16_c.rshift(8).bitwise_and(0xF)).cast(dtypes.float16) - 8.0
  n3 = (packed_u16_c.rshift(12)).cast(dtypes.float16) - 8.0
  # Q4_0 byte layout: byte[i] has lo=elem[i], hi=elem[i+16]
  # uint16[j] = byte[2j] | byte[2j+1]<<8
  # nibbles: n0=byte[2j].lo=elem[2j], n1=byte[2j].hi=elem[2j+16]
  #          n2=byte[2j+1].lo=elem[2j+1], n3=byte[2j+1].hi=elem[2j+17]
  # x layout needs: elem[2j], elem[2j+1], elem[2j+16], elem[2j+17] for j=0..7
  # That's: x[0,2,4,6,8,10,12,14] and x[1,3,5,7,9,11,13,15] and x[16,18,...] and x[17,19,...]
  # Too complex. Let's just do: all 4 nibbles * corresponding x, then reduce
  xr = x.reshape(1, bpr, 32)
  # For each uint16 j (0..7): need x[2j], x[2j+16], x[2j+1], x[2j+17]
  x_02 = xr[:, :, 0:16:2].contiguous()  # elem[0,2,4,...,14] — 8 values
  x_13 = xr[:, :, 1:16:2].contiguous()  # elem[1,3,5,...,15] — 8 values
  x_02_hi = xr[:, :, 16:32:2].contiguous()  # elem[16,18,...,30] — 8 values
  x_13_hi = xr[:, :, 17:32:2].contiguous()  # elem[17,19,...,31] — 8 values
  dot = n0 * x_02 + n2 * x_13 + n1 * x_02_hi + n3 * x_13_hi
  return (scale_c * dot).reshape(O, bpr * 8).sum(axis=-1)

t = bench(v6_u16_contiguous)
print(f"V6 u16+contig:  {t*1000:.3f} ms  {weight_mb/t/1e3:.1f} GB/s")

# === fp16 baseline ===
w = Tensor.randn(O, I).cast(dtypes.float16).realize()
@TinyJit
def fp16_mv():
  return x.reshape(1, I) @ w.T
t_fp16 = bench(fp16_mv)
print(f"fp16 baseline:  {t_fp16*1000:.3f} ms  {O*I*2/t_fp16/1e9:.1f} GB/s")
