#!/usr/bin/env python3
"""40 Q4_0 kernel variants with different fusion breaks. BEAM each one."""
import sys
from tinygrad import Tensor
from tinygrad.dtype import dtypes

def setup(O=10240, I=2048):
  bpr = I // 32
  blocks = Tensor.randint(O, bpr, 18, high=256, dtype=dtypes.uchar).realize()
  x = Tensor.randn(1, I).cast(dtypes.float16).realize()
  return blocks, x, O, I, bpr

# V0: current (fused, cast fp16)
def v0(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V1: no cast (let scheduler decide types)
def v1(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF) - 8.0
  hi = packed.rshift(4) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V2: contiguous scale and packed
def v2(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).contiguous()
  packed = blocks[:, :, 2:].contiguous()
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V3: contiguous after nibble dequant
def v3(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0).contiguous()
  hi = (packed.rshift(4).cast(dtypes.float16) - 8.0).contiguous()
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V4: two-level reduce (within-block 16, then across bpr)
def v4(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).sum(axis=-1).sum(axis=-1)

# V5: factored -8 (llama.cpp style): scale*(raw_dot - 8*x_sum)
def v5(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo_raw = packed.bitwise_and(0xF).cast(dtypes.float16)
  hi_raw = packed.rshift(4).cast(dtypes.float16)
  x_sum = (xr[:, 0, :] + xr[:, 1, :]).sum(axis=-1, keepdim=True)
  dot_raw = lo_raw * xr[:, 0, :] + hi_raw * xr[:, 1, :]
  return (scale * (dot_raw - 8.0 * x_sum)).reshape(O, bpr * 16).sum(axis=-1)

# V6: uint16 reads (4 nibbles per read)
def v6(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  p16 = blocks[:, :, 2:].bitcast(dtypes.ushort)  # (O, bpr, 8)
  xr = x.reshape(bpr, 4, 8)
  n0 = (p16.bitwise_and(0x000F)).cast(dtypes.float16) - 8.0
  n1 = (p16.rshift(4).bitwise_and(0xF)).cast(dtypes.float16) - 8.0
  n2 = (p16.rshift(8).bitwise_and(0xF)).cast(dtypes.float16) - 8.0
  n3 = (p16.rshift(12)).cast(dtypes.float16) - 8.0
  return (scale * (n0*xr[:,0,:] + n2*xr[:,1,:] + n1*xr[:,2,:] + n3*xr[:,3,:])).reshape(O, bpr*8).sum(axis=-1)

# V7: all float32
def v7(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).cast(dtypes.float)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float) - 8.0
  hi = packed.rshift(4).cast(dtypes.float) - 8.0
  return (scale * (lo * xr[:, 0, :].cast(dtypes.float) + hi * xr[:, 1, :].cast(dtypes.float))).reshape(O, bpr*16).sum(axis=-1).cast(dtypes.float16)

# V8: contiguous dot before scale
def v8(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  dot = (lo * xr[:, 0, :] + hi * xr[:, 1, :]).contiguous()
  return (scale * dot).reshape(O, bpr * 16).sum(axis=-1)

# V9: contiguous lo*x and hi*x separately
def v9(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo_x = ((packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0) * xr[:, 0, :]).contiguous()
  hi_x = ((packed.rshift(4).cast(dtypes.float16) - 8.0) * xr[:, 1, :]).contiguous()
  return (scale * (lo_x + hi_x)).reshape(O, bpr * 16).sum(axis=-1)

# V10: cat lo,hi into (O,bpr,32), single dot with x (bpr,32)
def v10(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  w = Tensor.cat(lo, hi, dim=-1)
  return (scale * (w * x.reshape(bpr, 32))).reshape(O, bpr * 32).sum(axis=-1)

# V11: contiguous after scale*dot, reduce separate
def v11(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  scaled = (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).contiguous()
  return scaled.reshape(O, bpr * 16).sum(axis=-1)

# V12: original with batch dim
def v12(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xp = x.reshape(-1, 1, bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xp[:,:,:,0,:] + hi * xp[:,:,:,1,:])).reshape(-1, O, bpr*16).sum(axis=-1)

# V13: scale at end: reduce dot first, then multiply scale
def v13(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  dot = (lo * xr[:, 0, :] + hi * xr[:, 1, :]).sum(axis=-1)  # (O, bpr)
  return (scale.reshape(O, bpr) * dot).sum(axis=-1)

# V14: pre-realize scale
def v14(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).realize()
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V15: split lo and hi into separate kernels, add
def v15(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo_r = (scale * ((packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0) * xr[:, 0, :])).reshape(O, bpr*16).sum(axis=-1).contiguous()
  hi_r = (scale * ((packed.rshift(4).cast(dtypes.float16) - 8.0) * xr[:, 1, :])).reshape(O, bpr*16).sum(axis=-1).contiguous()
  return lo_r + hi_r

# V16: reshape x as (bpr, 32) — flat within block
def v16(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 32)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, :16] + hi * xr[:, 16:])).reshape(O, bpr * 16).sum(axis=-1)

# V17: pre-realize packed bytes
def v17(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:].contiguous().realize()
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V18: factored -8 with two-level reduce
def v18(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo_raw = packed.bitwise_and(0xF).cast(dtypes.float16)
  hi_raw = packed.rshift(4).cast(dtypes.float16)
  x_sum = (xr[:, 0, :] + xr[:, 1, :]).sum(axis=-1, keepdim=True)
  dot = (lo_raw * xr[:, 0, :] + hi_raw * xr[:, 1, :]).sum(axis=-1)  # (O, bpr)
  return (scale.reshape(O, bpr) * (dot - 8.0 * x_sum.reshape(bpr))).sum(axis=-1)

# V19: contiguous on everything, then fused reduce
def v19(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).contiguous().realize()
  packed = blocks[:, :, 2:].contiguous().realize()
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V20: NR=2 — group 2 output rows together
def v20(blocks, x, O, I, bpr):
  b = blocks.reshape(O//2, 2, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O//2, 2, bpr*16).sum(axis=-1).reshape(O)

# V21: NR=4 — group 4 output rows (like llama.cpp N_R0_Q4_0=4)
def v21(blocks, x, O, I, bpr):
  b = blocks.reshape(O//4, 4, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O//4, 4, bpr*16).sum(axis=-1).reshape(O)

# V22: NR=8
def v22(blocks, x, O, I, bpr):
  b = blocks.reshape(O//8, 8, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O//8, 8, bpr*16).sum(axis=-1).reshape(O)

# V23: NR=16
def v23(blocks, x, O, I, bpr):
  b = blocks.reshape(O//16, 16, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O//16, 16, bpr*16).sum(axis=-1).reshape(O)

# V24: NR=4 + two-level reduce (16 within block, then bpr across)
def v24(blocks, x, O, I, bpr):
  b = blocks.reshape(O//4, 4, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  block_dot = (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).sum(axis=-1)
  return block_dot.sum(axis=-1).reshape(O)

# V25: transposed blocks (bpr, O, 18) — adjacent threads read same block pos, different O
def v25(blocks, x, O, I, bpr):
  bt = blocks.permute(1, 0, 2).contiguous()
  scale = bt[:, :, :2].bitcast(dtypes.float16)
  packed = bt[:, :, 2:]
  xr = x.reshape(bpr, 1, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * xr[:, :, 0, :] + hi * xr[:, :, 1, :])).sum(axis=-1).sum(axis=0)

# V26: pre-dequanted lo/hi as separate fp16 buffers (3 inputs, 1 fused kernel)
def v26(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  lo = (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0).contiguous().realize()
  hi = (packed.rshift(4).cast(dtypes.float16) - 8.0).contiguous().realize()
  xr = x.reshape(bpr, 2, 16)
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr*16).sum(axis=-1)

# V27: uint16 reads + factored -8 (llama.cpp inspired, avoids subtract per element)
def v27(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  p16 = blocks[:, :, 2:].bitcast(dtypes.ushort)
  xr = x.reshape(bpr, 4, 8)
  n0 = p16.bitwise_and(0x000F).cast(dtypes.float16)
  n1 = p16.rshift(4).bitwise_and(0xF).cast(dtypes.float16)
  n2 = p16.rshift(8).bitwise_and(0xF).cast(dtypes.float16)
  n3 = p16.rshift(12).cast(dtypes.float16)
  x_per_u16 = xr[:, 0, :] + xr[:, 1, :] + xr[:, 2, :] + xr[:, 3, :]
  dot = n0*xr[:,0,:] + n2*xr[:,1,:] + n1*xr[:,2,:] + n3*xr[:,3,:]
  return (scale * (dot - 8.0 * x_per_u16)).reshape(O, bpr*8).sum(axis=-1)

# V28: fp32 accumulation (cast products to fp32 before reduce)
def v28(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  prod = (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).cast(dtypes.float)
  return prod.reshape(O, bpr * 16).sum(axis=-1).cast(dtypes.float16)

# V29: fp32 throughout — dequant + dot + reduce all in fp32
def v29(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).cast(dtypes.float)
  packed = blocks[:, :, 2:]
  xr = x.cast(dtypes.float).reshape(bpr, 2, 16)
  lo_raw = packed.bitwise_and(0xF).cast(dtypes.float)
  hi_raw = packed.rshift(4).cast(dtypes.float)
  x_sum = (xr[:, 0, :] + xr[:, 1, :]).sum(axis=-1, keepdim=True)
  dot = (lo_raw * xr[:, 0, :] + hi_raw * xr[:, 1, :]).sum(axis=-1)
  return (scale.reshape(O, bpr) * (dot - 8.0 * x_sum.reshape(bpr))).sum(axis=-1).cast(dtypes.float16)

# V30: dequant to fp16 then standard matmul (2 separate kernels)
def v30(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  w = (scale * Tensor.cat(lo, hi, dim=-1)).reshape(O, I).contiguous()
  return x @ w.T

# V31: flat dequant-matvec — single reduce over I (no block structure in reduce)
def v31(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  w = (scale * Tensor.cat(lo, hi, dim=-1)).reshape(O, I)
  return (w * x.reshape(I)).sum(axis=-1)

# V32: NR=4 + fp32 accumulation
def v32(blocks, x, O, I, bpr):
  b = blocks.reshape(O//4, 4, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  prod = (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).cast(dtypes.float)
  return prod.reshape(O//4, 4, bpr*16).sum(axis=-1).reshape(O).cast(dtypes.float16)

# V33: NR=4 + uint16 reads
def v33(blocks, x, O, I, bpr):
  b = blocks.reshape(O//4, 4, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16)
  p16 = b[:, :, :, 2:].bitcast(dtypes.ushort)
  xr = x.reshape(bpr, 4, 8)
  n0 = p16.bitwise_and(0x000F).cast(dtypes.float16) - 8.0
  n1 = p16.rshift(4).bitwise_and(0xF).cast(dtypes.float16) - 8.0
  n2 = p16.rshift(8).bitwise_and(0xF).cast(dtypes.float16) - 8.0
  n3 = p16.rshift(12).cast(dtypes.float16) - 8.0
  return (scale * (n0*xr[:,0,:] + n2*xr[:,1,:] + n1*xr[:,2,:] + n3*xr[:,3,:])).reshape(O//4, 4, bpr*8).sum(axis=-1).reshape(O)

# V34: separate lo/hi reduces with fp32 accum, add at end
def v34(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo_prod = (scale * ((packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0) * xr[:, 0, :])).cast(dtypes.float)
  hi_prod = (scale * ((packed.rshift(4).cast(dtypes.float16) - 8.0) * xr[:, 1, :])).cast(dtypes.float)
  lo_r = lo_prod.reshape(O, bpr*16).sum(axis=-1)
  hi_r = hi_prod.reshape(O, bpr*16).sum(axis=-1)
  return (lo_r + hi_r).cast(dtypes.float16)

# V35: reduce dot within block first (sum 16), then scale, then sum bpr
def v35(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).reshape(O, bpr)
  packed = blocks[:, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  dot = (lo * xr[:, 0, :] + hi * xr[:, 1, :]).sum(axis=-1)
  return (scale * dot).sum(axis=-1)

# V36: contiguous dequant weights, then fused scale+reduce
def v36(blocks, x, O, I, bpr):
  packed = blocks[:, :, 2:]
  lo = (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0).contiguous()
  hi = (packed.rshift(4).cast(dtypes.float16) - 8.0).contiguous()
  scale = blocks[:, :, :2].bitcast(dtypes.float16)
  xr = x.reshape(bpr, 2, 16)
  return (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).reshape(O, bpr * 16).sum(axis=-1)

# V37: uint16 + two-level reduce (sum 8 per block, then sum bpr)
def v37(blocks, x, O, I, bpr):
  scale = blocks[:, :, :2].bitcast(dtypes.float16).reshape(O, bpr)
  p16 = blocks[:, :, 2:].bitcast(dtypes.ushort)
  xr = x.reshape(bpr, 4, 8)
  n0 = p16.bitwise_and(0x000F).cast(dtypes.float16) - 8.0
  n1 = p16.rshift(4).bitwise_and(0xF).cast(dtypes.float16) - 8.0
  n2 = p16.rshift(8).bitwise_and(0xF).cast(dtypes.float16) - 8.0
  n3 = p16.rshift(12).cast(dtypes.float16) - 8.0
  block_dot = (n0*xr[:,0,:] + n2*xr[:,1,:] + n1*xr[:,2,:] + n3*xr[:,3,:]).sum(axis=-1)
  return (scale * block_dot).sum(axis=-1)

# V38: NR=4 + factored -8 + two-level reduce
def v38(blocks, x, O, I, bpr):
  b = blocks.reshape(O//4, 4, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16).reshape(O//4, 4, bpr)
  packed = b[:, :, :, 2:]
  xr = x.reshape(bpr, 2, 16)
  lo_raw = packed.bitwise_and(0xF).cast(dtypes.float16)
  hi_raw = packed.rshift(4).cast(dtypes.float16)
  x_sum = (xr[:, 0, :] + xr[:, 1, :]).sum(axis=-1, keepdim=True)
  dot = (lo_raw * xr[:, 0, :] + hi_raw * xr[:, 1, :]).sum(axis=-1)
  return (scale * (dot - 8.0 * x_sum.reshape(bpr))).sum(axis=-1).reshape(O)

# V39: NR=4 + pre-separated contiguous buffers + fp32 accum
def v39(blocks, x, O, I, bpr):
  b = blocks.reshape(O//4, 4, bpr, 18)
  scale = b[:, :, :, :2].bitcast(dtypes.float16).contiguous()
  packed = b[:, :, :, 2:].contiguous()
  xr = x.reshape(bpr, 2, 16)
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  prod = (scale * (lo * xr[:, 0, :] + hi * xr[:, 1, :])).cast(dtypes.float)
  return prod.reshape(O//4, 4, bpr*16).sum(axis=-1).reshape(O).cast(dtypes.float16)

VERSIONS = {f"v{i}": fn for i, fn in enumerate([
  v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,
  v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,
])}

if __name__ == "__main__":
  which = sys.argv[1] if len(sys.argv) > 1 else "v0"
  fn = VERSIONS[which]
  args = setup()
  fn(*args).realize()
