from __future__ import annotations
import functools
from tinygrad import Device, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import all_int, getenv
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner

Q5K_LINEAR_MSL_TAG = "q5_k_linear_msl"
Q6K_LINEAR_MSL_TAG = "q6_k_linear_msl"

Q5K_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint N = %(N)d;
constant uint O = %(O)d;
constant uint I = %(I)d;
constant uint BPR = %(BPR)d;
constant uint NR = %(NR)d;

kernel void q5_k_linear(
  device half *out [[buffer(0)]],
  device const half *x [[buffer(1)]],
  device const uchar *blocks [[buffer(2)]],
  uint3 gid [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]]) {
  uint o_base = gid.x * NR;
  uint n = gid.y;
  if (n >= N) return;

  const device half *y = x + n*I;
  float sumf[NR];
#pragma unroll
  for (uint rr = 0; rr < NR; rr++) sumf[rr] = 0.0f;

  constexpr ushort kmask1 = 0x3f3f;
  constexpr ushort kmask2 = 0x0f0f;
  constexpr ushort kmask3 = 0xc0c0;

  const short tid = tiisg / 4;
  const short ix = tiisg %% 4;
  const short iq = tid / 4;
  const short ir = tid %% 4;

  const short l0 = 8 * ir;
  const short q_offset = 32 * iq + l0;
  const short y_offset = 64 * iq + l0;

  const uint8_t hm1 = uint8_t(1u << (2*iq));
  const uint8_t hm2 = uint8_t(hm1 << 1);
  const uint8_t hm3 = uint8_t(hm1 << 4);
  const uint8_t hm4 = uint8_t(hm2 << 4);

  float yl[16], yh[16];

  for (uint ib = ix; ib < BPR; ib += 4) {
    const uint ybase = ib*256 + y_offset;
    float4 sumy = float4(0.0f);
#pragma unroll
    for (short l = 0; l < 8; ++l) {
      yl[l+0] = float(y[ybase + l + 0]);   sumy[0] += yl[l+0];
      yl[l+8] = float(y[ybase + l + 32]);  sumy[1] += yl[l+8];
      yh[l+0] = float(y[ybase + l + 128]); sumy[2] += yh[l+0];
      yh[l+8] = float(y[ybase + l + 160]); sumy[3] += yh[l+8];
    }

#pragma unroll
    for (uint rr = 0; rr < NR; rr++) {
      uint o = o_base + rr;
      if (o >= O) continue;
      const device uchar *blk = blocks + (o*BPR + ib) * 176;
      const device half *dh = reinterpret_cast<const device half *>(blk);
      const device uchar *q1 = blk + 48 + q_offset;
      const device uchar *qh = blk + 16 + l0;
      const device ushort *a = reinterpret_cast<const device ushort *>(blk + 4) + iq;
      const device uchar *q2 = q1 + 64;

      ushort sc16[4];
      sc16[0] = a[0] & kmask1;
      sc16[1] = a[2] & kmask1;
      sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
      sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
      const thread uint8_t *sc8 = reinterpret_cast<const thread uint8_t *>(sc16);

      float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
#pragma unroll
      for (short l = 0; l < 8; ++l) {
        uint8_t h = qh[l];
        acc1[0] += yl[l+0] * float(q1[l] & 0x0F);
        acc1[1] += yl[l+8] * float(q1[l] & 0xF0);
        acc1[2] += yh[l+0] * float(q2[l] & 0x0F);
        acc1[3] += yh[l+8] * float(q2[l] & 0xF0);
        if (h & hm1) acc2[0] += yl[l+0];
        if (h & hm2) acc2[1] += yl[l+8];
        if (h & hm3) acc2[2] += yh[l+0];
        if (h & hm4) acc2[3] += yh[l+8];
      }

      float d = float(dh[0]), dmin = float(dh[1]);
      sumf[rr] += d * (float(sc8[0]) * (acc1[0]      + 16.0f*acc2[0]) +
                       float(sc8[1]) * (acc1[1]/16.0f + 16.0f*acc2[1]) +
                       float(sc8[4]) * (acc1[2]      + 16.0f*acc2[2]) +
                       float(sc8[5]) * (acc1[3]/16.0f + 16.0f*acc2[3])) -
                  dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
    }
  }

#pragma unroll
  for (uint rr = 0; rr < NR; rr++) {
    uint o = o_base + rr;
    if (o >= O) continue;
    float total = simd_sum(sumf[rr]);
    if (tiisg == 0) out[n*O + o] = half(total);
  }
}
"""

Q6K_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint N = %(N)d;
constant uint O = %(O)d;
constant uint I = %(I)d;
constant uint BPR = %(BPR)d;
constant uint NR = %(NR)d;

kernel void q6_k_linear(
  device half *out [[buffer(0)]],
  device const half *x [[buffer(1)]],
  device const uchar *blocks [[buffer(2)]],
  uint3 gid [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]]) {
  uint o_base = gid.x * NR;
  uint n = gid.y;
  if (n >= N) return;

  const device half *yrow = x + n*I;
  float sumf[NR];
#pragma unroll
  for (uint rr = 0; rr < NR; rr++) sumf[rr] = 0.0f;

  constexpr uint8_t kmask1 = 0x03;
  constexpr uint8_t kmask2 = 0x0C;
  constexpr uint8_t kmask3 = 0x30;
  constexpr uint8_t kmask4 = 0xC0;

  const short tid = tiisg / 2;
  const short ix = tiisg %% 2;
  const short ip = tid / 8;
  const short il = tid %% 8;
  const short l0 = 4 * il;
  const short is = 8 * ip + l0/16;

  const short y_offset = 128 * ip + l0;
  const short q_offset_l = 64 * ip + l0;
  const short q_offset_h = 32 * ip + l0;

  float yl[16];
  for (uint ib = ix; ib < BPR; ib += 2) {
    const uint ybase = ib*256 + y_offset;
    const device half *y = yrow + ybase;
#pragma unroll
    for (short l = 0; l < 4; ++l) {
      yl[4*l + 0] = float(y[l + 0]);
      yl[4*l + 1] = float(y[l + 32]);
      yl[4*l + 2] = float(y[l + 64]);
      yl[4*l + 3] = float(y[l + 96]);
    }
#pragma unroll
    for (uint rr = 0; rr < NR; rr++) {
      uint o = o_base + rr;
      if (o >= O) continue;
      const device uchar *blk = blocks + (o*BPR + ib) * 210;
      const device uchar *q1 = blk + q_offset_l;
      const device uchar *q2 = q1 + 32;
      const device uchar *qh = blk + 128 + q_offset_h;
      const device int8_t *sc = reinterpret_cast<const device int8_t *>(blk + 192) + is;
      const device half *dh = reinterpret_cast<const device half *>(blk + 208);

      float4 sums = float4(0.0f);
#pragma unroll
      for (short l = 0; l < 4; ++l) {
        sums[0] += yl[4*l + 0] * float(int((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
        sums[1] += yl[4*l + 1] * float(int((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
        sums[2] += yl[4*l + 2] * float(int((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
        sums[3] += yl[4*l + 3] * float(int((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);
      }

      float d = float(dh[0]);
      sumf[rr] += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    }
  }

#pragma unroll
  for (uint rr = 0; rr < NR; rr++) {
    uint o = o_base + rr;
    if (o >= O) continue;
    float total = simd_sum(sumf[rr]);
    if (tiisg == 0) out[n*O + o] = half(total);
  }
}
"""

class QKLinearMSLRunner(CompiledRunner):
  pass

def _threads_for_qk(ggml_type:int, out_features:int, in_features:int) -> int:
  base = getenv("QL_SHEXP_MSL_THREADS", 32)
  if ggml_type == 13 and out_features == 3072 and in_features == 2048:
    return getenv("QL_SHEXP_MSL_THREADS_3072_2048", base)
  if ggml_type == 14 and out_features == 2048 and in_features == 1536:
    return getenv("QL_SHEXP_MSL_THREADS_2048_1536", base)
  return base

def _nr_for_qk(ggml_type:int, out_features:int, in_features:int) -> int:
  base = getenv("QL_SHEXP_MSL_NR", 1)
  if ggml_type == 13 and out_features == 3072 and in_features == 2048:
    return getenv("QL_SHEXP_MSL_NR_3072_2048", base)
  if ggml_type == 14 and out_features == 2048 and in_features == 1536:
    return getenv("QL_SHEXP_MSL_NR_2048_1536", base)
  return base

@functools.lru_cache(maxsize=None)
def _get_q5k_runner(device:str, n_rows:int, out_features:int, in_features:int) -> QKLinearMSLRunner:
  assert in_features % 256 == 0 and out_features > 0 and n_rows > 0
  bpr = in_features // 256
  threads = _threads_for_qk(13, out_features, in_features)
  nr = _nr_for_qk(13, out_features, in_features)
  assert threads == 32, "q5_k msl kernel currently requires 32 threads"
  assert nr >= 1, "q5_k msl nr must be >= 1"
  src = Q5K_TEMPLATE % dict(N=n_rows, O=out_features, I=in_features, BPR=bpr, NR=nr)
  lib = Device[device].compiler.compile_cached(src)
  prg = Device[device].runtime("q5_k_linear", lib)
  p = ProgramSpec(
    name=f"q5_k_linear_msl_{n_rows}_{out_features}_{in_features}_nr{nr}",
    src=src, device=device, ast=UOp(Ops.NOOP), lib=lib,
    global_size=[(out_features + nr - 1) // nr, n_rows, 1], local_size=[threads, 1, 1],
    vars=[], globals=[0, 1, 2], outs=[0], ins=[1, 2],
  )
  return QKLinearMSLRunner(p, prg)

@functools.lru_cache(maxsize=None)
def _get_q6k_runner(device:str, n_rows:int, out_features:int, in_features:int) -> QKLinearMSLRunner:
  assert in_features % 256 == 0 and out_features > 0 and n_rows > 0
  bpr = in_features // 256
  threads = _threads_for_qk(14, out_features, in_features)
  nr = _nr_for_qk(14, out_features, in_features)
  assert threads == 32, "q6_k msl kernel currently requires 32 threads"
  assert nr >= 1, "q6_k msl nr must be >= 1"
  src = Q6K_TEMPLATE % dict(N=n_rows, O=out_features, I=in_features, BPR=bpr, NR=nr)
  lib = Device[device].compiler.compile_cached(src)
  prg = Device[device].runtime("q6_k_linear", lib)
  p = ProgramSpec(
    name=f"q6_k_linear_msl_{n_rows}_{out_features}_{in_features}_nr{nr}",
    src=src, device=device, ast=UOp(Ops.NOOP), lib=lib,
    global_size=[(out_features + nr - 1) // nr, n_rows, 1], local_size=[threads, 1, 1],
    vars=[], globals=[0, 1, 2], outs=[0], ins=[1, 2],
  )
  return QKLinearMSLRunner(p, prg)

def custom_q5_k_linear_msl(out:UOp, x:UOp, blocks:UOp) -> UOp:
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(blocks.shape) == 2
  assert all_int(out.shape + x.shape + blocks.shape), "q5_k_linear_msl requires static shapes"
  n_rows, out_features = out.shape
  in_features = x.shape[1]
  assert x.shape == (n_rows, in_features), "q5_k_linear_msl x shape mismatch"
  assert in_features % 256 == 0 and blocks.shape == (out_features * (in_features // 256), 176), "q5_k_linear_msl blocks mismatch"
  tag = UOp(Ops.CUSTOM, dtypes.void, arg=(Q5K_LINEAR_MSL_TAG, n_rows, out_features, in_features))
  return UOp(Ops.SINK, dtypes.void, src=(tag,), arg=KernelInfo(name=f"{Q5K_LINEAR_MSL_TAG}_{n_rows}_{out_features}_{in_features}"))

def custom_q6_k_linear_msl(out:UOp, x:UOp, blocks:UOp) -> UOp:
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(blocks.shape) == 2
  assert all_int(out.shape + x.shape + blocks.shape), "q6_k_linear_msl requires static shapes"
  n_rows, out_features = out.shape
  in_features = x.shape[1]
  assert x.shape == (n_rows, in_features), "q6_k_linear_msl x shape mismatch"
  assert in_features % 256 == 0 and blocks.shape == (out_features * (in_features // 256), 210), "q6_k_linear_msl blocks mismatch"
  tag = UOp(Ops.CUSTOM, dtypes.void, arg=(Q6K_LINEAR_MSL_TAG, n_rows, out_features, in_features))
  return UOp(Ops.SINK, dtypes.void, src=(tag,), arg=KernelInfo(name=f"{Q6K_LINEAR_MSL_TAG}_{n_rows}_{out_features}_{in_features}"))

def lower_qk_linear_msl_ast(ast:UOp, device:str) -> QKLinearMSLRunner|None:
  if ast.op is not Ops.SINK or len(ast.src) != 1: return None
  tag = ast.src[0]
  if tag.op is not Ops.CUSTOM or not isinstance(tag.arg, tuple) or len(tag.arg) != 4: return None
  if not isinstance(device, str) or not device.startswith("METAL"): return None
  name, n_rows, out_features, in_features = tag.arg
  assert all_int((n_rows, out_features, in_features)), "qk_linear_msl lowering requires static shapes"
  if name == Q5K_LINEAR_MSL_TAG: return _get_q5k_runner(device, int(n_rows), int(out_features), int(in_features))
  if name == Q6K_LINEAR_MSL_TAG: return _get_q6k_runner(device, int(n_rows), int(out_features), int(in_features))
  return None
