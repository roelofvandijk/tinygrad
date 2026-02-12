from __future__ import annotations
import functools
from tinygrad import Device, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import all_int
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner

Q4_LINEAR_MSL_TAG = "q4_0_linear_msl"

MSL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint N = %(N)d;
constant uint O = %(O)d;
constant uint I = %(I)d;
constant uint BPR = %(BPR)d;
constant uint NR = %(NR)d;
constant uint THREADS = 32;

kernel void q4_0_linear(
  device half *out [[buffer(0)]],
  device const half *x [[buffer(1)]],
  device const half *scale [[buffer(2)]],
  device const uchar *packed [[buffer(3)]],
  uint3 gid [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]]) {
  uint og = gid.x * NR;
  uint n = gid.y;
  if (n >= N) return;

  const device half *xrow = x + n*I;
  float acc[NR];
#pragma unroll
  for (uint rr = 0; rr < NR; rr++) acc[rr] = 0.0f;

  for (uint br = tiisg; br < BPR; br += THREADS) {
    float s[NR], s16[NR], s256[NR], s4096[NR], md[NR];
    const device uchar *pb[NR];
#pragma unroll
    for (uint rr = 0; rr < NR; rr++) {
      uint o = og + rr;
      if (o < O) {
        const device half *srow = scale + o * BPR;
        const device uchar *prow = packed + o * BPR * 16;
        s[rr] = float(srow[br]);
        s16[rr] = s[rr] * (1.0f/16.0f);
        s256[rr] = s[rr] * (1.0f/256.0f);
        s4096[rr] = s[rr] * (1.0f/4096.0f);
        md[rr] = -8.0f * s[rr];
        pb[rr] = prow + br*16;
      } else {
        s[rr] = s16[rr] = s256[rr] = s4096[rr] = md[rr] = 0.0f;
        pb[rr] = packed;
      }
    }

#pragma unroll
    for (uint j = 0; j < 16; j += 2) {
      uint xb = br*32 + j;
      float x0 = float(xrow[xb]);
      float x1 = float(xrow[xb + 16]);
      float x2 = float(xrow[xb + 1]);
      float x3 = float(xrow[xb + 17]);
#pragma unroll
      for (uint rr = 0; rr < NR; rr++) {
        ushort q = *reinterpret_cast<const device ushort *>(pb[rr] + j);
        float q0 = s[rr] * float(q & 0x000F) + md[rr];
        float q1 = s16[rr] * float(q & 0x00F0) + md[rr];
        float q2 = s256[rr] * float(q & 0x0F00) + md[rr];
        float q3 = s4096[rr] * float(q & 0xF000) + md[rr];
        acc[rr] += q0*x0 + q1*x1 + q2*x2 + q3*x3;
      }
    }
  }

#pragma unroll
  for (uint rr = 0; rr < NR; rr++) {
    uint o = og + rr;
    if (o >= O) continue;
    float total = simd_sum(acc[rr]);
    if (tiisg == 0) out[n*O + o] = half(total);
  }
}
"""

class Q4LinearMSLRunner(CompiledRunner):
  pass

def _nr_for_q4_linear(out_features:int, in_features:int) -> int:
  if out_features == 6144 and in_features == 2048: return 2
  if out_features == 3072 and in_features == 1536: return 2
  return 1

@functools.lru_cache(maxsize=None)
def _get_q4_linear_runner(device:str, n_rows:int, out_features:int, in_features:int) -> Q4LinearMSLRunner:
  assert in_features % 32 == 0 and out_features > 0 and n_rows > 0
  bpr = in_features // 32
  nr = _nr_for_q4_linear(out_features, in_features)
  src = MSL_TEMPLATE % dict(N=n_rows, O=out_features, I=in_features, BPR=bpr, NR=nr)
  lib = Device[device].compiler.compile_cached(src)
  prg = Device[device].runtime("q4_0_linear", lib)
  p = ProgramSpec(
    name=f"q4_0_linear_msl_{n_rows}_{out_features}_{in_features}_nr{nr}",
    src=src, device=device, ast=UOp(Ops.NOOP), lib=lib,
    global_size=[(out_features + nr - 1) // nr, n_rows, 1], local_size=[32, 1, 1],
    vars=[], globals=[0, 1, 2, 3], outs=[0], ins=[1, 2, 3],
  )
  return Q4LinearMSLRunner(p, prg)

def custom_q4_0_linear_msl(out:UOp, x:UOp, scale:UOp, packed:UOp) -> UOp:
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(scale.shape) == 3 and len(packed.shape) == 3
  assert all_int(out.shape + x.shape + scale.shape + packed.shape), "q4_0_linear_msl requires static shapes"
  n_rows, out_features = out.shape
  in_features = x.shape[1]
  bpr = in_features // 32
  assert x.shape == (n_rows, in_features), "q4_0_linear_msl x shape mismatch"
  assert scale.shape == (out_features, bpr, 1), "q4_0_linear_msl scale shape mismatch"
  assert packed.shape == (out_features, bpr, 16), "q4_0_linear_msl packed shape mismatch"
  tag = UOp(Ops.CUSTOM, dtypes.void, arg=(Q4_LINEAR_MSL_TAG, n_rows, out_features, in_features))
  return UOp(Ops.SINK, dtypes.void, src=(tag,), arg=KernelInfo(name=f"{Q4_LINEAR_MSL_TAG}_{n_rows}_{out_features}_{in_features}"))

def lower_q4_linear_msl_ast(ast:UOp, device:str) -> Q4LinearMSLRunner|None:
  if ast.op is not Ops.SINK or len(ast.src) != 1: return None
  tag = ast.src[0]
  if tag.op is not Ops.CUSTOM or not isinstance(tag.arg, tuple) or len(tag.arg) != 4: return None
  if not isinstance(device, str) or not device.startswith("METAL"): return None
  name, n_rows, out_features, in_features = tag.arg
  if name != Q4_LINEAR_MSL_TAG: return None
  assert all_int((n_rows, out_features, in_features)), "q4_0_linear_msl lowering requires static shapes"
  return _get_q4_linear_runner(device, int(n_rows), int(out_features), int(in_features))
