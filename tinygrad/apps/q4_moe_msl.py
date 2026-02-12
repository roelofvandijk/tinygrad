from __future__ import annotations
import functools
from tinygrad import Device, Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.ops import KernelInfo
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner, ExecItem, capturing, CAPTURING
from tinygrad.helpers import all_int

MSL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint N = %(N)d;
constant uint O = %(O)d;
constant uint I = %(I)d;
constant uint BPR = %(BPR)d;
constant uint E = %(E)d;
constant uint THREADS = %(THREADS)d;
constant uint NSG = %(NSG)d;

kernel void q4_moe_mul_mat_id(
  device half *out [[buffer(0)]],
  device const half *x [[buffer(1)]],
  device const half *scale [[buffer(2)]],
  device const uchar *packed [[buffer(3)]],
  device const int *sel [[buffer(4)]],
  uint3 gid [[threadgroup_position_in_grid]],
  uint3 lid [[thread_position_in_threadgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]]) {
  uint o = gid.x;
  uint n = gid.y;
  uint tid = lid.x;

  if (o >= O || n >= N || tid >= THREADS) return;

  int ei = sel[n];
  ei = min(max(ei, 0), int(E-1));

  const device half *xrow = x + n*I;
  const device half *srow = scale + (uint(ei)*O + o) * BPR;
  const device uchar *prow = packed + (uint(ei)*O + o) * BPR * 16;

  float acc = 0.0f;
  for (uint br = tid; br < BPR; br += THREADS) {
    float s = float(srow[br]);
    float s16 = s * (1.0f/16.0f);
    float s256 = s * (1.0f/256.0f);
    float s4096 = s * (1.0f/4096.0f);
    float md = -8.0f * s;
    const device uchar *pb = prow + br*16;
#pragma unroll
    for (uint j = 0; j < 16; j += 2) {
      ushort q = *reinterpret_cast<const device ushort *>(pb + j);
      float q0 = s * float(q & 0x000F) + md;
      float q1 = s16 * float(q & 0x00F0) + md;
      float q2 = s256 * float(q & 0x0F00) + md;
      float q3 = s4096 * float(q & 0xF000) + md;
      uint xb = br*32 + j;
      acc += q0*float(xrow[xb]) + q1*float(xrow[xb + 16]) + q2*float(xrow[xb + 1]) + q3*float(xrow[xb + 17]);
    }
  }

%(REDUCE_EPILOGUE)s
}
"""

def _threads_for_shape(n_sel:int, out_features:int, in_features:int) -> int:
  if n_sel == 4 and out_features == 2048 and in_features == 1536: return 32
  if n_sel == 4 and out_features == 3072 and in_features == 2048: return 32
  return 64

class Q4MoEMSLRunner(CompiledRunner):
  pass

Q4_MOE_MSL_TAG = "q4_moe_mul_mat_id_msl"

@functools.lru_cache(maxsize=None)
def _get_q4_moe_runner(device:str, n_sel:int, num_experts:int, out_features:int, in_features:int) -> Q4MoEMSLRunner:
  bpr = in_features // 32
  threads = _threads_for_shape(n_sel, out_features, in_features)
  nsg = (threads + 31) // 32
  reduce_epilogue = """  float sumf = simd_sum(acc);
  if (tiisg == 0) out[n*O + o] = half(sumf);""" if nsg == 1 else """  threadgroup float tacc[NSG];
  float sumf = simd_sum(acc);
  if (tiisg == 0) tacc[sgitg] = sumf;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgitg == 0) {
    float total = (tiisg < NSG) ? tacc[tiisg] : 0.0f;
    total = simd_sum(total);
    if (tiisg == 0) out[n*O + o] = half(total);
  }"""
  src = MSL_TEMPLATE % dict(N=n_sel, O=out_features, I=in_features, BPR=bpr, E=num_experts, THREADS=threads, NSG=nsg, REDUCE_EPILOGUE=reduce_epilogue)
  lib = Device[device].compiler.compile_cached(src)
  prg = Device[device].runtime("q4_moe_mul_mat_id", lib)
  p = ProgramSpec(
    name=f"q4_moe_mul_mat_id_msl_{n_sel}_{out_features}_{in_features}",
    src=src, device=device, ast=UOp(Ops.NOOP), lib=lib,
    global_size=[out_features, n_sel, 1], local_size=[threads, 1, 1],
    vars=[], globals=[0, 1, 2, 3, 4], outs=[0], ins=[1, 2, 3, 4],
  )
  return Q4MoEMSLRunner(p, prg)

def custom_q4_0_mul_mat_id_msl(out:UOp, x:UOp, scale:UOp, packed:UOp, sel:UOp) -> UOp:
  # Encode shape-specialized launch metadata in AST so lowering can pick the cached runner without forcing realization.
  n_sel, in_features = x.shape
  num_experts, out_features, bpr, one = scale.shape
  assert all_int((n_sel, in_features, num_experts, out_features, bpr, one)), "q4_moe_msl requires static int shapes"
  assert one == 1 and in_features == bpr * 32, "q4_moe_msl shape mismatch"
  tag = UOp(Ops.CUSTOM, dtypes.void, arg=(Q4_MOE_MSL_TAG, n_sel, num_experts, out_features, in_features))
  return UOp(Ops.SINK, dtypes.void, src=(tag,), arg=KernelInfo(name=f"{Q4_MOE_MSL_TAG}_{n_sel}_{out_features}_{in_features}"))

def lower_q4_moe_msl_ast(ast:UOp, device:str) -> Q4MoEMSLRunner|None:
  if ast.op is not Ops.SINK or len(ast.src) != 1: return None
  tag = ast.src[0]
  if tag.op is not Ops.CUSTOM or not isinstance(tag.arg, tuple) or len(tag.arg) != 5 or tag.arg[0] != Q4_MOE_MSL_TAG:
    return None
  _, n_sel, num_experts, out_features, in_features = tag.arg
  if not isinstance(device, str) or not device.startswith("METAL"): return None
  assert all_int((n_sel, num_experts, out_features, in_features)), "q4_moe_msl lowering requires static shapes"
  return _get_q4_moe_runner(device, int(n_sel), int(num_experts), int(out_features), int(in_features))

def q4_moe_mul_mat_id_msl(x:Tensor, scale:Tensor, packed:Tensor, sel:Tensor) -> Tensor:
  # x: (N, I), scale: (E, O, I//32, 1), packed: (E, O, I//32, 16), sel: (N,)
  if not isinstance(x.device, str) or not x.device.startswith("METAL"):
    raise RuntimeError("q4_moe_mul_mat_id_msl requires METAL device")
  if x.dtype != dtypes.float16:
    raise RuntimeError("q4_moe_mul_mat_id_msl requires float16 input")
  if len(x.shape) != 2 or len(scale.shape) != 4 or len(packed.shape) != 4 or len(sel.shape) != 1:
    raise RuntimeError("q4_moe_mul_mat_id_msl requires x(2D), scale(4D), packed(4D), sel(1D)")

  n_sel, in_features = x.shape
  num_experts, out_features, bpr, one = scale.shape
  if one != 1 or in_features != bpr * 32:
    raise RuntimeError("q4_moe_mul_mat_id_msl shape mismatch")
  if packed.shape != (num_experts, out_features, bpr, 16):
    raise RuntimeError("q4_moe_mul_mat_id_msl packed shape mismatch")
  if sel.shape[0] != n_sel:
    raise RuntimeError("q4_moe_mul_mat_id_msl sel shape mismatch")

  if scale.dtype != dtypes.float16: raise RuntimeError("q4_moe_mul_mat_id_msl requires float16 scale")
  if packed.dtype != dtypes.uint8: raise RuntimeError("q4_moe_mul_mat_id_msl requires uint8 packed")
  if sel.dtype != dtypes.int: raise RuntimeError("q4_moe_mul_mat_id_msl requires int32 sel")
  if not x.uop.has_buffer_identity():
    raise RuntimeError("q4_moe_mul_mat_id_msl requires buffer-identity x tensor")
  if not scale.uop.has_buffer_identity():
    raise RuntimeError("q4_moe_mul_mat_id_msl requires buffer-identity scale tensor")
  if not packed.uop.has_buffer_identity():
    raise RuntimeError("q4_moe_mul_mat_id_msl requires buffer-identity packed tensor")
  if not sel.uop.has_buffer_identity():
    raise RuntimeError("q4_moe_mul_mat_id_msl requires buffer-identity sel tensor")
  out = Tensor.empty((n_sel, out_features), device=x.device, dtype=dtypes.float16)

  runner = _get_q4_moe_runner(x.device, n_sel, num_experts, out_features, in_features)
  bufs:list[Buffer] = [out.uop.buffer.ensure_allocated(), x.uop.buffer.ensure_allocated(), scale.uop.buffer.ensure_allocated(),
                       packed.uop.buffer.ensure_allocated(), sel.uop.buffer.ensure_allocated()]
  ei = ExecItem(UOp(Ops.NOOP), bufs, prg=runner)
  if len(capturing) and CAPTURING: capturing[0].add(ei)
  ei.run({})
  return out
