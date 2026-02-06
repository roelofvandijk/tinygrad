from __future__ import annotations
import functools
from tinygrad import Device, Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.ops_metal import MetalProgram, MetalCompiler
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner, ExecItem, capturing, CAPTURING

# Tuned parameters: NR=8, NSG=4 gives ~400 GB/s (6x over 63 GB/s baseline)
N_SIMDGROUP = 4
N_SIMDWIDTH = 32
NR = 8  # rows per threadgroup
NR0 = NR // N_SIMDGROUP  # rows per simdgroup = 2

# Template with O, K baked as compile-time constants (no constant buffer args).
# This lets Q4KRunner be a CompiledRunner so MetalGraph can batch Q4K dispatches into ICBs.
def _make_q4k_src(out_features: int, in_features: int) -> str:
  return f"""
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12
#define NR0 {NR0}
#define O {out_features}
#define K {in_features}

typedef struct {{
  half d;
  half dmin;
  uchar scales[K_SCALE_SIZE];
  uchar qs[QK_K/2];
}} block_q4_K;

kernel void q4k_linear(
  device half *out [[buffer(0)]],
  device const half *x [[buffer(1)]],
  device const block_q4_K *w [[buffer(2)]],
  uint3 tg_id [[threadgroup_position_in_grid]],
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]
) {{
  const ushort ix = tiisg / 8;  // 0..3
  const ushort it = tiisg % 8;  // 0..7
  const ushort iq = it / 4;     // 0 or 1
  const ushort ir = it % 4;     // 0..3

  const uint row_base = tg_id.x * {NR} + sgitg * NR0;
  const uint batch = tg_id.y;
  if (row_base >= O) return;

  const uint nb = K / QK_K;
  device const half * y = x + batch * K;

  float yl[16];
  float yh[16];
  float sumf[NR0];
  for (uint r = 0; r < NR0; r++) sumf[r] = 0.0f;

  device const half * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

  uint16_t sc16[4];
  thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

  constexpr uint16_t kmask1 = 0x3f3f;
  constexpr uint16_t kmask2 = 0x0f0f;
  constexpr uint16_t kmask3 = 0xc0c0;

  for (uint ib = ix; ib < nb; ib += 4) {{
    float4 sumy = float4(0.0f);

    for (ushort i = 0; i < 8; ++i) {{
      yl[i+0] = float(y4[i+  0]); sumy[0] += yl[i+0];
      yl[i+8] = float(y4[i+ 32]); sumy[1] += yl[i+8];
      yh[i+0] = float(y4[i+128]); sumy[2] += yh[i+0];
      yh[i+8] = float(y4[i+160]); sumy[3] += yh[i+8];
    }}

    for (uint r = 0; r < NR0; r++) {{
      uint row = row_base + r;
      if (row >= O) continue;
      device const block_q4_K * xb = w + row * nb;
      device const uint16_t * sc = (device const uint16_t *)xb[ib].scales + iq;
      device const uint16_t * q1 = (device const uint16_t *)xb[ib].qs + 16 * iq + 4 * ir;
      device const half     * dh = &xb[ib].d;

      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

      device const uint16_t * q2 = q1 + 32;

      float4 acc1 = float4(0.0f);
      float4 acc2 = float4(0.0f);

      for (ushort i = 0; i < 4; ++i) {{
        acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
        acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
        acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
        acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
        acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
        acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
        acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
        acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
      }}

      sumf[r] += float(dh[0]) * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                 float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
    }}

    y4 += 4 * QK_K;
  }}

  for (uint r = 0; r < NR0; r++) {{
    uint row = row_base + r;
    if (row >= O) continue;
    float sum_all = simd_sum(sumf[r]);
    if (tiisg == 0) out[batch * O + row] = half(sum_all);
  }}
}}
"""

class Q4KRunner(CompiledRunner):
  def __init__(self, p: ProgramSpec, prg: MetalProgram, out_features: int, in_features: int):
    self._out_features, self._in_features = out_features, in_features
    super().__init__(p, prg)

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[str, int]|None=None, wait=False) -> float|None:
    if var_vals is None: var_vals = {}
    batch = rawbufs[1].nbytes // (self._in_features * 2)
    gsz = (self.p.global_size[0], batch, 1)
    lsz = tuple(self.p.local_size)
    return self._prg(*[x._buf for x in rawbufs], global_size=gsz, local_size=lsz, vals=(), wait=wait)

@functools.lru_cache(maxsize=None)
def _get_q4k_runner(device: str, out_features: int, in_features: int) -> Q4KRunner:
  src = _make_q4k_src(out_features, in_features)
  lib = MetalCompiler().compile(src)
  prg = MetalProgram(Device[device], "q4k_linear", lib)
  p = ProgramSpec(
    name=f"q4k_linear_msl_{out_features}x{in_features}", src=src, device=device, ast=UOp(Ops.NOOP), lib=lib,
    global_size=[(out_features + NR - 1) // NR, 1, 1], local_size=[N_SIMDGROUP * N_SIMDWIDTH, 1, 1],
    vars=[], globals=[0, 1, 2], outs=[0], ins=[1, 2],
  )
  return Q4KRunner(p, prg, out_features, in_features)

def q4k_linear_msl(x: Tensor, blocks: Tensor, out_features: int, in_features: int) -> Tensor:
  dev = x.device[0] if isinstance(x.device, tuple) else x.device
  if not isinstance(dev, str) or not dev.startswith("METAL"):
    raise RuntimeError("q4k_linear_msl requires METAL device")
  if x.dtype != dtypes.float16:
    raise RuntimeError("q4k_linear_msl requires float16 input")
  if in_features % 256 != 0:
    raise RuntimeError(f"q4k_linear_msl requires in_features % 256 == 0, got {in_features}")

  x = x.contiguous()
  blocks = blocks.contiguous().cast(dtypes.uint8)
  batch = x.shape[0]
  out = Tensor.empty((batch, out_features), device=x.device, dtype=dtypes.float16)
  Tensor.realize(x, blocks, out)

  runner = _get_q4k_runner(dev, out_features, in_features)
  bufs = [out.uop.buffer.ensure_allocated(), x.uop.buffer, blocks.uop.buffer]
  ei = ExecItem(UOp(Ops.NOOP), bufs, prg=runner)
  if len(capturing) and CAPTURING: capturing[0].add(ei)
  ei.run({})
  return out
