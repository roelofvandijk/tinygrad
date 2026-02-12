#!/usr/bin/env python3
from __future__ import annotations
import argparse, statistics, time
from tinygrad import Tensor, Device, TinyJit
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context
from tinygrad.apps.quantized import custom_q4_0_mul_mat_id
from tinygrad.apps.q4_moe_msl import q4_moe_mul_mat_id_msl
from tinygrad.apps.mla import _topk_pairwise
from tinygrad.apps.quantized import QuantizedExpertWeights
from tinygrad.apps.llm import Transformer, models

MSL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint N = {N};
constant uint O = {O};
constant uint I = {I};
constant uint BPR = {BPR};
constant uint E = {E};
constant uint THREADS = {THREADS};
constant uint NSG = {NSG};

kernel void q4_moe_mul_mat_id(
  device half *out [[buffer(0)]],
  device const half *x [[buffer(1)]],
  device const half *scale [[buffer(2)]],
  device const uchar *packed [[buffer(3)]],
  device const int *sel [[buffer(4)]],
  uint3 gid [[threadgroup_position_in_grid]],
  uint3 lid [[thread_position_in_threadgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]],
  ushort tiisg [[thread_index_in_simdgroup]]) {{
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
  for (uint br = tid; br < BPR; br += THREADS) {{
    float s = float(srow[br]);
    float s16 = s * (1.0f/16.0f);
    float s256 = s * (1.0f/256.0f);
    float s4096 = s * (1.0f/4096.0f);
    float md = -8.0f * s;
    const device uchar *pb = prow + br*16;
#pragma unroll
    for (uint j = 0; j < 16; j += 2) {{
      // Two packed bytes at once -> four q4 values (llama.cpp style mask/scaling, no explicit shifts).
      ushort q = *reinterpret_cast<const device ushort *>(pb + j);
      float q0 = s * float(q & 0x000F) + md;
      float q1 = s16 * float(q & 0x00F0) + md;
      float q2 = s256 * float(q & 0x0F00) + md;
      float q3 = s4096 * float(q & 0xF000) + md;
      uint xb = br*32 + j;
      acc += q0*float(xrow[xb]) + q1*float(xrow[xb + 16]) + q2*float(xrow[xb + 1]) + q3*float(xrow[xb + 17]);
    }}
  }}

  threadgroup float tacc[NSG];
  float sumf = simd_sum(acc);
  if (tiisg == 0) tacc[sgitg] = sumf;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgitg == 0) {{
    float total = (tiisg < NSG) ? tacc[tiisg] : 0.0f;
    total = simd_sum(total);
    if (tiisg == 0) out[n*O + o] = half(total);
  }}
}}
"""
DEFAULT_MODEL = "glm-4.7:flash-unsloth-Q4_0"

def build_inputs(N:int, O:int, I:int, E:int):
  assert I % 32 == 0
  bpr = I // 32
  x = Tensor.randn(N, I, dtype=dtypes.float16).contiguous().realize()
  scale = Tensor.randn(E, O, bpr, 1, dtype=dtypes.float16).reshape(E, O, bpr).contiguous().realize()
  packed = (Tensor.rand(E, O, bpr, 16) * 256).cast(dtypes.uint8).contiguous().realize()
  sel = (Tensor.rand(N) * E).cast(dtypes.int).contiguous().realize()
  return x, scale, packed, sel

def run_msl_kernel(N:int, O:int, I:int, E:int, threads:int, iters:int):
  bpr = I // 32
  x, scale, packed, sel = build_inputs(N, O, I, E)
  out = Tensor.empty(N, O, dtype=dtypes.float16, device=x.device).contiguous().realize()

  dev = Device[x.device]
  src = MSL_TEMPLATE.format(N=N, O=O, I=I, BPR=bpr, E=E, THREADS=threads, NSG=(threads + 31)//32)
  lib = dev.compiler.compile_cached(src)
  prg = dev.runtime("q4_moe_mul_mat_id", lib)

  # correctness against pure tensor math reference (independent of custom-kernel scheduling opts).
  x_pairs = x.reshape(N, 1, bpr, 2, 16)
  x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
  scale4 = scale.reshape(E, O, bpr, 1)
  scale_sel = scale4[sel]
  packed_sel = packed[sel]
  lo = packed_sel.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed_sel.rshift(4).cast(dtypes.float16) - 8.0
  ref = (scale_sel * (lo * x_lo + hi * x_hi)).reshape(N, O, bpr * 16).sum(axis=-1).cast(dtypes.float16).realize()

  for _ in range(3):
    prg(out._buffer()._buf, x._buffer()._buf, scale._buffer()._buf, packed._buffer()._buf, sel._buffer()._buf,
        global_size=(O, N, 1), local_size=(threads, 1, 1), wait=True)

  ts = []
  for _ in range(iters):
    ts.append(prg(out._buffer()._buf, x._buffer()._buf, scale._buffer()._buf, packed._buffer()._buf, sel._buffer()._buf,
                  global_size=(O, N, 1), local_size=(threads, 1, 1), wait=True))

  out.realize()
  max_diff = (out.float() - ref.float()).abs().max().item()
  med_s = statistics.median(ts)
  gflops = (2.0 * N * O * I) / med_s / 1e9
  print(f"shape N={N} O={O} I={I} E={E} threads={threads}")
  print(f"msl median: {med_s*1e6:.2f} us, gflops: {gflops:.1f}, max_abs_diff: {max_diff:.6f}")
  return med_s, max_diff

def run_tinygrad_path(N:int, O:int, I:int, E:int, iters:int):
  bpr = I // 32
  x, scale, packed, sel = build_inputs(N, O, I, E)
  times = []
  for _ in range(3):
    out = Tensor.empty(N, O, dtype=dtypes.float16, device=x.device)
    Tensor.mul_mat_id(out, x, scale.reshape(E, O, bpr, 1), packed, sel, fxn=custom_q4_0_mul_mat_id).realize()
  for _ in range(iters):
    st = time.perf_counter()
    out = Tensor.empty(N, O, dtype=dtypes.float16, device=x.device)
    Tensor.mul_mat_id(out, x, scale.reshape(E, O, bpr, 1), packed, sel, fxn=custom_q4_0_mul_mat_id).realize()
    Device[x.device].synchronize()
    times.append(time.perf_counter() - st)
  med_s = statistics.median(times)
  gflops = (2.0 * N * O * I) / med_s / 1e9
  print(f"tinygrad custom median (wall): {med_s*1e6:.2f} us, gflops: {gflops:.1f}")
  return med_s

def _bench_wall(fn, warmup:int, iters:int):
  for _ in range(warmup):
    fn().realize()
    Device.default.synchronize()
  times = []
  for _ in range(iters):
    Device.default.synchronize()
    st = time.perf_counter()
    fn().realize()
    Device.default.synchronize()
    times.append(time.perf_counter() - st)
  return statistics.median(times), min(times), sum(times)/len(times)

def _bench_jit_wall(fn, args:tuple, warmup:int, iters:int):
  for _ in range(warmup):
    fn(*args).realize()
    Device.default.synchronize()
  times = []
  for _ in range(iters):
    Device.default.synchronize()
    st = time.perf_counter()
    fn(*args).realize()
    Device.default.synchronize()
    times.append(time.perf_counter() - st)
  return statistics.median(times), min(times), sum(times)/len(times)

def _run_like_for_like(model:str, iters:int, warmup:int, decode_warmup:int):
  model_src = models.get(model, model)
  print(f"Loading quantized model for like-for-like: {model}")
  tmodel, _ = Transformer.from_gguf(Tensor.from_url(model_src), max_context=4096, quantized=True, realize=False)
  block = next(blk for blk in tmodel.blk if isinstance(getattr(blk, "ffn_gate_up_exps", None), QuantizedExpertWeights)
               and isinstance(getattr(blk, "ffn_down_exps", None), QuantizedExpertWeights))
  dim = block.ffn_gate_inp.weight.shape[1]
  print(f"Loaded MoE block: dim={dim}, experts={block.ffn_gate_inp.weight.shape[0]}, topk={block.num_experts_per_tok}, moe_hidden={block.moe_hidden_dim}")
  if decode_warmup > 0:
    print(f"Decode warmup ({decode_warmup} iter) to match loaded-state conditions...")
    for i in range(decode_warmup):
      xw = Tensor.randn(1, 1, dim).cast(dtypes.float16).realize()
      block(xw, i + 1).realize()
      Device.default.synchronize()

  # Build representative decode tensors from current model path.
  h = Tensor.randn(1, 1, dim).cast(dtypes.float16).realize()
  h_norm = block.ffn_norm(h).realize()
  router_logits = h_norm.float() @ block.ffn_gate_inp_f32.T
  gate_scores = router_logits.sigmoid() if block.expert_gating_func == 2 else router_logits.softmax(-1)
  selection_scores = gate_scores + block.exp_probs_b.bias if hasattr(block, "exp_probs_b") else gate_scores
  _, sel = _topk_pairwise(selection_scores, block.num_experts_per_tok)
  sel_flat = sel.reshape(-1).cast(dtypes.int).contiguous().realize()
  x_flat = h_norm.reshape(1, 1, 1, dim).expand(1, 1, block.num_experts_per_tok, dim).reshape(-1, dim).cast(dtypes.float16).contiguous().realize()

  gate_up = block.ffn_gate_up_exps
  down = block.ffn_down_exps
  gate_up._ensure_q4_0_separated(x_flat.device)
  down._ensure_q4_0_separated(x_flat.device)

  # down input derived from custom gate_up output to keep both paths aligned.
  gate_up_ref = Tensor.empty(x_flat.shape[0], gate_up.out_features, dtype=dtypes.float16, device=x_flat.device)
  gate_up_ref = Tensor.mul_mat_id(gate_up_ref, x_flat, gate_up._q4_0_scale, gate_up._q4_0_packed, sel_flat, fxn=custom_q4_0_mul_mat_id).realize()
  hid = gate_up.out_features // 2
  g, u = gate_up_ref[..., :hid], gate_up_ref[..., hid:]
  x_down = (g.silu() * u).cast(dtypes.float16).contiguous().realize()

  @TinyJit
  def custom_gate(x_in, sel_in):
    out = Tensor.empty(x_in.shape[0], gate_up.out_features, dtype=dtypes.float16, device=x_in.device)
    return Tensor.mul_mat_id(out, x_in, gate_up._q4_0_scale, gate_up._q4_0_packed, sel_in, fxn=custom_q4_0_mul_mat_id).contiguous()

  @TinyJit
  def msl_gate(x_in, sel_in):
    return q4_moe_mul_mat_id_msl(x_in, gate_up._q4_0_scale, gate_up._q4_0_packed, sel_in).contiguous()

  @TinyJit
  def custom_down(x_in, sel_in):
    out = Tensor.empty(x_in.shape[0], down.out_features, dtype=dtypes.float16, device=x_in.device)
    return Tensor.mul_mat_id(out, x_in, down._q4_0_scale, down._q4_0_packed, sel_in, fxn=custom_q4_0_mul_mat_id).contiguous()

  @TinyJit
  def msl_down(x_in, sel_in):
    return q4_moe_mul_mat_id_msl(x_in, down._q4_0_scale, down._q4_0_packed, sel_in).contiguous()

  # correctness snapshot on real model tensors
  cg = custom_gate(x_flat, sel_flat).float().realize()
  mg = msl_gate(x_flat, sel_flat).float().realize()
  cd = custom_down(x_down, sel_flat).float().realize()
  md = msl_down(x_down, sel_flat).float().realize()
  gate_diff = (cg - mg).abs()
  down_diff = (cd - md).abs()
  gate_nfinite = int(gate_diff.isfinite().sum().item())
  down_nfinite = int(down_diff.isfinite().sum().item())
  gate_total = int(gate_diff.numel())
  down_total = int(down_diff.numel())
  gate_max = gate_diff.where(gate_diff.isfinite(), 0).max().item()
  down_max = down_diff.where(down_diff.isfinite(), 0).max().item()
  print(f"Correctness snapshot (custom vs msl): gate_up finite={gate_nfinite}/{gate_total}, max_abs_diff={gate_max:.6f}")
  print(f"Correctness snapshot (custom vs msl): down    finite={down_nfinite}/{down_total}, max_abs_diff={down_max:.6f}")

  print("\n== like-for-like gate/up (3072x2048) ==")
  c_med, c_best, c_avg = _bench_jit_wall(custom_gate, (x_flat, sel_flat), warmup, iters)
  m_med, m_best, m_avg = _bench_jit_wall(msl_gate, (x_flat, sel_flat), warmup, iters)
  print(f"custom median/best/avg: {c_med*1e6:.2f}/{c_best*1e6:.2f}/{c_avg*1e6:.2f} us")
  print(f"msl    median/best/avg: {m_med*1e6:.2f}/{m_best*1e6:.2f}/{m_avg*1e6:.2f} us")
  print(f"speedup (custom/msl): {c_med/m_med:.2f}x")

  print("\n== like-for-like down (2048x1536) ==")
  c_med, c_best, c_avg = _bench_jit_wall(custom_down, (x_down, sel_flat), warmup, iters)
  m_med, m_best, m_avg = _bench_jit_wall(msl_down, (x_down, sel_flat), warmup, iters)
  print(f"custom median/best/avg: {c_med*1e6:.2f}/{c_best*1e6:.2f}/{c_avg*1e6:.2f} us")
  print(f"msl    median/best/avg: {m_med*1e6:.2f}/{m_best*1e6:.2f}/{m_avg*1e6:.2f} us")
  print(f"speedup (custom/msl): {c_med/m_med:.2f}x")

def main():
  ap = argparse.ArgumentParser(description="Prototype llama-style MSL Q4 MoE mul_mat_id microbench")
  ap.add_argument("--mode", choices=["synthetic", "like-for-like"], default="like-for-like")
  ap.add_argument("--threads", type=int, default=64)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--n", type=int, default=4)
  ap.add_argument("--e", type=int, default=64)
  ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
  ap.add_argument("--decode-warmup", type=int, default=1)
  args = ap.parse_args()

  with Context(BEAM=0):
    if args.mode == "like-for-like":
      _run_like_for_like(args.model, args.iters, args.warmup, args.decode_warmup)
      return

    print("== gate/up shape (3072x2048) ==")
    msl_gate, diff_gate = run_msl_kernel(args.n, 3072, 2048, args.e, args.threads, args.iters)
    tg_gate = run_tinygrad_path(args.n, 3072, 2048, args.e, args.iters)
    print(f"speedup vs tinygrad path: {tg_gate/msl_gate:.2f}x")
    print()
    print("== down shape (2048x1536) ==")
    msl_down, diff_down = run_msl_kernel(args.n, 2048, 1536, args.e, args.threads, args.iters)
    tg_down = run_tinygrad_path(args.n, 2048, 1536, args.e, args.iters)
    print(f"speedup vs tinygrad path: {tg_down/msl_down:.2f}x")

    # fp16 output + different reduction order gives up to ~0.5 absolute diff in stress tests.
    if max(diff_gate, diff_down) > 0.55:
      raise SystemExit("correctness check failed")

if __name__ == "__main__":
  main()
