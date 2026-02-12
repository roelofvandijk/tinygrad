#!/usr/bin/env python3
from __future__ import annotations
import argparse, statistics, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt import Opt, OptOps
import tinygrad.apps.quantized as qmod
from tinygrad.apps.quantized import custom_q4_0_linear

CASES = {
  "ffn_in": (1, 2048, 5120),
  "ffn_mid": (1, 5120, 768),
  "ffn_out": (1, 768, 2048),
}

def _parse_opt_token(tok:str) -> Opt:
  # token format: OP:axis:arg, e.g. LOCAL:0:32 or GROUPTOP:1:8
  op_s, ax_s, arg_s = tok.split(":")
  op = getattr(OptOps, op_s)
  return Opt(op, int(ax_s), int(arg_s))

def _install_linear_opts_mode(mode:str):
  """
  Monkey-patch quantized._q4_0_linear_opts so this microbench can iterate scheduler candidates
  quickly without editing production code between runs.
  """
  orig = qmod._q4_0_linear_opts
  if mode == "default":
    return orig
  if mode == "auto":
    qmod._q4_0_linear_opts = lambda _n, _o, _i: None
    return orig
  if mode == "none":
    qmod._q4_0_linear_opts = lambda _n, _o, _i: ()
    return orig
  toks = [t.strip() for t in mode.split(",") if t.strip()]
  fixed = tuple(_parse_opt_token(t) for t in toks)
  qmod._q4_0_linear_opts = lambda _n, _o, _i: fixed
  return orig

def _make_inputs(n:int, o:int, i:int):
  assert i % 32 == 0, "Q4_0 requires input dim divisible by 32"
  bpr = i // 32
  x = Tensor.randn(n, i, dtype=dtypes.float16).contiguous().realize()
  scale = Tensor.randn(o, bpr, 1, dtype=dtypes.float16).contiguous().realize()
  packed = (Tensor.rand(o, bpr, 16) * 256).cast(dtypes.uint8).contiguous().realize()
  return x, scale, packed

def _q4_ref(x:Tensor, scale:Tensor, packed:Tensor) -> Tensor:
  n, i = x.shape
  o = scale.shape[0]
  bpr = i // 32
  x_pairs = x.reshape(n, 1, bpr, 2, 16)
  x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
  lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
  hi = packed.rshift(4).cast(dtypes.float16) - 8.0
  return (scale * (lo * x_lo + hi * x_hi)).reshape(n, o, bpr*16).sum(axis=-1).cast(dtypes.float16)

def _run_case(n:int, o:int, i:int, warmup:int, iters:int, compare_msl:bool, opts_mode:str):
  orig_opts = _install_linear_opts_mode(opts_mode)
  x, scale, packed = _make_inputs(n, o, i)
  device = x.device

  out = Tensor.empty(n, o, dtype=dtypes.float16, device=x.device)
  out = Tensor.custom_kernel(out, x, scale, packed, fxn=custom_q4_0_linear)[0].realize()
  ref = _q4_ref(x, scale, packed).realize()
  out_np = out.float().numpy()
  ref_np = ref.float().numpy()
  max_diff = float(np.max(np.abs(out_np - ref_np)))

  @TinyJit
  def dsl_run(xi:Tensor, si:Tensor, pi:Tensor):
    out = Tensor.empty(n, o, dtype=dtypes.float16, device=device)
    return Tensor.custom_kernel(out, xi, si, pi, fxn=custom_q4_0_linear)[0]

  for _ in range(warmup):
    dsl_run(x, scale, packed).realize()
    Device.default.synchronize()

  times = []
  for _ in range(iters):
    Device.default.synchronize()
    st = time.perf_counter()
    dsl_run(x, scale, packed).realize()
    Device.default.synchronize()
    times.append(time.perf_counter() - st)

  med = statistics.median(times)
  best = min(times)
  avg = sum(times) / len(times)
  gflops = (2.0 * n * o * i) / med / 1e9
  print(f"DSL[{opts_mode}]  N={n} O={o} I={i}  median={med*1e6:.2f}us best={best*1e6:.2f}us avg={avg*1e6:.2f}us gflops={gflops:.1f} max_diff={max_diff:.6f}")

  if not compare_msl:
    qmod._q4_0_linear_opts = orig_opts
    return
  if not isinstance(x.device, str) or not x.device.startswith("METAL"):
    print("MSL compare skipped (device is not METAL)")
    qmod._q4_0_linear_opts = orig_opts
    return

  from tinygrad.apps.q4_linear_msl import custom_q4_0_linear_msl
  @TinyJit
  def msl_run(xi:Tensor, si:Tensor, pi:Tensor):
    out = Tensor.empty(n, o, dtype=dtypes.float16, device=device)
    return Tensor.custom_kernel(out, xi, si, pi, fxn=custom_q4_0_linear_msl)[0]

  for _ in range(warmup):
    msl_run(x, scale, packed).realize()
    Device.default.synchronize()
  times_msl = []
  for _ in range(iters):
    Device.default.synchronize()
    st = time.perf_counter()
    msl_run(x, scale, packed).realize()
    Device.default.synchronize()
    times_msl.append(time.perf_counter() - st)
  med_msl = statistics.median(times_msl)
  best_msl = min(times_msl)
  avg_msl = sum(times_msl) / len(times_msl)
  gflops_msl = (2.0 * n * o * i) / med_msl / 1e9
  print(f"MSL  N={n} O={o} I={i}  median={med_msl*1e6:.2f}us best={best_msl*1e6:.2f}us avg={avg_msl*1e6:.2f}us gflops={gflops_msl:.1f}")
  print(f"speedup DSL/MSL: {med/med_msl:.2f}x")
  qmod._q4_0_linear_opts = orig_opts

def main():
  ap = argparse.ArgumentParser(description="Fast microbench for DSL Q4_0 dense custom kernel")
  ap.add_argument("--case", default="ffn_in", choices=sorted(CASES.keys()))
  ap.add_argument("--all", action="store_true", help="Run all default GLM Q4_0 dense shapes")
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--iters", type=int, default=30)
  ap.add_argument("--compare_msl", action="store_true", help="Also run custom_q4_0_linear_msl")
  ap.add_argument("--opts_mode", type=str, default="default",
                  help="default|auto|none|CSV of OP:axis:arg (e.g. LOCAL:0:32,GROUPTOP:1:8)")
  args = ap.parse_args()

  if args.all:
    for name in ("ffn_in", "ffn_mid", "ffn_out"):
      print(f"\n== {name} ==")
      _run_case(*CASES[name], args.warmup, args.iters, args.compare_msl, args.opts_mode)
  else:
    _run_case(*CASES[args.case], args.warmup, args.iters, args.compare_msl, args.opts_mode)

if __name__ == "__main__":
  main()
