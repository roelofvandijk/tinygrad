#!/usr/bin/env python3
from __future__ import annotations
import argparse, statistics, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt import Opt, OptOps
import tinygrad.apps.quantized as qmod
from tinygrad.apps.quantized import custom_q4_0_mul_mat_id

CASES = {
  "gate_up": (4, 3072, 2048),
  "down": (4, 2048, 1536),
}

def _parse_opt_token(tok:str) -> Opt:
  op_s, ax_s, arg_s = tok.split(":")
  return Opt(getattr(OptOps, op_s), int(ax_s), int(arg_s))

def _install_moe_opts_mode(mode:str):
  orig = qmod._q4_0_mul_mat_id_opts
  if mode == "default": return orig
  if mode == "auto":
    qmod._q4_0_mul_mat_id_opts = lambda _n, _o, _i: None
    return orig
  if mode == "none":
    qmod._q4_0_mul_mat_id_opts = lambda _n, _o, _i: ()
    return orig
  fixed = tuple(_parse_opt_token(t.strip()) for t in mode.split(",") if t.strip())
  qmod._q4_0_mul_mat_id_opts = lambda _n, _o, _i: fixed
  return orig

def _make_inputs(n:int, o:int, i:int, experts:int=64):
  bpr = i // 32
  x = Tensor.randn(n, i, dtype=dtypes.float16).contiguous().realize()
  sel = (Tensor.rand(n) * experts).cast(dtypes.int32).contiguous().realize()
  scale = Tensor.randn(experts, o, bpr, 1, dtype=dtypes.float16).contiguous().realize()
  packed = (Tensor.rand(experts, o, bpr, 16) * 256).cast(dtypes.uint8).contiguous().realize()
  return x, sel, scale, packed

def _ref_q4_moe(x:Tensor, sel:Tensor, scale:Tensor, packed:Tensor) -> np.ndarray:
  x_np = x.float().numpy().astype(np.float32)                         # (N, I)
  sel_np = sel.numpy().astype(np.int64)                               # (N,)
  scale_np = scale.float().numpy().astype(np.float32)                 # (E, O, bpr, 1)
  packed_np = packed.numpy().astype(np.uint8)                         # (E, O, bpr, 16)
  n, i = x_np.shape
  bpr = i // 32
  x_pairs = x_np.reshape(n, 1, bpr, 2, 16)
  x_lo = x_pairs[:, :, :, 0, :]
  x_hi = x_pairs[:, :, :, 1, :]
  p_sel = packed_np[sel_np]                                           # (N, O, bpr, 16)
  s_sel = scale_np[sel_np]                                            # (N, O, bpr, 1)
  lo = (p_sel & 0xF).astype(np.float32) - 8.0
  hi = (p_sel >> 4).astype(np.float32) - 8.0
  ref = (s_sel * (lo * x_lo + hi * x_hi)).reshape(n, p_sel.shape[1], bpr*16).sum(axis=-1)
  return ref.astype(np.float16)

def _run_case(n:int, o:int, i:int, warmup:int, iters:int, compare_msl:bool, opts_mode:str):
  orig_opts = _install_moe_opts_mode(opts_mode)
  x, sel, scale, packed = _make_inputs(n, o, i)
  device = x.device

  out = Tensor.empty(n, o, dtype=dtypes.float16, device=device)
  out = Tensor.mul_mat_id(out, x, scale, packed, sel, fxn=custom_q4_0_mul_mat_id).realize()
  ref = _ref_q4_moe(x, sel, scale, packed)
  max_diff = float(np.max(np.abs(out.float().numpy() - ref.astype(np.float32))))

  @TinyJit
  def dsl_run(xi:Tensor, si:Tensor, sc:Tensor, pk:Tensor):
    outi = Tensor.empty(n, o, dtype=dtypes.float16, device=device)
    return Tensor.mul_mat_id(outi, xi, sc, pk, si, fxn=custom_q4_0_mul_mat_id)

  for _ in range(warmup):
    dsl_run(x, sel, scale, packed).realize()
    Device.default.synchronize()

  times = []
  for _ in range(iters):
    Device.default.synchronize()
    st = time.perf_counter()
    dsl_run(x, sel, scale, packed).realize()
    Device.default.synchronize()
    times.append(time.perf_counter() - st)
  med = statistics.median(times); best = min(times); avg = sum(times)/len(times)
  gflops = (2.0 * n * o * i) / med / 1e9
  print(f"DSL[{opts_mode}]  N={n} O={o} I={i}  median={med*1e6:.2f}us best={best*1e6:.2f}us avg={avg*1e6:.2f}us gflops={gflops:.1f} max_diff={max_diff:.6f}")

  if compare_msl and isinstance(device, str) and device.startswith("METAL"):
    from tinygrad.apps.q4_moe_msl import custom_q4_0_mul_mat_id_msl
    @TinyJit
    def msl_run(xi:Tensor, si:Tensor, sc:Tensor, pk:Tensor):
      outi = Tensor.empty(n, o, dtype=dtypes.float16, device=device)
      return Tensor.mul_mat_id(outi, xi, sc, pk, si, fxn=custom_q4_0_mul_mat_id_msl)
    for _ in range(warmup):
      msl_run(x, sel, scale, packed).realize()
      Device.default.synchronize()
    t2=[]
    for _ in range(iters):
      Device.default.synchronize()
      st=time.perf_counter()
      msl_run(x, sel, scale, packed).realize()
      Device.default.synchronize()
      t2.append(time.perf_counter()-st)
    med2=statistics.median(t2); best2=min(t2); avg2=sum(t2)/len(t2)
    g2=(2.0*n*o*i)/med2/1e9
    print(f"MSL  N={n} O={o} I={i}  median={med2*1e6:.2f}us best={best2*1e6:.2f}us avg={avg2*1e6:.2f}us gflops={g2:.1f}")
    print(f"speedup DSL/MSL: {med/med2:.2f}x")

  qmod._q4_0_mul_mat_id_opts = orig_opts

def main():
  ap = argparse.ArgumentParser(description="Fast microbench for DSL Q4_0 MoE custom kernel")
  ap.add_argument("--case", default="gate_up", choices=sorted(CASES.keys()))
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--iters", type=int, default=30)
  ap.add_argument("--compare_msl", action="store_true")
  ap.add_argument("--opts_mode", type=str, default="default",
                  help="default|auto|none|CSV of OP:axis:arg (example: LOCAL:1:8,GROUP:1:8)")
  args = ap.parse_args()

  if args.all:
    for name in ("gate_up", "down"):
      print(f"\n== {name} ==")
      _run_case(*CASES[name], args.warmup, args.iters, args.compare_msl, args.opts_mode)
  else:
    _run_case(*CASES[args.case], args.warmup, args.iters, args.compare_msl, args.opts_mode)

if __name__ == "__main__":
  main()

