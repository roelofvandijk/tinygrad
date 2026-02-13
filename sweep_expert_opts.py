#!/usr/bin/env python3
"""Microbench sweep for Q4_0 mul_mat_id expert kernel opts."""
import os, time, sys
os.environ.setdefault("DEVICE", "METAL")
os.environ.setdefault("BEAM_REDUCE_ONLY", "1")

from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.apps.quantized import (
  QuantizedExpertWeights, custom_q4_0_mul_mat_id,
  _q4_0_mul_mat_id_opts,
)
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.nn.state import GGML_QUANT_INFO
from bench_block import make_q4_0_expert_weights

def bench_expert_kernel(num_experts, out_features, in_features, n_sel, opts_fn, n_iter=20, warmup=5):
  """Benchmark a single expert kernel configuration."""
  ew = make_q4_0_expert_weights(num_experts, out_features, in_features, expert_first_in_memory=True)
  x = Tensor.randn(1, 1, n_sel, in_features).cast(dtypes.float16).realize()
  sel = Tensor.randint(1, 1, n_sel, high=num_experts, dtype=dtypes.int32).realize()

  # Monkey-patch the opts function
  import tinygrad.apps.quantized as qmod
  old_fn = qmod._q4_0_mul_mat_id_opts
  qmod._q4_0_mul_mat_id_opts = opts_fn

  try:
    # Warmup
    for _ in range(warmup):
      out = ew(sel, x).realize()
      Device.default.synchronize()

    # Benchmark
    times = []
    for _ in range(n_iter):
      Device.default.synchronize()
      st = time.perf_counter()
      out = ew(sel, x).realize()
      Device.default.synchronize()
      times.append((time.perf_counter() - st) * 1000)
  finally:
    qmod._q4_0_mul_mat_id_opts = old_fn

  times.sort()
  median = times[len(times)//2]
  return median

def sweep(shape_name, num_experts, out_features, in_features, n_sel):
  print(f"\n{'='*60}")
  print(f"Sweeping {shape_name}: E={num_experts} O={out_features} I={in_features} K={n_sel}")
  print(f"{'='*60}")

  configs = []
  for local_sz in [4, 8, 16]:
    for group_sz in [4, 8, 16]:
      for group_type in ["GROUPTOP", "GROUP"]:
        opt_type = OptOps.GROUPTOP if group_type == "GROUPTOP" else OptOps.GROUP
        def make_fn(ls=local_sz, gs=group_sz, ot=opt_type):
          def fn(n, o, i):
            if n == n_sel and o == out_features and i == in_features:
              return (Opt(OptOps.LOCAL, 1, ls), Opt(ot, 1, gs))
            return _q4_0_mul_mat_id_opts(n, o, i)
          return fn
        configs.append((f"LOCAL(1,{local_sz})+{group_type}(1,{group_sz})", make_fn()))

  results = []
  for name, fn in configs:
    try:
      t = bench_expert_kernel(num_experts, out_features, in_features, n_sel, fn, n_iter=15, warmup=3)
      bpr = in_features // 32
      data_bytes = n_sel * (out_features * bpr * (2 + 16) + in_features * 2 + out_features * 2)
      gbps = data_bytes / (t / 1000) / 1e9
      results.append((t, gbps, name))
      print(f"  {name:40s} {t:8.3f} ms  {gbps:7.1f} GB/s")
    except Exception as e:
      print(f"  {name:40s} FAILED: {e}")

  results.sort()
  print(f"\nTop 3 for {shape_name}:")
  for i, (t, gbps, name) in enumerate(results[:3]):
    print(f"  #{i+1}: {name:40s} {t:8.3f} ms  {gbps:7.1f} GB/s")

if __name__ == "__main__":
  # Down expert: (4, 2048, 1536) — this is the one we just fixed
  sweep("down_expert", 64, 2048, 1536, 4)
  # Gate_up expert: (4, 3072, 2048) — already working but maybe can be tuned
  sweep("gate_up_expert", 64, 3072, 2048, 4)
