#!/usr/bin/env python3
"""Representative Q4_0 MoE expert microbench for GLM decode shapes."""
import argparse, os, time

os.environ.setdefault("DEVICE", "METAL")
os.environ.setdefault("BEAM_REDUCE_ONLY", "1")

from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
from tinygrad.apps.quantized import QuantizedExpertWeights

def _median(vals:list[float]) -> float:
  v = sorted(vals)
  n = len(v)
  return v[n//2] if n % 2 else 0.5 * (v[n//2 - 1] + v[n//2])

def _quantile(vals:list[float], q:float) -> float:
  v = sorted(vals)
  return v[int((len(v) - 1) * q)]

def _stats(vals:list[float]) -> dict[str, float]:
  med = _median(vals)
  p10, p25 = _quantile(vals, 0.10), _quantile(vals, 0.25)
  p75, p90 = _quantile(vals, 0.75), _quantile(vals, 0.90)
  return {
    "median": med,
    "best": min(vals),
    "avg": sum(vals) / len(vals),
    "p10": p10,
    "p90": p90,
    "iqr": p75 - p25,
    "mad": _median([abs(x - med) for x in vals]),
  }

def _make_q4_0_blocks(n_blocks:int) -> Tensor:
  blocks = Tensor.randint(n_blocks, 18, high=256, dtype=dtypes.uint8)
  blocks[:, 0].assign(0)
  blocks[:, 1].assign(0x3C)
  return blocks.contiguous().realize()

def _make_q4_0_experts(num_experts:int, out_features:int, in_features:int) -> QuantizedExpertWeights:
  bpr = in_features // 32
  return QuantizedExpertWeights(
    _make_q4_0_blocks(num_experts * out_features * bpr),
    (num_experts, out_features, in_features),
    ggml_type=2,
    expert_first_in_memory=True,
  )

def _bench_case(name:str, experts:QuantizedExpertWeights, sel:Tensor, x:Tensor, iters:int, warmup:int) -> float:
  for _ in range(warmup):
    experts(sel, x).realize()
    Device.default.synchronize()

  times = []
  for _ in range(iters):
    Device.default.synchronize()
    st = time.perf_counter()
    experts(sel, x).realize()
    Device.default.synchronize()
    times.append((time.perf_counter() - st) * 1000)

  st = _stats(times)
  print(f"{name:18s}: median {st['median']:.2f} ms  best {st['best']:.2f} ms  avg {st['avg']:.2f} ms")
  print(f"{'':18s}  p10/p90 {st['p10']:.2f}/{st['p90']:.2f} ms  IQR {st['iqr']:.2f} ms  MAD {st['mad']:.2f} ms")
  return st["median"]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--iters", type=int, default=40)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--experts", type=int, default=64)
  parser.add_argument("--k", type=int, default=4)
  args = parser.parse_args()

  B, T, D, H = 1, 1, 2048, 1536
  K, E = args.k, args.experts

  print(f"Representative decode shapes: B={B} T={T} D={D} H={H} K={K} E={E}")
  sel = Tensor.randint(B, T, K, high=E, dtype=dtypes.int).realize()
  x_gate_up = Tensor.randn(B, T, D).cast(dtypes.float16).realize()
  x_down = Tensor.randn(B, T, K, H).cast(dtypes.float16).realize()

  gate_up = _make_q4_0_experts(E, 2*H, D)
  down = _make_q4_0_experts(E, D, H)

  print("Benchmarking production QuantizedExpertWeights(sel, x) path:")
  t_gate_up = _bench_case("gate_up (3072x2048)", gate_up, sel, x_gate_up, args.iters, args.warmup)
  t_down = _bench_case("down (2048x1536)", down, sel, x_down, args.iters, args.warmup)
  print(f"combined median: {t_gate_up + t_down:.2f} ms")

if __name__ == "__main__":
  main()
