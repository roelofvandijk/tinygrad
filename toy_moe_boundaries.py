#!/usr/bin/env python3
"""
Toy MoE boundary A/B benchmark.

Purpose:
  Validate whether removing forced fusion boundaries actually helps kernel count
  and wall time before touching full-model code.
"""
import time
import argparse
from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters

class ToyQ4ExpertWeights:
  """Minimal Q4_0 expert matmul with inline dequant."""
  def __init__(self, num_experts:int, in_dim:int, out_dim:int):
    self.num_experts, self.in_dim, self.out_dim = num_experts, in_dim, out_dim
    bpr = in_dim // 32
    blocks = Tensor.randint(num_experts * out_dim * bpr, 18, low=0, high=256, dtype=dtypes.uint8).realize()
    self.expert_blocks = blocks.reshape(num_experts, out_dim, bpr, 18)

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    B, T, K = sel.shape
    n_sel = B * T * K
    bpr = self.in_dim // 32
    sel_blocks = self.expert_blocks[sel.reshape(-1)]  # (n_sel, out_dim, bpr, 18)
    blocks = sel_blocks.reshape(n_sel, self.out_dim, bpr, 18)
    scale, packed = blocks[:, :, :, :2].bitcast(dtypes.float16), blocks[:, :, :, 2:]

    x_flat = x.unsqueeze(2).expand(B, T, K, self.in_dim).reshape(n_sel, self.in_dim).cast(dtypes.float16)
    x_pairs = x_flat.reshape(n_sel, 1, bpr, 2, 16)
    x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
    lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
    hi = packed.rshift(4).cast(dtypes.float16) - 8.0
    out = (scale.reshape(n_sel, self.out_dim, bpr) * (lo * x_lo + hi * x_hi).sum(axis=-1)).sum(axis=-1)
    return out.reshape(B, T, K, self.out_dim)

class ToyMoEBlock:
  def __init__(self, *, dim=2048, hidden=1536, num_experts=64, k=4, split_boundaries=True):
    self.dim, self.hidden, self.num_experts, self.k = dim, hidden, num_experts, k
    self.split_boundaries = split_boundaries
    self.gate_experts = ToyQ4ExpertWeights(num_experts, dim, hidden)
    self.up_experts = ToyQ4ExpertWeights(num_experts, dim, hidden)
    self.down_experts = ToyQ4ExpertWeights(num_experts, hidden, dim)

  def __call__(self, x:Tensor, sel:Tensor, probs:Tensor) -> Tensor:
    gate = self.gate_experts(sel, x).silu()
    up = self.up_experts(sel, x)
    if self.split_boundaries:
      gate, up = gate.contiguous(), up.contiguous()
    gated = gate * up
    expert_out = self.down_experts(sel, gated)
    if self.split_boundaries:
      expert_out = expert_out.contiguous()
    return x + (expert_out * probs).sum(axis=2)

def _run_mode(split_boundaries:bool, x0:Tensor, sels:list[Tensor], probs:list[Tensor],
              *, hidden:int, num_experts:int, warmup:int=3):
  block = ToyMoEBlock(split_boundaries=split_boundaries, dim=x0.shape[-1], hidden=hidden,
                     num_experts=num_experts, k=sels[0].shape[-1])
  x = x0
  for i in range(warmup):
    x = block(x, sels[i], probs[i])
    x.realize()

  ms, kernels = [], []
  for i in range(warmup, len(sels)):
    GlobalCounters.reset()
    st = time.perf_counter()
    x = block(x, sels[i], probs[i])
    x.realize()
    ms.append((time.perf_counter()-st)*1e3)
    kernels.append(GlobalCounters.kernel_count)
  return sum(ms)/len(ms), sum(kernels)/len(kernels), ms, kernels

def benchmark(n_tokens=12, B=1, T=1, dim=2048, hidden=1536, k=4, num_experts=64):
  Tensor.manual_seed(0)
  x0 = Tensor.randn(B, T, dim, dtype=dtypes.float16).realize()
  sels = [Tensor.randint(B*T*k, low=0, high=num_experts, dtype=dtypes.int32).reshape(B, T, k).realize() for _ in range(n_tokens)]
  probs = [Tensor.randn(B, T, k, dtype=dtypes.float16).softmax(-1).unsqueeze(-1).realize() for _ in range(n_tokens)]

  split_ms, split_kpt, split_ms_all, split_k_all = _run_mode(True, x0, sels, probs, hidden=hidden, num_experts=num_experts)
  fused_ms, fused_kpt, fused_ms_all, fused_k_all = _run_mode(False, x0, sels, probs, hidden=hidden, num_experts=num_experts)

  print("\n" + "="*72)
  print("TOY MOE BOUNDARY A/B")
  print("="*72)
  print(f"shape: B={B} T={T} dim={dim} hidden={hidden} experts={num_experts} k={k}")
  print(f"measured tokens: {n_tokens-3} (after 3 warmup)")
  print("")
  print("Mode                       avg ms/token   avg kernels/token")
  print("-----------------------------------------------------------")
  print(f"split_boundaries=True      {split_ms:10.2f}   {split_kpt:16.1f}")
  print(f"split_boundaries=False     {fused_ms:10.2f}   {fused_kpt:16.1f}")
  print("")
  print(f"kernel delta (fused-split): {fused_kpt-split_kpt:+.1f}")
  print(f"time delta ms/token:        {fused_ms-split_ms:+.2f}")
  print(f"time speedup (fused/split): {split_ms/fused_ms:0.3f}x")
  print("="*72 + "\n")
  return {
    "split": {"ms": split_ms, "kpt": split_kpt, "ms_all": split_ms_all, "k_all": split_k_all},
    "fused": {"ms": fused_ms, "kpt": fused_kpt, "ms_all": fused_ms_all, "k_all": fused_k_all},
  }

def benchmark_both_orders(n_tokens=12, B=1, T=1, dim=2048, hidden=1536, k=4, num_experts=64):
  """Run A/B both orders to reduce warmup/order bias."""
  Tensor.manual_seed(0)
  x0 = Tensor.randn(B, T, dim, dtype=dtypes.float16).realize()
  sels = [Tensor.randint(B*T*k, low=0, high=num_experts, dtype=dtypes.int32).reshape(B, T, k).realize() for _ in range(n_tokens)]
  probs = [Tensor.randn(B, T, k, dtype=dtypes.float16).softmax(-1).unsqueeze(-1).realize() for _ in range(n_tokens)]

  split_ms, split_kpt, *_ = _run_mode(True, x0, sels, probs, hidden=hidden, num_experts=num_experts)
  fused_ms, fused_kpt, *_ = _run_mode(False, x0, sels, probs, hidden=hidden, num_experts=num_experts)

  Tensor.manual_seed(0)
  x0 = Tensor.randn(B, T, dim, dtype=dtypes.float16).realize()
  sels = [Tensor.randint(B*T*k, low=0, high=num_experts, dtype=dtypes.int32).reshape(B, T, k).realize() for _ in range(n_tokens)]
  probs = [Tensor.randn(B, T, k, dtype=dtypes.float16).softmax(-1).unsqueeze(-1).realize() for _ in range(n_tokens)]
  fused_ms_2, fused_kpt_2, *_ = _run_mode(False, x0, sels, probs, hidden=hidden, num_experts=num_experts)
  split_ms_2, split_kpt_2, *_ = _run_mode(True, x0, sels, probs, hidden=hidden, num_experts=num_experts)

  split_ms_avg, fused_ms_avg = (split_ms + split_ms_2) / 2.0, (fused_ms + fused_ms_2) / 2.0
  split_k_avg, fused_k_avg = (split_kpt + split_kpt_2) / 2.0, (fused_kpt + fused_kpt_2) / 2.0

  print("\n" + "="*72)
  print("TOY MOE BOUNDARY A/B (BOTH ORDERS)")
  print("="*72)
  print(f"shape: B={B} T={T} dim={dim} hidden={hidden} experts={num_experts} k={k}")
  print(f"measured tokens per run: {n_tokens-3} (after 3 warmup)")
  print("")
  print("Mode                       avg ms/token   avg kernels/token")
  print("-----------------------------------------------------------")
  print(f"split_boundaries=True      {split_ms_avg:10.2f}   {split_k_avg:16.1f}")
  print(f"split_boundaries=False     {fused_ms_avg:10.2f}   {fused_k_avg:16.1f}")
  print("")
  print(f"kernel delta (fused-split): {fused_k_avg-split_k_avg:+.1f}")
  print(f"time delta ms/token:        {fused_ms_avg-split_ms_avg:+.2f}")
  print(f"time speedup (fused/split): {split_ms_avg/fused_ms_avg:0.3f}x")
  print("="*72 + "\n")
  return {
    "split": {"ms": split_ms_avg, "kpt": split_k_avg},
    "fused": {"ms": fused_ms_avg, "kpt": fused_k_avg},
  }

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Toy MoE boundary A/B benchmark")
  parser.add_argument("--tokens", type=int, default=12)
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--seq", type=int, default=1)
  parser.add_argument("--dim", type=int, default=2048)
  parser.add_argument("--hidden", type=int, default=1536)
  parser.add_argument("--experts", type=int, default=64)
  parser.add_argument("--topk", type=int, default=4)
  parser.add_argument("--sweep-hidden", type=str, default="")
  parser.add_argument("--preset", choices=["glm47"], default="")
  parser.add_argument("--both-orders", action="store_true")
  args = parser.parse_args()

  if args.preset == "glm47":
    args.dim, args.hidden, args.experts, args.topk = 2048, 1536, 64, 4

  if args.sweep_hidden:
    for hidden in [int(x) for x in args.sweep_hidden.split(",") if x.strip()]:
      if args.both_orders:
        benchmark_both_orders(n_tokens=args.tokens, B=args.batch, T=args.seq, dim=args.dim,
                              hidden=hidden, k=args.topk, num_experts=args.experts)
      else:
        benchmark(n_tokens=args.tokens, B=args.batch, T=args.seq, dim=args.dim,
                  hidden=hidden, k=args.topk, num_experts=args.experts)
  else:
    if args.both_orders:
      benchmark_both_orders(n_tokens=args.tokens, B=args.batch, T=args.seq, dim=args.dim,
                            hidden=args.hidden, k=args.topk, num_experts=args.experts)
    else:
      benchmark(n_tokens=args.tokens, B=args.batch, T=args.seq, dim=args.dim,
                hidden=args.hidden, k=args.topk, num_experts=args.experts)
