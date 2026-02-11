#!/usr/bin/env python3
"""
Toy RMSNorm - tests reduce+broadcast fusion.
After scheduler fusion fix, should be 1 kernel instead of 2.
"""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Timing

def toy_rmsnorm(B=9, T=1, dim=2048, eps=1e-6):
  """
  RMSNorm: x / sqrt(mean(x^2) + eps)

  This tests the scheduler fusion from indexing.py:236.
  Should create 1 fused kernel instead of 2 separate (reduce, then elementwise).
  """
  print(f"\nRMSNorm: ({B}, {T}, {dim})")

  # Pre-allocate inputs
  x = Tensor.randn(B, T, dim, dtype=dtypes.float16).realize()
  weight = Tensor.ones(dim, dtype=dtypes.float16).realize()

  # Measure only the RMSNorm computation
  with Timing("  Time: "):
    rms = (x * x).mean(axis=-1, keepdim=True).sqrt()
    normed = x / (rms + eps)
    result = normed * weight
    result.realize()

  print(f"  Output: {result.shape}")
  print(f"  Expected: 1 kernel (reduce+broadcast fused)")
  return result

def toy_rmsnorm_manual(B=9, T=1, dim=2048, eps=1e-6):
  """Manual RMSNorm to test specific patterns"""
  print(f"\nRMSNorm (manual): ({B}, {T}, {dim})")

  # Pre-allocate inputs
  x = Tensor.randn(B, T, dim, dtype=dtypes.float16).realize()
  weight = Tensor.ones(dim, dtype=dtypes.float16).realize()

  # Measure only the RMSNorm computation
  with Timing("  Time: "):
    x_sq = x * x
    mean_sq = x_sq.sum(axis=-1, keepdim=True) / dim
    rms = (mean_sq + eps).sqrt()
    normed = x / rms
    result = normed * weight
    result.realize()

  print(f"  Output: {result.shape}\n")
  return result

if __name__ == "__main__":
  print("="*70)
  print("TOY RMSNORM - Reduce+Broadcast Fusion Test")
  print("="*70)

  # GLM has 5 RMSNorms per block × 46 blocks = 230 instances
  result1 = toy_rmsnorm()
  result2 = toy_rmsnorm_manual()

  print("="*70)
  print("After indexing.py:236 fix:")
  print("  Before: 2 kernels (reduce, then broadcast+elementwise)")
  print("  After:  1 kernel (fused)")
  print("  Savings: ~230 kernels in full GLM")
  print("="*70)
