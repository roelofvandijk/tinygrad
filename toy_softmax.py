#!/usr/bin/env python3
"""
Toy Softmax - tests reduce fusion and numerical stability.
"""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Timing

def toy_softmax(B=9, H=32, T=5, S=128):
  """
  Softmax over attention scores.
  Pattern: exp(x - max) / sum(exp(x - max))

  Real GLM: 3 kernels → 2 kernels after scheduler fusion
  - Reduce to find max
  - Broadcast and compute exp(x - max) / sum(exp)
  - After fusion: max+exp+sum in fewer kernels
  """
  print(f"\nSoftmax: ({B}, {H}, {T}, {S})")

  # Pre-allocate input
  x = Tensor.randn(B, H, T, S, dtype=dtypes.float16).realize()

  # Measure only softmax computation
  with Timing("  Time: "):
    x_max = x.max(axis=-1, keepdim=True)
    x_shifted = x - x_max
    exp_x = x_shifted.exp()
    result = exp_x / exp_x.sum(axis=-1, keepdim=True)
    result.realize()

  print(f"  Output: {result.shape}")
  print(f"  Expected: 2-3 kernels\n")
  return result

def toy_softmax_fused(B=9, H=32, T=5, S=128):
  """
  Single-expression softmax - may fuse better.
  """
  print(f"\nSoftmax (fused): ({B}, {H}, {T}, {S})")

  # Pre-allocate input
  x = Tensor.randn(B, H, T, S, dtype=dtypes.float16).realize()

  # Measure only softmax computation
  with Timing("  Time: "):
    result = (x - x.max(axis=-1, keepdim=True)).exp()
    result = result / result.sum(axis=-1, keepdim=True)
    result.realize()

  print(f"  Output: {result.shape}\n")
  return result

if __name__ == "__main__":
  print("="*70)
  print("TOY SOFTMAX - Reduce Fusion Test")
  print("="*70)

  result1 = toy_softmax()
  result2 = toy_softmax_fused()

  print("="*70)
  print("Optimization opportunity:")
  print("  Fuse max+exp+sum operations")
  print("  Current: 3 kernels → Target: 1-2 kernels")
  print("="*70)
