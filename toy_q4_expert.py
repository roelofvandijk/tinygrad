#!/usr/bin/env python3
"""
Toy Q4_0 expert matmul - reproduces 46-81 GB/s bottleneck.
Target: 145 GB/s (like hand-written Metal).
"""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Timing

def toy_q4_expert_gate(n_sel=4, hidden=2048, out_dim=10240):
  """
  Q4_0 expert gate matmul.
  Real GLM: r_384_16_4_4_4_16 @ 46 GB/s (should be 145 GB/s)

  Pattern: Indexed expert weights with inline Q4_0 dequant
  - n_sel: number of selected experts (B*T*K, typically 4-8)
  - hidden: input dimension (2048 in GLM)
  - out_dim: output dimension (10240 in GLM)
  """
  print(f"\nQ4_0 Expert Gate: ({n_sel}, {hidden}) @ ({hidden}, {out_dim})")

  # Input activations (fp16)
  x = Tensor.randn(n_sel, hidden, dtype=dtypes.float16).realize()

  # Q4_0 weights: 32 elements/block, 2 bytes scale + 16 bytes data = 18 bytes/block
  bpr = hidden // 32  # blocks per row
  blocks = Tensor.randint(out_dim * bpr, 18, low=0, high=256, dtype=dtypes.uint8).realize()
  blocks = blocks.reshape(out_dim, bpr, 18)

  print(f"  Blocks: ({out_dim}, {bpr}, 18)")
  print(f"  Total weight data: {out_dim * bpr * 18 / 1024:.1f} KB")

  # Q4_0 packed-dot with inline dequant
  with Timing("  Matmul time: "):
    scale = blocks[:, :, :2].bitcast(dtypes.float16)
    packed = blocks[:, :, 2:]

    x_pairs = x.reshape(n_sel, 1, bpr, 2, 16)
    x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]

    # THE BOTTLENECK: Bitwise ops + scattered reads
    lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
    hi = packed.rshift(4).cast(dtypes.float16) - 8.0

    result = (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, out_dim, bpr * 16).sum(axis=-1)
    result.realize()

  print(f"  Output: {result.shape}\n")
  return result

def toy_q4_expert_down(n_sel=4, hidden=10240, out_dim=2048):
  """
  Q4_0 expert down projection - output of gate*up back to model dim.
  Real GLM: r_2048_16_2_2_3_16 @ 32 GB/s (should be 145 GB/s)

  This kernel is even slower than gate/up because it reads larger fp16 inputs.
  """
  print(f"\nQ4_0 Expert Down: ({n_sel}, {hidden}) @ ({hidden}, {out_dim})")

  # Input: result of gate * up (fp16, larger dimension)
  x = Tensor.randn(n_sel, hidden, dtype=dtypes.float16).realize()

  # Q4_0 weights
  bpr = hidden // 32
  blocks = Tensor.randint(out_dim * bpr, 18, low=0, high=256, dtype=dtypes.uint8).realize()
  blocks = blocks.reshape(out_dim, bpr, 18)

  print(f"  Blocks: ({out_dim}, {bpr}, 18)")

  # Q4_0 packed-dot
  with Timing("  Matmul time: "):
    scale = blocks[:, :, :2].bitcast(dtypes.float16)
    packed = blocks[:, :, 2:]

    x_pairs = x.reshape(n_sel, 1, bpr, 2, 16)
    x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]

    lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
    hi = packed.rshift(4).cast(dtypes.float16) - 8.0

    result = (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, out_dim, bpr * 16).sum(axis=-1)
    result.realize()

  print(f"  Output: {result.shape}\n")
  return result

if __name__ == "__main__":
  print("="*70)
  print("TOY Q4_0 EXPERT MATMULS")
  print("="*70)

  # GLM-4.7 dimensions: 4 experts per token, hidden=10240, dim=2048
  gate = toy_q4_expert_gate(n_sel=4, hidden=2048, out_dim=10240)
  down = toy_q4_expert_down(n_sel=4, hidden=10240, out_dim=2048)

  print("="*70)
  print("Target: 145 GB/s (hand-written Metal performance)")
  print("Current: 46-81 GB/s (tinygrad generated)")
  print("Gap: 1.8-3.2x speedup needed")
  print("="*70)
