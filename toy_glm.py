#!/usr/bin/env python3
"""
Toy GLM model with exact bottleneck patterns for fast iteration.
Reproduces the slow kernels without full model overhead.
"""
from tinygrad import Tensor, dtypes
import numpy as np

class ToyMLAAttention:
  """Minimal MLA attention with dynamic KV cache - reproduces the 186ms @ 5GB/s kernel"""
  def __init__(self, dim=2048, n_heads=32, kv_lora_rank=512, max_context=1024):
    self.dim = dim
    self.n_heads = n_heads
    self.kv_lora_rank = kv_lora_rank
    self.max_context = max_context
    self.head_dim = dim // n_heads  # 64

  def __call__(self, x: Tensor, start_pos: int):
    """
    Reproduces: q @ k.T where k has dynamic dimension (start_pos+T)
    This creates the slow r_288_2_(start_pos+1)_5_32_2_2_6_576 kernel
    """
    B, T, _ = x.shape
    cache_dim = self.kv_lora_rank + 64

    # Simulate Q projection
    q = Tensor.randn(B, self.n_heads, T, cache_dim, dtype=dtypes.float16)

    # Simulate KV cache with dynamic length
    if not hasattr(self, 'cache_k'):
      self.cache_k = Tensor.zeros(B, 1, self.max_context, cache_dim, dtype=dtypes.float16).contiguous().realize()

    # Update cache
    k_new = Tensor.randn(B, T, cache_dim, dtype=dtypes.float16).reshape(B, 1, T, cache_dim)
    self.cache_k[:, :, start_pos:start_pos+T, :].assign(k_new).realize()

    # THE BOTTLENECK: Dynamic slice creates variable-shape kernel
    k = self.cache_k[:, :, 0:start_pos+T, :]

    # QK matmul - this is the slow kernel!
    qk = q.matmul(k.transpose(-2, -1)) / (self.head_dim ** 0.5)

    # Mask and softmax
    if T > 1:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=dtypes.float16).triu(start_pos+1)
      qk = qk + mask
    attn = qk.softmax(-1)

    # Weighted sum (simplified)
    v = k[:, :, :, :self.kv_lora_rank]  # Use just kv_lora_rank dimension for V
    out = attn.matmul(v)
    # Project to output dim (in real MLA this is absorbed V projection)
    out = out.reshape(B, T, self.n_heads * self.kv_lora_rank)
    # Simple linear projection back to dim
    out = out[:, :, :self.dim]  # Truncate for toy model
    return out

class ToyQ4ExpertWeights:
  """Minimal Q4_0 expert matmul - reproduces 46-81 GB/s kernels"""
  def __init__(self, num_experts=4, hidden=2048, dim=2048):
    self.num_experts = num_experts
    self.hidden = hidden
    self.dim = dim

    # Q4_0: 32 elements per block, 2 bytes scale + 16 bytes data = 18 bytes/block
    blocks_per_row = hidden // 32
    # Create fake Q4_0 data: (num_experts, dim, blocks_per_row, 18)
    self.expert_blocks = Tensor.randint(num_experts * dim * blocks_per_row, 18,
                                        low=0, high=256, dtype=dtypes.uint8)
    self.expert_blocks = self.expert_blocks.reshape(num_experts, dim, blocks_per_row, 18)

  def __call__(self, sel: Tensor, x: Tensor):
    """
    Q4_0 packed-dot expert matmul - reproduces r_384_16_4_4_4_16 @ 46 GB/s
    sel: (B, T, K) expert indices
    x: (B, T, hidden) activations
    """
    B, T, K = sel.shape
    n_sel = B * T * K

    # Gather expert weights
    sel_blocks = self.expert_blocks[sel.reshape(-1)]  # (n_sel, dim, bpr, 18)

    # Q4_0 packed-dot with inline dequant
    blocks = sel_blocks.reshape(n_sel, self.dim, self.hidden // 32, 18)
    scale = blocks[:, :, :, :2].bitcast(dtypes.float16)
    packed = blocks[:, :, :, 2:]

    x_flat = x.reshape(n_sel, self.hidden).cast(dtypes.float16)
    x_pairs = x_flat.reshape(n_sel, 1, self.hidden // 32, 2, 16)
    x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]

    # THE BOTTLENECK: Bitwise ops create patterns MATVEC handles suboptimally
    lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
    hi = packed.rshift(4).cast(dtypes.float16) - 8.0

    # Matmul with inline dequant
    result = (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, self.dim, self.hidden).sum(axis=-1)
    return result.reshape(B, T, K, self.dim)

class ToyGLM:
  """Minimal GLM with just the bottleneck operations"""
  def __init__(self, dim=2048, n_heads=32, num_experts=4, hidden=10240):
    self.attention = ToyMLAAttention(dim, n_heads)
    self.gate_experts = ToyQ4ExpertWeights(num_experts, dim, hidden)
    self.up_experts = ToyQ4ExpertWeights(num_experts, dim, hidden)
    self.down_experts = ToyQ4ExpertWeights(num_experts, hidden, dim)

  def __call__(self, x: Tensor, start_pos: int):
    """One transformer block with attention + MoE"""
    B, T, dim = x.shape

    # 1. MLA Attention - creates the slow 186ms kernel
    attn_out = self.attention(x, start_pos)
    # Project back to dim if needed
    if attn_out.shape[-1] != dim:
      attn_out = attn_out.reshape(B, T, dim)  # Simple reshape for toy model
    x = x + attn_out[:, :, :dim]  # Ensure matching dims

    # 2. MoE FFN - creates the slow 46-81 GB/s Q4_0 kernels
    # Simulate expert selection (normally from router)
    K = 2  # experts per token
    sel = Tensor.randint(B * T * K, low=0, high=self.gate_experts.num_experts, dtype=dtypes.int32).reshape(B, T, K)

    # Expert matmuls - these create the bottleneck Q4_0 kernels
    gate = self.gate_experts(sel, x).silu()  # r_384_16_4_4_4_16 @ 46 GB/s
    up = self.up_experts(sel, x)             # r_384_16_4_4_4_16n1 @ 48 GB/s
    gated = gate * up
    expert_out = self.down_experts(sel, gated)  # r_2048_16_2_2_3_16 @ 32 GB/s

    # Weighted sum (simplified)
    probs = Tensor.ones(B, T, K, 1, dtype=dtypes.float16) / K
    x = x + (expert_out * probs).sum(axis=2)

    return x

def benchmark_toy_model(n_tokens=10):
  """Fast benchmark of toy model"""
  print(f"\n{'='*60}")
  print(f"TOY GLM BENCHMARK ({n_tokens} tokens)")
  print(f"{'='*60}\n")

  # Create tiny model
  model = ToyGLM(dim=2048, n_heads=32, num_experts=4, hidden=10240)

  # Run tokens
  B = 1
  x = Tensor.randn(B, 1, 2048, dtype=dtypes.float16)

  from tinygrad.helpers import Timing
  times = []
  for i in range(n_tokens):
    def save_time(ns, times=times):
      times.append(ns/1e6)
      return ""
    with Timing(f"Token {i+1}: ", on_exit=save_time):
      out = model(x, start_pos=i)
      out.realize()

  # Stats
  warmup = times[:3]
  steady = times[3:]
  if steady:
    avg_ms = sum(steady) / len(steady)
    tok_per_sec = 1000 / avg_ms
    print(f"\n{'='*60}")
    print(f"Warmup: {warmup[:3]} ms")
    print(f"Steady: avg {avg_ms:.1f} ms/token ({tok_per_sec:.1f} tok/s)")
    print(f"{'='*60}\n")

  return model, out

if __name__ == "__main__":
  import sys
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
  model, out = benchmark_toy_model(n)
  print(f"Output shape: {out.shape}")
