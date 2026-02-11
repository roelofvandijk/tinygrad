#!/usr/bin/env python3
"""
Complete toy suite for GLM bottleneck patterns.
Run individual components to test optimizations in isolation.
"""

def print_suite_info():
  print("\n" + "="*70)
  print("GLM TOY MODEL SUITE")
  print("="*70)
  print("""
Available tests:

1. toy_attention.py
   - Attention QK matmul with dynamic KV cache
   - Current: ~10 GB/s (3-5% efficiency)
   - Target: 80-100 GB/s
   - Bottleneck: Dynamic cache_len dimension + scattered memory access

2. toy_q4_expert.py
   - Q4_0 quantized expert matmuls (gate, up, down)
   - Current: 46-81 GB/s
   - Target: 145 GB/s (hand-written Metal)
   - Bottleneck: Bitwise dequant ops create suboptimal kernels

3. toy_rmsnorm.py
   - RMSNorm reduce+broadcast fusion
   - Tests indexing.py:236 scheduler fusion fix
   - Should be 1 kernel instead of 2

4. toy_softmax.py
   - Softmax with max+exp+sum fusion
   - Tests reduce fusion opportunities
   - Current: 3 kernels, Target: 1-2 kernels

5. toy_moe_boundaries.py
   - A/B forced fusion boundaries around MoE matmuls
   - Reports avg ms/token and kernels/token for split vs fused modes
   - Use this before changing `.contiguous()` boundaries in model code

Usage:
  # Run individual tests
  python toy_attention.py
  python toy_q4_expert.py
  python toy_rmsnorm.py
  python toy_softmax.py
  python toy_moe_boundaries.py

  # With DEBUG to see kernel details
  DEBUG=2 python toy_attention.py
  DEBUG=6 python toy_attention.py  # Full Metal source

  # With BEAM search to find best opts
  BEAM=2 DEBUG=2 python toy_attention.py

  # Profile specific patterns
  python -m cProfile -s tottime toy_attention.py

Priority order (by impact on GLM 40 tok/s goal):
  1. toy_attention.py - 31% of GLM time, needs 10x speedup
  2. toy_q4_expert.py - 30% of GLM time, needs 2-3x speedup
  3. toy_rmsnorm/softmax - 10% of GLM time, fusion helps
""")
  print("="*70 + "\n")

if __name__ == "__main__":
  print_suite_info()

  print("Run individual toy_*.py files to test each component.")
  print("Do NOT run the full GLM model until components are optimized.\n")
