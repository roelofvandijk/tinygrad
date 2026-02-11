#!/usr/bin/env python3
"""Profile dynamic vs bucketed toy attention kernels."""
import os
os.environ["DEBUG"] = "3"

from tinygrad import Tensor, dtypes
from toy_attention import toy_attention_qk

print("\n" + "="*70)
print("PROFILING TOY ATTENTION KERNEL")
print("="*70 + "\n")

B, H, T, cache_len, dim = 9, 32, 5, 129, 576
q = Tensor.randn(B, H, T, dim, dtype=dtypes.float16).realize()
cache_k = Tensor.randn(B, 1, 512, dim, dtype=dtypes.float16).realize()

print("Dynamic length kernel:")
qk_dyn = toy_attention_qk(q, cache_k, cache_len=cache_len, min_bucket=0)
qk_dyn.realize()

print("\nBucketed length kernel (min_bucket=64):")
qk_bucket = toy_attention_qk(q, cache_k, cache_len=cache_len, min_bucket=64)
qk_bucket.realize()

print("\n" + "="*70)
print("Compare the kernel names above:")
print("- Dynamic path should encode cache_len directly")
print("- Bucketed path should encode only the static bucket length")
print("="*70 + "\n")
