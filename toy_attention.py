#!/usr/bin/env python3
"""Toy attention benchmark for dynamic vs bucketed KV cache lengths."""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Timing

def _next_pow2(x:int) -> int: return 1 if x <= 1 else 1 << (x-1).bit_length()
def _bucket_len(cache_len:int, max_context:int, min_bucket:int) -> int:
  return min(max_context, max(min_bucket, _next_pow2(cache_len)))

def toy_attention_qk(q:Tensor, cache_k:Tensor, cache_len:int, min_bucket:int=0) -> Tensor:
  """
  QK matmul with optional decode bucketing.
  - min_bucket=0: dynamic length (shape includes cache_len)
  - min_bucket>0: static bucket length, mask invalid tail (removes start_pos+1 style dynamic axis)
  """
  attn_len = cache_len if min_bucket == 0 else _bucket_len(cache_len, cache_k.shape[2], min_bucket)
  k = cache_k[:, :, :attn_len, :]
  qk = q.matmul(k.transpose(-2, -1))
  if attn_len > cache_len:
    valid = Tensor.arange(attn_len, requires_grad=False, device=q.device).reshape(1, 1, 1, attn_len) < cache_len
    qk = valid.where(qk, qk.full_like(float("-inf")))
    qk = qk[:, :, :, :cache_len]
  return qk

def benchmark_growing_cache(B=9, H=32, T=5, dim=576, max_context=512, min_bucket=64):
  """Benchmark with growing cache like real decode, comparing dynamic vs bucketed lengths."""
  print("="*70)
  print("BENCHMARK: Growing KV Cache (dynamic vs bucketed)")
  print("="*70)
  print(f"\nQ shape: ({B}, {H}, {T}, {dim}), K cache shape: ({B}, 1, {max_context}, {dim})")
  print(f"Bucket mode: min_bucket={min_bucket} (removes per-token dynamic K length)\n")

  q = Tensor.randn(B, H, T, dim, dtype=dtypes.float16).realize()
  cache_k = Tensor.randn(B, 1, max_context, dim, dtype=dtypes.float16).realize()

  for cache_len in [65, 129, 257, 511]:
    print(f"--- Cache length: {cache_len} ---")
    with Timing("  Dynamic QK: "):
      qk_dyn = toy_attention_qk(q, cache_k, cache_len, min_bucket=0)
      qk_dyn.realize()
    with Timing("  Bucketed QK: "):
      qk_bucket = toy_attention_qk(q, cache_k, cache_len, min_bucket=min_bucket)
      qk_bucket.realize()
    max_err = (qk_dyn - qk_bucket).abs().max().item()
    print(f"  Max error: {max_err:.3e}  | Output: {qk_bucket.shape}\n")

  print("="*70)
  print("Dynamic: kernel shape tracks cache_len (start_pos+1 style axis)")
  print("Bucketed: kernel shape tracks only bucket (fewer compiled kernels, no per-token dynamic axis)")
  print("="*70 + "\n")

if __name__ == "__main__":
  benchmark_growing_cache()
