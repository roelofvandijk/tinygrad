# Challenge: youtu-llm:2b-Q4_0 at 51 tok/s, need 80+

## The Model
youtu-llm:2b-Q4_0 — DeepSeek-V2/MLA architecture, 32 blocks, 0.69 GB params, Q4_0 quantized.
Theoretical bandwidth limit at 100 GB/s = 145 tok/s. Currently 51 tok/s (19.6ms/tok).

## The Bottleneck: 586 kernels per token

All 586 kernels fit in 1 ICB (indirect command buffer). Average 33us per kernel (execution + dispatch).
The GPU runs at 100-160 GB/s when actually computing. The problem is pure dispatch overhead.

## Per-Block Anatomy (18 kernels × 32 blocks + 10 overhead = 586)

Each block has ONE scheduling barrier: `cache_k.assign(k_new).realize()` mid-attention.
This splits each block into two scheduling regions. The `.contiguous()` at block end is elided (no-op).

Verified via VIZ=-1 + extra/viz/cli.py:

```
Region 1 (7 kernels): Q/K computation → cache write
  r_16_128n1         — attn_norm RMSNorm reduce (→1 scalar)                    14us
  r_1536_16_4_16     — FUSED: attn_q_a + attn_kv_a Q4_0 dequant+matmul       100us 162GB/s
  r_16_96            — q_a_norm RMSNorm reduce (→1 scalar)                      12us
  r_32_32_3_48_16    — attn_q_b Q4_0 dequant+matmul                           192us  63GB/s
  r_2_18_8_16_2_32_4 — K absorption: q_nope @ k_b^T + RoPE cat                 34us  64GB/s
  E_9_32_2n1         — kv_a_norm ewise + RoPE + cat → k_new                    11us
  E_9_16_4n1         — cache assign: copy k_new into cache[start_pos]            9us

Region 2 (11 kernels): attention + FFN
  r_16_32            — kv_a_norm RMSNorm reduce (→1 scalar)                    11us
  r_16_sp_16_36      — QK matmul (attention scores)                             13us
  r_16_sp            — softmax exp().sum()                                       9us
  r_8_4_16_4_4_sp    — softmax_div + attn@V + V_absorb fused                   11us
  r_2048_16_32n1     — attn_output Q4_0 dequant+matmul                         41us  52GB/s
  r_2048_16_4_16     — residual + Q4_0 matmul (fused add)                      87us 152GB/s
  r_16_128           — ffn_norm RMSNorm reduce (→1 scalar)                      16us
  r_64_32_3_...      — FUSED: ffn_gate+silu * ffn_up Q4_0                     545us 118GB/s
  r_2048_16_12_16    — ffn_down Q4_0 + residual                                335us 104GB/s
  r_16_128n1         — next block's attn_norm reduce                            14us
```

## What's Already Fused (scheduler is doing well)

- RMSNorm elementwise (x * factor * weight) fused INTO consumer Q4_0 matmuls
- attn_q_a + attn_kv_a matmuls fused into one kernel
- ffn_gate + silu + ffn_up fused into one kernel
- Residual adds fused into Q4_0 matmuls
- `.contiguous()` is elided (output already contiguous)

## What Can't Fuse (verified)

- RMSNorm reduces (→1 scalar output) can't merge into matmuls (different output shape)
- Softmax sum can't merge into QK (reduce dependency)
- Cache assign is a pure copy, necessary for KV cache

## The Key Finding

```
Kernels from .realize() calls inside blocks: 567
Kernels from final lazy .schedule():          15
Total:                                       582
```

The raw graph (no realize barriers) schedules to **15 kernels for all 32 blocks**.
But each block's `cache.assign(k_new).realize()` forces a scheduling split.
567/32 = 17.7 kernels per realize call.

**Each block has its own independent cache.** Block N's cache write doesn't affect block M's cache read. The only intra-block dependency is: within one block, the cache must be written before it's read for attention.

## Approaches Tried

| Approach | Kernels | Notes |
|----------|---------|-------|
| Current (per-block realize) | 586 | 51 tok/s |
| Deferred cache (cat old + new) | 548 | .cat() prevents fusions, only -6.5% |
| Remove .contiguous() | 3624 | Scheduler partitions cross-block graph into 13/blk |
| No cache at all (theoretical) | 15 | Not achievable with autoregressive decode |

## The Question

How do we get from 586 → ~380 kernels (= 80 tok/s)?

The scheduler PROVES it can do 15 kernels when it sees the full graph. The barrier is the per-block `cache.assign(k_new).realize()`. Each block's cache is independent — there's no cross-block cache dependency.

Options we see but haven't cracked:
1. **Two-phase execution**: Phase 1 = compute all k_new + cache writes for all 32 blocks (small realizes). Phase 2 = all attention+FFN lazily (→15 kernels). But phase 1 still needs the residual chain from previous block's FFN output.
2. **JIT-level batching**: Teach the JIT to batch multiple scheduling regions into one `.schedule()` call when buffers don't conflict.
3. **Eliminate cache read dependency**: The attention needs `cache[0:start_pos+1]` which includes current token. If we cat `cache[0:start_pos]` with `k_new`, we bypass the write→read dependency, but `.cat()` kills fusion.
4. **Something we haven't thought of.**

## Reproduce

```bash
# Setup (macOS, Apple Silicon)
python3 -m venv .venv2 && source .venv2/bin/activate && pip install -e .

# Benchmark (51 tok/s baseline)
JIT_BATCH_SIZE=700 python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4_0" --benchmark 20

# Profile (kernel breakdown)
VIZ=-1 JIT_BATCH_SIZE=700 python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4_0" --benchmark 3
PYTHONPATH=. python extra/viz/cli.py --profile --device METAL

# Inspect specific kernel AST
PYTHONPATH=. python extra/viz/cli.py --kernel 'r_2048_16_4_16n1' --select 'View Base AST'

# Count realize vs lazy split
python test_realize_count.py

# Verify .contiguous() is no-op and no-contiguous explodes
python test_contig_vs_realize.py

# DEBUG=2 for scheduling pattern
DEBUG=2 JIT_BATCH_SIZE=700 python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4_0" --benchmark 5

# Smoke test (correctness)
python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4_0" --prompt "Hello" --count 10
```

## Key Files

- `tinygrad/apps/mla.py` — MLA attention + FFN (the hot loop)
- `tinygrad/apps/llm.py` — Transformer forward, JIT wrapper
- `tinygrad/apps/quantized.py` — Q4_0 QuantizedLinear
- `tinygrad/engine/jit.py` — JIT capture and replay
- `tinygrad/engine/schedule.py` — Graph → kernel scheduling
- `tinygrad/codegen/opt/heuristic.py` — Kernel optimization heuristics
