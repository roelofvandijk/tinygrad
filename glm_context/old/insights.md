# Why llama.cpp Is Faster Than tinygrad — Fundamental Analysis

## The Three Gaps

| | llama.cpp | tinygrad | Impact |
|--|-----------|----------|--------|
| **Kernels/token** | ~50-100 (youtu), ~200 (GLM) | 586 (youtu), 1700 (GLM) | 6-12x more dispatch overhead |
| **Per-kernel overhead** | ~0 (pipelined, selective barriers) | ~34us (every kernel barrier-serialized) | 60-80% of wall time is overhead |
| **Kernel quality** | Hand-tuned SIMD+coalesced+multi-row | Decent (100+ GB/s) but not hand-tuned | 1.5-2x bandwidth gap on Q4_0 |

**Bottom line**: llama.cpp dispatches 6-12x fewer operations, each with near-zero overhead, and each individually more efficient. tinygrad's compiler generates reasonable kernels but wraps them in 6x the dispatch overhead.

## Gap 1: Kernel Count (6-12x)

llama.cpp has ~10-15 hand-written operations per layer. tinygrad generates 18 kernels/layer because:

1. **RMSNorm = 2 kernels** (reduce + elementwise). llama.cpp: 1 fused kernel.
2. **Softmax = 2-3 kernels** (QK + exp+sum + div). llama.cpp: online softmax in 1.
3. **MoE routing = 3+ kernels** (gate + topk + gather). llama.cpp: fused topk-moe (CUDA) or argsort (Metal).
4. **Expert dispatch = 4+ kernels** (2 gathers + probs + fused dequant+matmul). llama.cpp: 1 `mul_mat_id` per projection.
5. **Cache ops = 2-5 kernels** (assign + contiguous boundaries). llama.cpp: direct pointer writes.

The scheduler can't fuse across `.realize()` / `.contiguous()` boundaries. These exist because without them, the scheduler sees the entire 32-layer graph and explodes to 3600+ kernels with terrible partitioning.

### The 18-kernel structure is a local optimum

- Removing `.contiguous()` → 3600+ kernels (scheduler can't partition large graphs)
- Adding more barriers → 652+ kernels (fragments fusion)
- Combining projections (q_a+kv_a, gate+up) → same or more kernels (split consumption duplicates, breaks parallel reduce fusion)
- The current 18/layer is the best the scheduler can do with current fusion heuristics

## Gap 2: Dispatch Overhead (~34us vs ~0)

Every kernel in a Metal ICB gets `.setBarrier()` (`metal.py:50`), fully serializing all kernels. Independent operations (q_a and kv_a projections, gate and up matmuls) can't overlap.

**llama.cpp's approach** (`ggml-metal-common.cpp:280-359`):
- Track read/write buffer ranges per kernel
- Only insert `memoryBarrierWithScope` when RAW/WAR/WAW conflict detected
- Look ahead N_FORWARD=8 nodes to reorder and maximize concurrency windows
- Result: independent kernels pipeline freely, ~0 per-kernel overhead

**tinygrad already has the info needed**: `CompiledRunner.p` has `outs` (written buffers) and `ins` (read buffers). The fix is ~50 lines in `metal.py`: build read/write sets, only barrier on conflict.

**Expected impact**: 34us → ~15us per kernel. youtu: 586×15us + 7ms = 15.8ms = **63 tok/s** (from 56).

## Gap 3: Individual Kernel Quality

### Dense matvec (QuantizedLinear)
tinygrad's Q4_0 packed-dot kernels achieve **103-147 GB/s** (45-64% of theoretical 229 GB/s). llama.cpp's hand-tuned MSL kernel gets **229 GB/s** via:
- **ushort reads**: 2 bytes → 4 nibbles (tinygrad: 1 byte → 2 nibbles)
- **SIMD reduction**: `simd_sum` hardware shuffle (tinygrad: threadgroup memory)
- **Multi-row**: amortizes activation reads across nr0 output rows
- **Stride-4 block interleaving**: 4 adjacent threads read consecutive blocks (coalesced)
- **The 1/256 trick**: leaves nibble in high byte, multiplies by 1/256 instead of explicit shift

**Ushort was tried at tensor level and codegen level — both failed**:
- Tensor bitcast: shape mismatch, wrong nibble access patterns, NaN/inf
- Codegen pattern rewrite: confirmed ushort loads generated, but **-4.8% regression** (48.8 vs 51.3 tok/s)
- Root cause: ALU overhead of unpacking ushort → 4 nibbles exceeds the bandwidth saving. Metal's memory controller already coalesces byte loads efficiently. The real llama.cpp advantage is the full kernel design (SIMD reduction, multi-row, interleaving), not just the load width.

### MoE expert matvec (QuantizedExpertWeights)
This is where the gap is largest. tinygrad fuses dequant+matmul into one kernel, but:
- **MV heuristic fails**: requires `MUL(INDEX, INDEX)`, sees `MUL(dequant_chain(INDEX), INDEX)`
- **GROUPTOP fails**: output dims (6×2048 = 12288) exceed 2048 threshold
- Result: **serial reduction at 2-15 GB/s** instead of 80+ GB/s

llama.cpp's `mul_mat_id`: template wrapper around optimized matvec. Grid Z = experts × tokens. Each Z-threadgroup runs full SIMD-parallel matvec for one expert. No gather, no serial loop. **100+ GB/s**.

## Current Performance

| Model | tok/s | ms/tok | Kernels | Theoretical | Efficiency |
|-------|-------|--------|---------|-------------|------------|
| youtu-llm:2b-Q4 | **56** | 17.8 | 586 | 145 (6.9ms) | 39% |
| deepseek-v2-lite | **26.5** | 37.7 | 840 | — | — |
| GLM-4.7-Flash | **18** | 55.6 | 1700 | 80 (12.4ms) | 22% |

### Per-token time breakdown (youtu, 17.8ms)

| Component | Time | Notes |
|-----------|------|-------|
| GPU compute (weight reads) | ~7ms | 0.69 GB @ 100 GB/s |
| ICB dispatch + barrier | ~8ms | 586 kernels × ~14us within ICB |
| ICB transition overhead | ~2ms | Between 5 ICB batches |
| Python/JIT | ~1ms | JIT replay loop |

## Ranked Action Plan

| # | Action | Expected | Risk | Status |
|---|--------|----------|------|--------|
| 1 | **Metal ICB barrier removal** | 56 → 63+ tok/s | Medium | Not started. llama.cpp reference code available. ~50 lines in metal.py |
| 2 | **Custom mul_mat_id for MoE** | GLM 18 → 30+ tok/s | Medium | Prototype exists (Q4K MoE MSL). Needs plugged-in benchmark |
| 3 | **Scheduler fusion** | 586 → ~400 kernels | High | Blocked on scheduler understanding. Norm+matmul, QK+softmax |
| 4 | ~~Ushort codegen~~ | ~~56 → 65 tok/s~~ | ~~Low~~ | **Tried, failed (-4.8%)**. Metal coalesces bytes already |

### The math (youtu, all improvements combined)

- Barrier removal: 34us → 15us per kernel
- Scheduler fusion: 586 → 400 kernels
- Compute stays ~7ms (ushort didn't help)
- Total: 400 × 0.015 + 7 = **13ms = 77 tok/s**

## What Was Tried and Failed

| Attempt | Result | Why |
|---------|--------|-----|
| Combined q_a+kv_a projection | Same kernels | Split consumption → scheduler duplicates |
| Combined gate+up projection | +1 kernel | **Breaks** parallel reduce fusion (gate+up same shape = 1 kernel already) |
| Ushort loads (codegen level) | -4.8% | ALU overhead > bandwidth saving. Metal coalesces bytes |
| Ushort loads (tensor bitcast) | NaN/inf | Wrong nibble access pattern |
| Remove .contiguous() | 3600+ kernels | Scheduler can't partition large graphs |
| Split KV cache (avoid cat) | +66 kernels | 2× realize barriers fragment fusion |
| MOE_FUSED custom UOp kernels | 6x slower | Dispatch overhead, scattered byte reads |
| Per-expert dispatch | -18% | 6× more kernel dispatches |
| MV heuristic relaxation | 1 match | Fused dequant ranges don't align |
| GROUPTOP on fused MoE | +4% | 768KB threadgroup memory > 32KB limit |
| BEAM on fused dequant | No help | Can't fix structural problem |
| FP16 softmax with .max() | +32 kernels | Max adds kernels, no speedup |

## Key Architectural Insights

1. **Dispatch overhead dominates, not kernel quality**: 586 × 34us = 20ms overhead. Compute = 7ms. Cutting overhead in half would be a bigger win than doubling kernel bandwidth.

2. **The scheduler fuses THROUGH .cat() but NOT through .realize()**: Cat is just a tensor op the scheduler optimizes through. Realize is an execution barrier. Don't try to eliminate cats by splitting caches.

3. **Parallel reduce fusion is free**: Two reduces from same input with same output shape → 1 kernel (gate+up FFN). Don't combine them manually — you'll break it.

4. **Fused dequant kills the optimizer**: `MUL(dequant_chain(INDEX), INDEX)` fails MV heuristic. `prod(output) > 2048` fails GROUPTOP. The optimizer has no good path for fused quantized matvecs. This is THE fundamental limitation for MoE performance.

5. **Cross-block scheduling is already smart**: The scheduler prefetches next block's initial ops (norm, kv_a, rope, cache assign) within the current block's schedule call. 18 kernels/layer includes 5 from the next block.

6. **Isolated micro-benchmarks lie**: 4.6x speedup in isolation → same speed in full model. The scheduler's fusion decisions depend on the entire graph, not individual operations.

7. **Metal ICB profiler divides time evenly**: Within an ICB, all kernels show ~37us regardless of actual work. Only warm-up (pre-JIT) times reflect real per-kernel bandwidth. Trust wall-clock benchmarks, not ICB kernel profiles.

## Per-Layer Kernel Map (youtu, 18 kernels)

**Attention (10 kernels)**:
1. `r_32_64n1` — attn_norm RMSNorm reduce
2. `r_1536_32_2_16n1` — q_a Q4_0 matmul (2048→1536)
3. `r_32_48` — q_a_norm reduce
4. `r_32_32_3_48_16n1` — q_b Q4_0 matmul (1536→3072)
5. `r_288_2_16_4_4_8` — kv_a Q4_0 matmul + k_b absorption + cat
6. `r_(T)_16_4_4_36` — QK scores + softmax exp
7. `r_4_4_(T)` — softmax sum
8. `r_8_4_16_4_4_(T)` — attn@V + softmax div
9. `r_128_16_4_4_32n1` — V absorption (v_b matmul)
10. `r_2048_32_2_16n1` — output proj + residual

**FFN (3 kernels)**:
11. `r_32_64` — ffn_norm reduce
12. `r_64_32_3_64_16_64_16n1` — gate+up Q4_0 fused + silu + mul (parallel reduce)
13. `r_2048_32_6_16n1` — down Q4_0 + residual

**Next block prefetch (5 kernels)**:
14-18. Next block's attn_norm, kv_a matmul, kv_a_norm, RoPE elementwise, KV cache assign
