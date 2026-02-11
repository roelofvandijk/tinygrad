# tinygrad LLM Optimization — Start Here

## Current Performance (Feb 10, 2026)

| Model | Params | tok/s | ms/tok | Kernels/tok | Theoretical | Efficiency |
|-------|--------|-------|--------|-------------|-------------|------------|
| youtu-llm:2b-Q4_0 | 0.69 GB | **56** | 17.8 | 586 (5 ICBs) | 145 (6.9ms) | 39% |
| deepseek-v2-lite | ~2 GB | **26.5** | 37.7 | 840 (5 ICBs) | — | — |
| GLM-4.7-Flash Q4_0 | 1.24 GB | **20.0** | 49.0 | 1358 (6 ICBs) | 80 (12.4ms) | 25% |

All models use MLA (Multi-head Latent Attention). GLM and deepseek-v2-lite also use MoE.

**Latest**: GLM Q4_0 achieves 20 tok/s with **pure tinygrad DSL** (no custom Metal/MSL kernels). Key optimization: merged gate_up experts (14 → 20 tok/s, +43%). Target: 35 tok/s (llama.cpp parity).

## 10 Key Findings

1. **Dispatch overhead dominates, not kernel quality.** 586 kernels × 34us = 20ms overhead vs 7ms compute. Cutting overhead matters more than doubling kernel bandwidth.

2. **JIT batching is the #1 perf lever for custom kernels.** Making Q4K/Q6K runners extend `CompiledRunner` instead of `Runner` collapsed 417 dispatch items → 5, giving 1.8x speedup (31 → 55.5 tok/s).

3. **Fused dequant breaks the optimizer.** `MUL(dequant_chain(INDEX), INDEX)` fails the MV heuristic's `MUL(INDEX, INDEX)` pattern match. GROUPTOP fails because output dims > 2048 threshold. Result: serial reduction at 2-15 GB/s.

4. **Q4_0 is 11-22% faster than Q4_K for MoE models.** Simpler dequant fuses better with tinygrad's optimizer. One scale per 32 elements vs hierarchical 8-sub-block Q4_K.

5. **Isolated micro-benchmarks don't predict full-model behavior.** 4.6x speedup in isolation → same speed in full model. The scheduler's fusion decisions depend on the entire graph.

6. **The scheduler fuses THROUGH `.cat()` but NOT through `.realize()`.** Cat is a tensor op the scheduler optimizes through. Realize is an execution barrier. Don't try to eliminate cats by splitting caches.

7. **Metal ICB profiler divides time evenly among kernels.** Within an ICB, all kernels show ~37us regardless of work. Only warm-up (pre-JIT) times reflect real per-kernel bandwidth.

8. **Barriers are necessary evils.** Without `.contiguous()` barriers: 3600+ kernels with terrible partitioning. Without `.realize()` on cache writes: incorrect output. The current 18 kernels/layer is a local optimum.

9. **Data amplification: fp16 cache reads 3.56x more than Q4_0 packed-dot.** For batch=1, the per-token cost of reading cached fp16 weights dwarfs the one-time dequant saving. Q4_0 packed-dot was the correct fix.

10. **ICB barrier removal gave 0% speedup.** Nearly all 586 kernels are data-dependent (each output feeds next input). Conflict-based barriers re-insert on almost every kernel. Would need kernel reordering to create concurrency.

11. **GROUP heuristic for Q4_0 dequant has ~50 GB/s ceiling.** Detecting bitwise ops (AND/SHR) in reduce chain and applying GROUP+LOCAL+UPCAST gets individual expert kernels from 34→42 GB/s (down) and 31→50 GB/s (gate/up). But scattered byte reads in generated code prevent reaching 100+ GB/s. The `.contiguous()` splits needed to enable GROUP add ~26 kernels, partially offsetting gains. Net: +7% tok/s on ds2-lite.

## What Worked

| Change | Impact | Details |
|--------|--------|---------|
| Q4K/Q6K CompiledRunner | **1.8x** (31→55.5 tok/s) | JIT ICB batching: 417→5 dispatch items |
| Q6K MoE MSL kernel | **+48%** (12→18 tok/s GLM) | llama.cpp access pattern for expert down proj |
| MoE down proj `.contiguous()` break | **+41%** (18→25.5 tok/s ds2) | Separate kernel gets proper parallelism |
| Shared expert `.contiguous()` split | **+17%** (18→20.9 tok/s GLM) | Q5_K fp16 multireduce → two MV-matched matmuls (34→89 GB/s) |
| Pairwise topk | **4.4x** per topk | 29→3 kernels, constant kernel count for any N |
| Q4_0 packed-dot QuantizedLinear | **+96%** (26.5→52 tok/s) | 3.56x less bandwidth than fp16 cache |
| Single K cache (V = slice of K) | -130 kernels | Matches llama.cpp MLA approach |
| Selective .contiguous() split (Q5K+Q6K) | **+27%** (11→14 tok/s GLM) | Split benefits types with low compression ratio |
| Q4_0 format (vs Q4_K) | **+11-22%** | Simpler dequant fuses better |
| QK dequant cache for non-Q4 | **+55%** (20→31 tok/s) | Dequant to fp16 once, reuse |

## What Failed

| Attempt | Result | Why |
|---------|--------|-----|
| MOE_FUSED custom UOp kernels | **6x slower** | Dispatch overhead + scattered byte reads |
| Per-expert dispatch | **-18%** | 6x more kernel dispatches per MoE layer |
| Remove `.contiguous()` | **3600+ kernels** | Scheduler can't partition large graphs |
| Split KV cache (avoid cat) | **+66 kernels** | 2x realize barriers fragment fusion |
| Ushort loads (codegen level) | **-4.8%** | ALU overhead > bandwidth saving |
| Ushort loads (tensor bitcast) | **NaN/inf** | Wrong nibble access pattern |
| MV heuristic relaxation | **1 match** | Fused dequant ranges don't align |
| GROUPTOP on fused MoE | **+4%** | 768KB threadgroup memory > 32KB limit |
| BEAM on fused dequant | **No help** | Can't fix structural problem |
| FP16 softmax with `.max()` | **+32 kernels** | Max adds kernels, no speedup |
| ICB barrier removal | **0%** | Almost all kernels are data-dependent |
| Combined q_a+kv_a projection | **Same** | Split consumption → scheduler duplicates |
| Combined gate+up projection | **+1 kernel** | Breaks parallel reduce fusion |
| Pre-dequant to fp16 for experts | **Worse** | 3.56x bandwidth increase > bandwidth saving |
| Q4_0 GROUP heuristic (ds2-lite) | **+7%** (25.4→27.1) | GROUP fires on all expert kernels but ceiling ~50 GB/s; +26 kernels from splits offset gains |

## Model Specifications

### youtu-llm:2b-Q4_0
- Architecture: DeepSeek-V2/MLA, 32 blocks, dense FFN (no MoE)
- Params: 0.69 GB, Q4_0 quantized
- MLA: kv_lora=512, q_lora=1536, qk_nope=128, qk_rope=64, v_head=128, 16 heads
- 18 kernels/block (7 pre-cache + 11 post-cache)

### deepseek-v2-lite
- Architecture: DeepSeek-V2/MLA, 28 blocks (1 dense + 27 MoE)
- Params: ~2 GB, 64 experts, 6 selected/token, sigmoid gating
- MLA: kv_lora=512, q_lora=0 (direct projection), qk_nope=128, qk_rope=64, v_head=128, 16 heads

### GLM-4.7-Flash
- Architecture: DeepSeek-V2/MLA (deepseek2), 47 blocks (1 dense + 46 MoE)
- Params: 1.24 GB active (30B total, 3B active per token)
- 64 experts, 4 selected/token, 1 shared expert, sigmoid gating with 1.8x scale
- MLA: kv_lora=512, q_lora=768, qk_nope=192, qk_rope=64, v_head=256, 20 heads
- Quant breakdown: Q4_K (260 tensors), Q6_K (70 tensors), Q5_K (92 tensors)

## Action Plan (Ranked)

| # | Action | Expected | Status |
|---|--------|----------|--------|
| 1 | **Custom `mul_mat_id` CompiledRunner for MoE** | GLM 18→30+ tok/s | Prototype exists (Q4K/Q6K MoE MSL) |
| 2 | **Scheduler fusion** (norm+matmul, QK+softmax) | 586→~400 kernels | Blocked on scheduler understanding |
| 3 | ~~Metal ICB barrier removal~~ | ~~56→63 tok/s~~ | **Tried, 0% gain** (all kernels data-dependent) |
| 4 | ~~Ushort codegen~~ | ~~56→65 tok/s~~ | **Tried, -4.8%** (ALU overhead > bandwidth saving) |

## File Index

### Core Documentation
| File | Contents |
|------|----------|
| **[architecture.md](architecture.md)** | MLA formulations, MoE routing, quantization formats, per-block kernel anatomy |
| **[bottlenecks.md](bottlenecks.md)** | 3 performance gaps, per-model budgets, heuristic failures, all experiments tried |
| **[llama_cpp.md](llama_cpp.md)** | Reference implementation: mul_mat_id, ICB barriers, kernel design patterns |
| **[tools.md](tools.md)** | Commands, profiling workflows, benchmarking rules, chat templates |

### Detailed Analysis (NEW)
| File | Contents |
|------|----------|
| **[experiments.md](experiments.md)** | Detailed log of all optimization experiments with results and learnings |
| **[kernel_analysis.md](kernel_analysis.md)** | Per-kernel breakdown of GLM-4.7-Flash Q4_0, bottleneck analysis |
| **[advanced_ideas.md](advanced_ideas.md)** | E-graph, PARAM normalization, atomic scatter - next-gen optimizations |

## Key Files in Codebase

| File | Purpose |
|------|---------|
| `tinygrad/apps/llm.py` | Transformer forward, JIT wrapper, tokenizer |
| `tinygrad/apps/mla.py` | MLA attention + MoE FFN (the hot loop) |
| `tinygrad/apps/quantized.py` | QuantizedLinear, QuantizedExpertWeights |
| `tinygrad/codegen/opt/heuristic.py` | MV heuristic, GROUPTOP, kernel optimization |
| `tinygrad/engine/schedule.py` | Graph → kernel scheduling |
| `tinygrad/engine/jit.py` | JIT capture and replay |
