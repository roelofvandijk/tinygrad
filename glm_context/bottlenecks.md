# Performance Bottlenecks & Experiments

## The Three Gaps

| | llama.cpp | tinygrad | Impact |
|--|-----------|----------|--------|
| **Kernels/token** | ~50-100 (youtu), ~200 (GLM) | 586 (youtu), 1700 (GLM) | 6-12x more dispatch overhead |
| **Per-kernel overhead** | ~0 (pipelined, selective barriers) | ~34us (every kernel barrier-serialized) | 60-80% of wall time is overhead |
| **Kernel quality** | Hand-tuned SIMD+coalesced+multi-row | Decent (100+ GB/s) but not hand-tuned | 1.5-2x bandwidth gap on Q4_0 |

**Bottom line**: llama.cpp dispatches 6-12x fewer operations, each with near-zero overhead, and each individually more efficient. tinygrad's compiler generates reasonable kernels but wraps them in 6x the dispatch overhead.

---

## Gap 1: Kernel Count (6-12x)

llama.cpp has ~10-15 hand-written operations per layer. tinygrad generates 18-37 kernels/layer because:

1. **RMSNorm = 2 kernels** (reduce + elementwise). llama.cpp: 1 fused kernel. **FIXED: 1 kernel with indexing.py:236 change**
2. **Softmax = 2-3 kernels** (QK + exp+sum + div). llama.cpp: online softmax in 1. **IMPROVED: 2 kernels with indexing.py:236 change**
3. **MoE routing = 3+ kernels** (gate + topk + gather). llama.cpp: fused topk-moe (CUDA) or argsort (Metal).
4. **Expert dispatch = 4+ kernels** (2 gathers + probs + fused dequant+matmul). llama.cpp: 1 `mul_mat_id` per projection.
5. **Cache ops = 2-5 kernels** (assign + contiguous boundaries). llama.cpp: direct pointer writes.

The scheduler can't fuse across `.realize()` / `.contiguous()` boundaries. These exist because without them, the scheduler sees the entire 32-layer graph and explodes to 3600+ kernels with terrible partitioning.

### The 18-kernel structure is a local optimum (youtu)

- Removing `.contiguous()` → 3600+ kernels (scheduler can't partition large graphs)
- Adding more barriers → 652+ kernels (fragments fusion)
- Combining projections (q_a+kv_a, gate+up) → same or more kernels (split consumption duplicates, breaks parallel reduce fusion)
- The current 18/layer is the best the scheduler can do with current fusion heuristics

### indexing.py:236 change breaks through the optimum
- **Mechanism**: Remove `not (PCONTIG > 1)` guard → ranges only end when out of order
- **Result**: Reduce+broadcast patterns (RMSNorm, softmax) fuse into single kernels
- **GLM impact**: ~45 kernels/block → potentially ~35/block (5 RMSNorms × 1 kernel saved + softmax savings)
- **Blocker**: 3-matmul gate+up+down fusion creates LOCAL BUFFERIZE with 10240 floats → exceeds Metal shared_max
- **See**: `glm_context/scheduler_fusion.md` for full analysis

### 15 vs 586: the raw graph proof

```
Kernels from .realize() calls inside blocks: 567
Kernels from final lazy .schedule():          15
```

The raw graph (no realize barriers) schedules to **15 kernels for all 32 blocks**. But each block's `cache.assign(k_new).realize()` forces a scheduling split. Each block's cache is independent — there's no cross-block cache dependency, yet the per-block barriers are necessary for correctness within a block.

---

## Gap 2: Dispatch Overhead (~34us vs ~0)

Every kernel in a Metal ICB gets `.setBarrier()` (`metal.py:50`), fully serializing all kernels. Independent operations (q_a and kv_a projections, gate and up matmuls) can't overlap.

**llama.cpp's approach** (`ggml-metal-common.cpp:280-359`):
- Track read/write buffer ranges per kernel
- Only insert `memoryBarrierWithScope` when RAW/WAR/WAW conflict detected
- Look ahead N_FORWARD=8 nodes to reorder and maximize concurrency windows

**tinygrad already has the info needed**: `CompiledRunner.p` has `outs` (written buffers) and `ins` (read buffers).

**Tested Feb 8**: Replaced unconditional `setBarrier()` with conflict-based barriers. **Result: 0% speedup** (54 tok/s → 54 tok/s). Nearly all 586 kernels are data-dependent (each output feeds next input). Only ~2 kernels/layer are independent (q_a and kv_a both read x_norm) = ~32 pairs = ~1ms savings, invisible in benchmarks. Would need kernel reordering to create concurrency opportunities.

### Steady-state ICB profiling discovery

All previous kernel-level timing was misleading. **Every kernel shows ~37us per call in steady state — regardless of size.** The Metal profiler divides total ICB time evenly among kernels. It cannot measure individual kernel durations within an ICB. Only warm-up (pre-JIT) times reflect real per-kernel bandwidth. Trust wall-clock benchmarks, not ICB kernel profiles.

---

## Gap 3: Individual Kernel Quality

### Dense matvec (QuantizedLinear)
tinygrad's Q4_0 packed-dot kernels achieve **103-147 GB/s** (45-64% of theoretical 229 GB/s). llama.cpp's hand-tuned MSL kernel gets **229 GB/s** via ushort reads, SIMD reduction, multi-row amortization, stride-4 block interleaving, and the 1/256 trick.

**Ushort was tried at both tensor and codegen levels — both failed**:
- Tensor bitcast: shape mismatch, wrong nibble access patterns, NaN/inf
- Codegen pattern rewrite: confirmed ushort loads generated, but **-4.8% regression** (48.8 vs 51.3 tok/s). ALU overhead of unpacking ushort → 4 nibbles exceeds the bandwidth saving. Metal's memory controller already coalesces byte loads efficiently.

### MoE expert matvec (QuantizedExpertWeights)
This is where the gap is largest. tinygrad fuses dequant+matmul into one kernel, but:
- **MV heuristic fails**: requires `MUL(INDEX, INDEX)`, sees `MUL(dequant_chain(INDEX), INDEX)`
- **GROUPTOP fails**: output dims (6×2048 = 12288) exceed 2048 threshold
- Without GROUP: **serial reduction at 2-15 GB/s** instead of 80+ GB/s
- With Q4_0 GROUP heuristic (Feb 8): **42-54 GB/s** — improved but **~50 GB/s ceiling** from scattered byte reads in generated code. Custom MSL needed for 100+ GB/s.

---

## Per-Model Time Budgets

### youtu-llm:2b-Q4_0 (17.8ms/tok, 56 tok/s)

| Component | Time | Notes |
|-----------|------|-------|
| GPU compute (weight reads) | ~7ms | 0.69 GB @ 100 GB/s |
| ICB dispatch + barrier | ~8ms | 586 kernels × ~14us within ICB |
| ICB transition overhead | ~2ms | Between 5 ICB batches |
| Python/JIT | ~1ms | JIT replay loop |

### deepseek-v2-lite-Q4_0 (34.8ms/tok, 28.8 tok/s)

| Category | ms/tok | % | Kernels/tok |
|----------|--------|---|-------------|
| Expert gate+up (fused) | 16.9 | 45% | 27 |
| Expert down | 7.2 | 19% | 27 |
| Shared expert + weighted sum (fused) | ~6.0 | 16% | 27 |
| Attention (q_a, kv_a, q_b, attn_out) | ~3.0 | 8% | ~108 |
| Small reductions (norms, softmax, topk) | ~1.0 | 3% | ~263 |
| Elementwise | ~0.7 | 2% | ~276 |
| **Total** | **~34.8** | **100%** | **728** |

Expert kernels = **80% of token time**. Everything else is noise.

### GLM-4.7-Flash-Q4_0 (49ms/tok, 20.0 tok/s) — Latest: Feb 10, 2026

**→ See [kernel_analysis.md](kernel_analysis.md) for complete per-kernel breakdown**

Top bottlenecks by time percentage:

| Component | % | us/call | GB/s | Status |
|-----------|--:|--------:|-----:|--------|
| Shared expert gate+silu*up Q4_0 | 22% | 370 | 34 | Scheduler already fuses at kernel level |
| Expert gather gate_up | 21% | 356 | 160 | Pure data copy waste |
| Expert Q4_0 gate_up matmul | 16% | 278 | 51 | Low bandwidth, MV heuristic could help |
| Attn output Q4_0 | 9% | 151 | 93 | Well-optimized |
| Expert gather down_proj | 9% | 167 | 127 | Pure data copy waste |
| Expert Q4_0 down matmul | 8% | 153 | 47 | Low bandwidth |
| **MoE total** | **54%** | — | — | Expert gathers = 30% pure waste |

**Key finding**: Pure tinygrad DSL (no custom MSL) achieves 20 tok/s. Scheduler already fuses gate+silu*up at kernel level. Weight-level merge tested: +1.65% = noise.

---

## Heuristic Failures

### MV Heuristic Cannot Match Fused Dequant

MV (heuristic.py:72-73) requires: `mulop.src[act_i].op is Ops.INDEX`

The outermost MUL in fused dequant kernels is `MUL(scale_bitcast_chain, ADD(MUL(lo-8, x), MUL(hi-8, x)))`. Neither operand is an INDEX — one is a bitcast chain from INDEX, the other is ADD of MULs. The activation INDEXes are buried TWO levels deep inside the ADD.

**Attempts to fix**:
1. Walk `backward_slice` to find INDEX through chains — finds INDEXes but range checks fail
2. Use `.ranges` instead of `split_uop(Ops.ADD)` — helps but `ranges_subset` still fails
3. Try both orderings of idx0/idx1 — neither ordering passes all checks

**Root cause**: Fused dequant creates MULTIPLE reduce ranges (block index + element-within-block). Weight INDEX has extra ranges (sub-block indices) not in activation INDEX. `all(r in idx_b.ranges for r in idx_a.ranges)` fails in both orderings.

### GROUPTOP Threshold Blocks MoE Kernels

GROUPTOP (heuristic.py:92) requires: `prod(output_shape[i] for i in upcastable_dims) <= 2048`

Output for expert gate+up kernel: 6 experts * 1408 outputs = 8448 > 2048. For expert down: 6 × 2048 = 12288 > 2048. Both blocked.

**Raising threshold to 16384**: 12.8 tok/s — SEVERE regression. GROUPTOP applied to ALL kernels including attention that was already well-optimized with MV.

### Multireduce Prevents GROUP

Gate+up fused kernel has TWO reduction groups. GROUP(0, N) only splits the FIRST reduction's outer dim. The second reduction is unaffected. Both reductions run serially, with GROUP sync overhead paid but only one reduction benefits.

### Q4_0 GROUP Heuristic — Works But ~50 GB/s Ceiling (Feb 8, 2026)

**Approach**: Detect Q4_0 packed-dot kernels via bitwise ops (AND/SHR) in reduce chain, apply GROUP+LOCAL+UPCAST. Added to heuristic.py after MV section.

**Detection**: `reduce_product >= 256 and len(reduce_rngs) >= 2 and any(u.op in {Ops.AND, Ops.SHR} ...)`

**Key learnings**:
1. **Must try ALL global axes**, not just the first. Down kernel has global axes [2, 32] — first axis (2) too small for any LOCAL+UPCAST combo, but second (32) works. The MV-style `break` after first axis prevented this.
2. **Flexible GROUP sizes matter**: First reduce range of 44 needs GROUP(0,4) since 44 % 16 ≠ 0 and 44 % 8 ≠ 0. Try [16, 8, 4].
3. **Tighten detection**: Without `reduce_product >= 256` and `len(reduce_rngs) >= 2`, GROUP pollutes small topk kernels (r_64_16_4: 10→20us, 2x worse).
4. **`.contiguous()` after `.silu()` breaks fusion** (ALU op prevents `found_contiguous` elision in rangeify.py). But `.contiguous()` after `.reshape()` is elided.
5. **Shared expert split** requires `.contiguous()` before addition to break multireduce → single-reduce. Then GROUP fires.

**Per-kernel results** (ds2-lite, warm-up times):

| Kernel | Before | After | GROUP config |
|--------|--------|-------|-------------|
| Gate `r_528_*` | 201us, 50 GB/s | 198us, 50 GB/s | GROUP(0,16), LOCAL(0,4), UPCAST(0,4) |
| Up `r_528_*n1` | 198us, 51 GB/s | 192us, 54 GB/s | GROUP(0,16), LOCAL(0,4), UPCAST(0,4) |
| Down `r_*_44_16` | 281us, 34 GB/s | 241us, 42 GB/s | GROUP(0,4), LOCAL(1,4), UPCAST(1,4) |
| Shared `r_288_*` | 200us, 27 GB/s | 196us, 28 GB/s | GROUP(0,16), LOCAL(0,4), UPCAST(0,4) |

**Net**: 25.4 → 27.1 tok/s (+7%). Per-kernel gains offset by +26 kernels from `.contiguous()` splits. Generated code still uses scattered byte reads — fundamental limit for heuristic approach.

---

## Data Amplification: 3.56x fp16 Tax

### The insight (Feb 7, 2026)

llama.cpp reads Q4_0 blocks directly in every matvec kernel — 0.5625 bytes/element. tinygrad's `QuantizedLinear._dequant_cache` dequants to fp16 at load time, then reads 2 bytes/element every token. That's **3.56x data amplification** on every QuantizedLinear call.

### GLM bytes read per token

| Component | fp16 cache | Q4_0 direct | Savings |
|-----------|-----------|-------------|---------|
| Attention linears (×47) | 1.61 GB | 0.45 GB | **1.16 GB** |
| Shared expert (×46) | 5.78 GB | 1.62 GB | **4.16 GB** |
| Expert weights (×46) | 0.98 GB | 0.98 GB | 0 |
| Absorbed k_b/v_b (×47) | 0.43 GB | 0.43 GB | 0 (already fp16) |
| **Total** | **8.80 GB** | **3.48 GB** | **5.32 GB (60% less)** |

### Q4_0 packed-dot was the correct fix

The packed-dot approach achieves **26 GB/s** for pure tinygrad tensor ops (no MSL):
```python
lo = packed.bitwise_and(0xF).cast(fp16)
hi = packed.rshift(4).cast(fp16)
result = (scale * (lo * x_lo + hi * x_hi)).sum(...)
```
This structures the computation so inner products create `MUL(INDEX, INDEX)` patterns the optimizer CAN match.

For Q4_0 offset: `sum((nib - 8) * x) = sum(nib * x) - 8 * sum(x_block)`. The `-8` offset becomes a simple correction term.

---

## Problematic Kernel Deep-Dives

### Expert Gate+Up Fused (ds2-lite: 45% of token time)

**Name**: `r_880_32_3_64_16_64_16` — 626us each, 33 GB/s, MULTIREDUCE

Opts applied: `UPCAST(0, 3) UNROLL(3, 0) LOCAL(0, 32)`. NO GROUP. NO GROUPTOP. Each thread does the full reduction serially.

**Why it's slow**:
1. **Multireduce prevents GROUP**: Two reduction groups (gate and up) — GROUP only splits one
2. **Asymmetric unrolling**: First reduce (gate) loops `for Ridx0_1 in range(16)`, second (up) fully unrolled with half4 vectorized reads. Gate half ~2x slower.
3. **No cooperative reduction**: 32 threads each do 64×16 = 1024 iterations serially per reduction
4. **Scattered byte reads**: 32 threads span 110,592 bytes vs 32KB L1 cache

Metal source (first reduce — the slow one):
```metal
for (int Ridx0_0 = 0; Ridx0_0 < 64; Ridx0_0++) {
    int alu6 = (Ridx0_0*18);
    for (int Ridx0_1 = 0; Ridx0_1 < 16; Ridx0_1++) {
        // ONE packed byte per row per iteration (not vectorized!)
        unsigned char val7 = (*(data1+(alu8+2)));
        // scale * ((nibble & 0xF) - 8) * x_lo + ((nibble >> 4) - 8) * x_hi
        acc0[0] += scale * ...;
    }
}
```

### Expert Down Projection (ds2-lite: 19% of token time)

**Name**: `r_15_32_4_16_4_44_16` — 266us each, 39 GB/s, SINGLE REDUCE

Opts: `UPCAST(1, 4) UNROLL(1, 0) LOCAL(0, 4) LOCAL(1, 16)`. NO GROUP. This IS a single-reduce kernel — GROUP can work here.

Each thread does 44 serial iterations. With GROUP(0, 4), each would do 11 iterations + sync = ~140us. Expected saving: 3.4ms/token.

### Simplified Dequant Format Benchmarks (2048×5120 matvec)

| Format | Time | BW | Why |
|--------|------|----|-----|
| Q8 repack (scale×int8) | **146us** | **72 GB/s** | Simple MUL, GROUPTOP matches |
| Cached fp16 | 456us | 46 GB/s | 2x more data than Q8 |
| Q4 repack (scale×nib+off) | 2622us | 2 GB/s | Nibble unpack fragments kernel |
| Q4K full (8 sub-scales) | ~1400us | 14 GB/s | Complex UOp graph |

---

## Compression Ratio Determines Fusion Strategy

| Type | Dims | Fused (ms) | Split (ms) | Winner | Why |
|------|------|-----------|-----------|--------|-----|
| Q4K gate_exps | 64×1536×2048 | **1.15** | 1.32 | Fused | 3.56x compression > split gain |
| Q4K up_exps | 64×1536×2048 | **0.89** | 1.22 | Fused | Same |
| Q6K down_exps | 64×2048×1536 | 1.81 | **1.51** | Split | 2.44x compression, split dominates |
| Q5K shared_gate | 64×1536×2048 | 1.95 | **1.50** | Split | 2.91x compression, split dominates |

Crossover: `compression_ratio × fused_bandwidth > split_bandwidth`. For Q4K: 3.56 × 14 ≈ 50 > 60 → fused wins. For Q6K: 2.44 × 2 ≈ 5 << 60 → split wins.

---

## Q4_0 vs Q4_K for MoE

Q4_0 is **11-22% faster** than Q4_K despite 8x more blocks:
- **deepseek-v2-lite**: Q4_0 = 23.4 tok/s vs Q4_K = 21.0 tok/s (+11%)
- **GLM-4.7-Flash**: Q4_0 = 14.6 tok/s vs Q4_K = 12.0 tok/s (+22%)

Why: Q4_0 dequant = ~5 ops. Q4_K dequant = ~20 ops with hierarchical sub-block scale extraction. Simpler UOp graph → better fusion → better optimizer pattern matching.

**Critical**: The bartowski Q4_0 GGUF was broken (0.02 tok/s), but unsloth Q4_0 works fine. GGUF source matters.

---

## Complete Experiment Log

### youtu-llm:2b-Q4_0 (baseline: 51→56 tok/s)

| Experiment | tok/s | vs baseline | Notes |
|-----------|-------|-------------|-------|
| Current (per-block realize) | 56 | — | 586 kernels, 5 ICBs |
| Deferred cache (cat old + new) | 548 kernels | -6.5% | .cat() prevents fusions |
| Remove .contiguous() | 3624 kernels | — | Scheduler partitions cross-block graph |
| No cache at all (theoretical) | 15 kernels | — | Not achievable with autoregressive decode |
| ICB barrier removal | 54 | 0% | Nearly all kernels data-dependent |

### deepseek-v2-lite (baseline: 19.2→28.8 tok/s)

| Experiment | tok/s | vs baseline | Notes |
|-----------|-------|-------------|-------|
| Baseline (no MSL) | 19.2 | — | 840 kernels |
| MoE down .contiguous() break | 25.5 | +33% | Single-reduce kernel gets GROUP |
| Q4_0 format (vs Q4_K) | 23.4 | +22% | Simpler dequant fuses better |
| MV backward_slice walk | 19.0 | -1% | Caught multireduce, regression |
| GROUPTOP_MAX=16384 | 12.8 | -33% | Applied to ALL kernels |
| Targeted packed-dot GROUP | 18.8 | -2% | Still caught multireduce |
| Split gate+up .contiguous() | 28.8 | 0% | Scheduler re-fuses (found_contiguous) |
| Split gate+up `.silu().contiguous()` | 28.8 | 0% | ALU prevents re-fusion, but no GROUP without heuristic |
| Q4_0 GROUP heuristic (AND/SHR detect) | 27.1 | +7% | GROUP on all 4 expert kernels; 780 kernels (+26 from splits) |
| MV_ROWS_PER_THREAD=1 | 29.2 | +1% | Small win on attention |
| Per-expert dispatch | — | -18% | 6x more dispatches |
| JIT_BATCH_SIZE=728 | 28.4 | -1% | ICB transitions not bottleneck |
| Q8 repack for Linear | 20 | +4% | MV matched but ~1min load time |
| Q8 repack for Experts | OOM | — | 64 experts blows RAM |
| fp16 dequant-cache for Experts | OOM | — | 30GB+ |
| .contiguous() split (dequant then fp16) | 12 | -37% | +33MB write+read per expert |

### GLM-4.7-Flash (baseline: 12→18→20.9 tok/s)

| Experiment | tok/s | Notes |
|-----------|-------|-------|
| GLM Q4_K_M baseline | ~12 | 1550-1703 kernels |
| GLM Q4_0 (unsloth) | 14.6 | +22% from simpler dequant |
| Selective .contiguous() (Q5K+Q6K) | 14.0 | +27% from split benefit |
| Q6K MoE MSL kernel | 17.97 | +48% for expert down proj |
| **Shared expert .contiguous() split** | **20.9** | **+17%** — Q5_K multireduce → two MV matmuls (34→89 GB/s) |
| Component isolation: no shared expert | 17.5 | -13.3ms from shared expert |
| Component isolation: attention only | 38.3 | 26ms for attention alone |
| Packed-dot QuantizedLinear | 14.1 | +142 kernels wiped out bandwidth savings |
| Remove .float() casts | 10.1 | -14% — casts are load-bearing for fp32 accumulation |
| MOE_FUSED custom UOp kernels | — | 6x slower (4.25 vs 26 tok/s) |
| BEAM on fused dequant | — | No help — can't fix structural problem |
| FP16 softmax with .max() | — | +32 kernels, no speedup |
| Combined q_a+kv_a projection | — | Same — split consumption duplicates |
| Combined gate+up projection | — | +1 kernel, breaks parallel reduce fusion |
| Pre-dequant to fp16 for experts | — | 3.56x bandwidth increase > saving |

### Approaches tried for expert gate+up specifically

| Idea | Result | Why |
|------|--------|-----|
| Break fusion (split gate/up) | 28.8 tok/s = 0% | found_contiguous re-fuses, or still no GROUP |
| Extend MV to walk dequant chains | 19 tok/s regression | Extended MV fired on multireduce too |
| Raise GROUPTOP threshold | 12.8 tok/s severe regression | Applied to ALL kernels |
| Pre-dequant to int8 at load | Not tested | 1.9x data increase probably not worth it |
| Different tensor expression | Not tested | Doubles scale multiplications |
| Pre-dequant experts to fp16 | Math says worse | 80/3.56 = 22.5 GB/s effective < 33 GB/s Q4_0 |
| Teach optimizer about fused dequant | Blocked | Multireduce + ranges_subset makes this very hard |
| Q4_0 GROUP heuristic + split | +7% | GROUP fires, gate 50GB/s, down 42GB/s — ceiling ~50 GB/s from scattered byte reads |

---

## .float() Casts Are Load-Bearing

Removing `.float()` from mla.py and quantized.py: **10.11 tok/s → 11.66 regression** (same 1703 kernels).

The casts don't create separate kernels — they're fused into adjacent operations. But removing them changes accumulation from fp32 to fp16. Q5K dequant computes in fp32 internally; with `.float()`, chain is `fp32→fp16→fp32` which compiler optimizes to keep fp32 accumulation. Without: fp16 accumulation = fewer bits = slower convergence in the reduction loop.

---

## GLM Kernel Budget Breakdown (1703 kernels)

| Category | Per block | × 46 blocks | Notes |
|----------|-----------|-------------|-------|
| RMSNorm (reduce + ewise) | 10 | 460 | 5 norms × 2 kernels each |
| Q4K MSL runners | 3 | 138 | q_a, q_b, output proj |
| Absorbed matmuls | 3 | 138 | kv_a, kv_absorbed, V proj |
| Attention (QK + softmax + rope) | 5 | 230 | QK, softmax, rope prep ×2, norm |
| MoE routing | 3 | 138 | router matmul, topk ×2 |
| Expert gather | 4 | 184 | gate, up, down, probs |
| Expert compute | 2 | 92 | gate·up fused, down fused |
| Shared expert + residual | 1 | 46 | fused into one kernel |
| Weight prep / dequant | 3 | 138 | q_b reshape, Q5K dequant cache |
| **Total** | **37** | **1702** | +dense block ≈ 1703 |

### Optimization targets (kernel count reduction)

1. **Eliminate cache_v** — v = cache_k[:, :, :pos, :kv_lora_rank]. Saves ~47 kernels.
2. **Fuse RMSNorm reduce + elementwise** — 5 norms × 46 blocks = 230 kernels → 0
3. **Fuse QK + softmax** — 46 kernels saved
4. **Fuse absorbed matmul pairs** — kv_a + kv_absorbed → 1. 46 kernels saved
5. **Expert gather consolidation** — 4 → 1. 138 kernels saved

Target: 1703 → ~1200 kernels (reduce dispatch overhead by ~30%)
