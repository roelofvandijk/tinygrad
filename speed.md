## Youtu Q4_0 Speed Sprint: 26.5 → 70 tok/s (Feb 7, 2026)

**Baseline**: 26.5 tok/s, 780 kernels, 58us/kernel, 45.2ms/token
**Target**: 70 tok/s (llama.cpp), 14ms/token
**Model**: youtu-llm:2b-Q4_0 (32 layers, MLA attention, dense FFN, no MoE)
**Rule**: No custom kernels, no BEAM. Everything else in tinygrad on the table.

### 780 kernels / 32 layers = 24.4 kernels/layer

Per layer, the operations and likely kernel counts:

**Attention (~16 kernels):**
- RMSNorm: reduce + ewise = 2-3 kernels
- q_a matmul, q_a_norm (2-3k), q_b matmul = 4-5 kernels
- kv_a_mqa matmul, kv_a_norm (2-3k) = 3-4 kernels
- RoPE (q + k) = 1-2 kernels
- absorbed K einsum = 1 kernel
- QK matmul + scale = 1 kernel
- softmax (max, exp-sub, sum, div) = 2-3 kernels
- V matmul = 1 kernel
- absorbed V einsum = 1 kernel
- output matmul + residual = 1-2 kernels

**Dense FFN (~8 kernels):**
- RMSNorm: 2-3 kernels
- gate matmul = 1 kernel
- silu = 1 kernel (or fused)
- .contiguous() FORCES BREAK
- up matmul = 1 kernel
- mul (gate_silu * up) = 1 kernel
- down matmul = 1 kernel
- .float().cast() + residual = 1-2 kernels

### Experiments Plan

| # | Change | Expected impact | Why |
|---|--------|----------------|-----|
| A | Remove .contiguous() from FFN + __call__ | -2-3 kernels/layer | Allows fusion across boundaries |
| B | Remove .float() casts, keep fp16 throughout | -3-4 kernels/layer | Eliminates cast kernels |
| C | Merge attn RMSNorm + q_a projection | -1-2 kernels/layer | One fused kernel |
| D | Remove .silu().contiguous() split | -1 kernel/layer | Fuse gate_silu with up |
| E | Simplify softmax (stay in fp16) | -1-2 kernels/layer | No float32 round-trip |

If each saves even 2 kernels/layer: 780 - 64 = 716, ~8% faster. Need more radical changes.

### Results

---

## Component Isolation Experiments (Feb 7, 2026)

GLM-4.7-Flash Q4_0 — measured by disabling components with env vars:

| Experiment | tok/s | ms/tok | Kernels | Delta kernels | Delta ms |
|------------|-------|--------|---------|--------------|----------|
| **Full model (baseline)** | **14.20** | **70.4** | **1638** | — | — |
| No shared expert | 17.50 | 57.1 | 1592 | -46 | -13.3 |
| Attention only (NO_FFN=1) | 38.34 | 26.1 | 952 | -686 | -44.3 |
| Skip topk (hardcoded experts) | 13.46 | 74.3 | 1546 | -92 | +3.9 |
| Explicit k adds (vs .sum) | 14.20 | 70.4 | 1638 | 0 | 0 |
| Packed-dot QuantizedLinear | 14.10 | 70.9 | 1780 | +142 | +0.5 |

### Time Breakdown (derived)

| Component | ms | Kernels | kernels/layer | us/kernel |
|-----------|-----|---------|---------------|-----------|
| Attention | 26.1 | 952 | 20.3 | 27.4 |
| MoE experts+routing | 31.0 | 640 | 13.9 | 48.4 |
| Shared expert | 13.3 | 46 | 1.0 | 289 |
| **Total** | **70.4** | **1638** | **34.9** | **43.0** |

### Key Learnings

1. **Attention is the first bottleneck**: Even with NO FFN, attention-only is 38 tok/s (26ms).
   Target is 50+ tok/s. Must fix attention FIRST before FFN matters.
2. **952 attention kernels / 47 layers = 20.3 kernels/layer**. llama.cpp does ~15.
   At 27.4 us/kernel overhead, reducing to 15 would save (952-705) × 27.4 = 6.8ms → 19.3ms → 52 tok/s.
3. **Shared expert = 46 kernels but 13.3ms** → 289us each. These are big fp16 matmuls (2048×10240).
   Bandwidth-bound, not dispatch-bound. Q4_0 direct reads would help but adds kernels.
4. **Packed-dot for QuantizedLinear was WORSE**: +142 kernels wiped out bandwidth savings.
   The tensor DSL approach fragments into too many small kernels. Need a different strategy.
5. **Topk removal didn't help**: -92 kernels but SLOWER (48.1 us/kernel vs 43.0).
   Suggests the routing kernels are efficient and their removal changes scheduling unfavorably.
6. **Explicit k adds = identical to .sum()**: tinygrad optimizes them to the same thing.

### Next: Focus on Attention Kernel Reduction

Use youtu-llm:2b-Q4 for fast iteration (same MLA attention, 4x faster).
Target: reduce from 20 kernels/layer to ~12 kernels/layer.

---

## KEY INSIGHT: We Read 3.56x Too Much Data (Feb 7, 2026 night)

### The llama.cpp Difference

llama.cpp reads Q4_0 blocks directly in every matvec kernel — 0.5625 bytes/element.
We dequant to fp16 at load time (QuantizedLinear._dequant_cache), then read 2 bytes/element every token.
That's **3.56x data amplification** on every QuantizedLinear call.

### GLM Architecture (deepseek2, 47 blocks, dim=2048)

```
Per block, batch=1, T=1:

ATTENTION (QuantizedLinear — fp16 dequant-cache, read every token):
  attn_q_a:      2048 → 768    Q4_0: 0.88MB  fp16: 3.15MB  (×47 = 148MB)
  attn_q_b:      768 → 5120    Q4_0: 2.21MB  fp16: 7.86MB  (×47 = 369MB)
  attn_kv_a_mqa: 2048 → 576    Q4_0: 0.66MB  fp16: 2.36MB  (×47 = 111MB)
  attn_output:   5120 → 2048   Q4_0: 5.90MB  fp16: 20.97MB (×47 = 986MB)
  absorbed k_b:  already fp16   3.93MB                       (×47 = 185MB)
  absorbed v_b:  already fp16   5.24MB                       (×47 = 246MB)

SHARED EXPERT (QuantizedLinear — fp16 dequant-cache, THE BIGGEST TARGET):
  ffn_gate_shexp: 2048 → 10240  Q4_0: 11.8MB  fp16: 41.9MB (×46 = 1928MB)
  ffn_up_shexp:   2048 → 10240  Q4_0: 11.8MB  fp16: 41.9MB (×46 = 1928MB)
  ffn_down_shexp: 10240 → 2048  Q4_0: 11.8MB  fp16: 41.9MB (×46 = 1928MB)

EXPERTS (QuantizedExpertWeights — reads Q4_0 directly, already efficient):
  gate_exps: 4/64 × 64×1536×2048  Q4_0: 7.1MB per call     (×46 = 325MB)
  up_exps:   same                  Q4_0: 7.1MB              (×46 = 325MB)
  down_exps: 4/64 × 64×2048×1536  Q4_0: 7.1MB              (×46 = 325MB)
```

### Bytes Read Per Token

| Component | Current (fp16 cache) | With Q4_0 direct | Savings |
|-----------|---------------------|------------------|---------|
| Attention linears (×47) | 1.61 GB fp16 | 0.45 GB Q4_0 | **1.16 GB** |
| Shared expert (×46) | 5.78 GB fp16 | 1.62 GB Q4_0 | **4.16 GB** |
| Expert weights (×46) | 0.98 GB Q4_0 | 0.98 GB Q4_0 | 0 |
| Absorbed k_b/v_b (×47) | 0.43 GB fp16 | 0.43 GB fp16 | 0 (already fp16) |
| **Total** | **8.80 GB** | **3.48 GB** | **5.32 GB (60% less)** |

At 100 GB/s bandwidth:
- Current: 8.80 GB → **88ms → 11.4 tok/s theoretical** (we're at 15, so ~30% overhead)
- With Q4_0 direct: 3.48 GB → **34.8ms → 28.7 tok/s theoretical**
- With overhead: ~23 tok/s realistic (1.5x current)

### Why We Were Wrong Before

We assumed the dequant-cache was free because it's a one-time cost. But at batch=1,
the **per-token cost is reading the cached weights** — and fp16 is 3.56x bigger than Q4_0.
Every single token pays this 3.56x tax on every QuantizedLinear call.

The packed-dot approach (reading Q4_0 directly) saves bandwidth at the cost of some
extra ALU for inline dequant. But Apple Silicon has 100 GB/s bandwidth and ~10 TFLOPS
compute — we're massively bandwidth-limited, so trading bandwidth for compute is a huge win.

llama.cpp figured this out from day one. Their entire architecture is built around
reading quantized data directly, never materializing fp16 weight matrices.

### What We Tried That DIDN'T Help (and why)

| Attempt | Result | Why it failed |
|---------|--------|---------------|
| Packed-dot for QuantizedExpertWeights | +3.6% | Experts are only 0.98 GB — 11% of total reads |
| MV heuristic relaxation | 2 MATVEC matches | Expert MUL structure doesn't match |
| TC_OPT=1,2 | no change | TC useless for batch=1 matvec |
| Contiguous split | 2x worse | +134 kernels of dispatch overhead |
| Full expert dequant cache | OOM | 56 GB fp16 for all experts |

**Root cause**: We were optimizing the expert path (0.98 GB) while ignoring the
QuantizedLinear path (7.39 GB = 84% of reads). The shared expert alone is 5.78 GB.

### Plan: Packed-Dot for QuantizedLinear (Phase 1)

**Goal**: Replace fp16 dequant-cache with Q4_0 packed-dot for all QuantizedLinear.
Read Q4_0 blocks directly, compute matvec inline. No fp16 weight materialization.

**Implementation**:
```python
class QuantizedLinear:
  def __call__(self, x: Tensor) -> Tensor:
    # Instead of: x @ self._dequant_cache.T  (reads fp16)
    # Do: packed-dot on self.blocks directly (reads Q4_0)
    bpr = self.in_features // 32
    blocks = self.blocks.reshape(self.out_features, bpr, 18)
    scale = blocks[:, :, :2].bitcast(dtypes.float16)
    packed = blocks[:, :, 2:]
    x_fp16 = x.cast(dtypes.float16).reshape(-1, 1, bpr, 2, 16)
    x_lo, x_hi = x_fp16[..., 0, :], x_fp16[..., 1, :]
    lo = packed.bitwise_and(0xF).cast(dtypes.float16)
    hi = packed.rshift(4).cast(dtypes.float16)
    nib_dot = (lo * x_lo + hi * x_hi).sum(axis=-1)
    x_block_sum = x_fp16.reshape(-1, 1, bpr, 32).sum(axis=-1)
    return (scale.squeeze(-1) * (nib_dot - 8.0 * x_block_sum)).sum(axis=-1)
```

**Expected savings**:
- Shared expert: 5.78 GB → 1.62 GB (saves 4.16 GB)
- Attention: 1.61 GB → 0.45 GB (saves 1.16 GB)
- Total: 8.80 GB → 3.48 GB reads per token
- Memory: frees ~3 GB RAM (no fp16 dequant caches)

**Risk**: Packed-dot kernel must achieve >30 GB/s to beat fp16 matmul.
Expert packed-dot measured 22 GB/s — but QuantizedLinear has simpler shapes
(no expert gather) so optimizer should handle it better.

**Success criteria**: >20 tok/s on GLM Q4_0 (currently 15.15).

### Phase 2: Kernel Count Reduction

Even at 3.48 GB / 100 GB/s = 34.8ms compute, we need <500 kernels at <15us overhead
to hit 50 tok/s. Current: 1550 kernels × 43us = 67ms overhead alone.

After Phase 1, profile again and attack kernel count.

---

## Plan: GLM Q4_0 → 50 tok/s (Feb 7, 2026)

### Profile: GLM Q4_0 (unsloth) — 12.15 tok/s, 1550 kernels/token, 82.5ms/tok

```
  STEADY     82.5    12.15        35.1
  Kernels: 1550 per token, 6 ICBs (32+64+128+256+512+558)
  Avg overhead per kernel: 53.2 us
```

### Per-Block Kernel Map (32 kernels × ~40 blocks + overhead = 1550)

Each block has TWO schedule calls: 28-kernel MoE schedule + 4-kernel shared-expert tail.
Every block has a UNIQUE schedule hash (different weight buffers → different cache key).
After first miss, all subsequent tokens hit cache (JIT captures full 1550-kernel sequence).

**The three killer kernels** (per block, warmup times, ~41 calls/token each):

| # | Kernel | Time | BW | What | Labels |
|---|--------|------|----|------|--------|
| 65 | `r_64_32_3_64_32_64_32` | **1115us** | **13 GB/s** | Expert gate_up+silu+mul (Q4_0 fused dequant+matmul) | bitcast,rshift,bitwise_and,cat,__sub__,cast,__matmul__,silu,__mul__ |
| 59 | `r_1536_16_128_2048` | **800us** | **16 GB/s** | Shared expert gate_up+silu+mul (fused dequant) | cast,linear,silu,__mul__ |
| 47 | `r_768_16_128n1` | **720us** | **4 GB/s** | Attention KV_A projection (fused dequant) | __add__ |

Everything else per block is fast (50-160 GB/s): attention projections, attention compute, norms.

**Full per-block trace** (28-kernel schedule [6e082f08] + 4-kernel schedule [efe5d8bb]):
```
  #46 r_16_128n1        14us  -       RMSNorm pre-attention
  #47 r_768_16_128n1   720us  4GB/s   *** Attn KV_A projection (fused Q4_0 dequant)
  #48 r_16_48            9us  -       Norm
  #49 r_40_32_4_192_4  150us  53GB/s  Attn KV_B projection
  #50 r_5_18_4_16_2_48_4 53us 87GB/s  Attn Q projection
  #51 r_20_T_16_36      10us  -       QK matmul
  #52 r_5_4_T             8us -       Softmax reduce
  #53 r_5_4_Tn1           8us -       Softmax normalize
  #54 r_5_8_16_4_4_T    12us  -       Attn V matmul
  #55 r_40_32_4_128_4n1  72us 70GB/s  Attn output proj
  #56 r_2048_16_320n1   288us 73GB/s  Linear + residual
  #57 r_16_128           16us -       RMSNorm pre-MoE
  #58 r_64_16_128        22us -       MoE gate (sigmoid routing)
  #59 r_1536_16_128_2048 800us 16GB/s *** Shared expert gate_up+silu (fused Q4_0 dequant)
  #60 r_64_16_4          10us -       Topk reduce
  #61 r_4_16_4            8us -       Topk select
  #62 E_18432_32_4_3    165us 129GB/s Expert block gather (gate/up)
  #63 E_18432_32_4_3n1  166us 128GB/s Expert block gather (gate/up)
  #64 E_4                 9us -       Routing probs gather
  #65 r_64_32_3_64_32_64_32 1115us 13GB/s *** Expert dequant+matmul+silu+mul (THE bottleneck)
  #66 r_4                 7us -       Routing norm
  #67 r_4_128_8_4_4_192 251us 161GB/s Expert down matmul (good!)
  #68 r_2048_16_4_96     88us 72GB/s  MoE output combine + residual
  #69 r_16_128n1         13us -       RMSNorm
  #70 r_576_16_128n1     40us 60GB/s  Shared expert down
  #71 r_16_32            13us -       Norm
  #72 E_9_32_2n1          9us -       Routing elementwise
  #73 E_9_16_4n1          8us -       Routing elementwise
  --- 4-kernel tail schedule [efe5d8bb] ---
  #74 r_16_128n1         13us -       Norm
  #75 r_512_16_128n1     36us 66GB/s  Projection
  #76 r_16_32n1          13us -       Norm
  #77 E_4_32_4n48         8us -       Elementwise
```

### Root Cause: Fused Dequant Kills Optimizer

The Q4_0 `dequantize_q4_0` generates `bitwise_and → rshift → cat → cast → sub → mul` chains.
When tinygrad fuses these with matmul, the resulting kernel has terrible parallelism:
- MV heuristic requires `MUL(INDEX, INDEX)` but sees `MUL(dequant_chain, INDEX)` → no match
- GROUPTOP can't apply (output > 2048 threshold)
- Result: serial reduction, 13-16 GB/s instead of 80+ GB/s

### The Packed-Dot Breakthrough (bench_q4_repack.py)

The `q4_packed_dot` approach achieves **196us at 26 GB/s** for 4096×4096 — pure tinygrad, no MSL:
```python
# Instead of: unpack nibbles → full weight matrix → matmul
# Do: paired nibble × activation dot product → scale → reduce
lo = packed.bitwise_and(0xF).cast(fp16)
hi = packed.rshift(4).cast(fp16)
result = (scale * (lo * x_lo + hi * x_hi)).sum(...)
```
This structures the computation so the inner products create `MUL(INDEX, INDEX)` patterns
that the optimizer CAN match. 5.7x faster than fused dequant (1115us → 196us scale).

For Q4_0 with offset: `sum((nib - 8) * x) = sum(nib * x) - 8 * sum(x_block)`
The `-8` offset becomes a simple `block_sum * scale * (-8)` correction term.

### Phased Plan

**Phase 1: Packed-dot Q4_0 for QuantizedExpertWeights** (target ~25 tok/s)
- Replace `dequant → reshape → matmul` with packed-dot in `QuantizedExpertWeights.__call__`
- For Q4_0 (type 2) only: restructure computation to paired nibble×activation products
- Pure tinygrad tensor ops, no MSL
- Success: expert dequant+matmul kernels go from 13 GB/s to 25+ GB/s

**Phase 2: Packed-dot for QuantizedLinear** (target ~30 tok/s)
- Same packed-dot for attention projections and shared expert
- Fixes r_768_16_128n1 at 4 GB/s and r_1536_16_128_2048 at 16 GB/s
- Success: ALL dequant+matmul kernels at 25+ GB/s

**Phase 3: Kernel count reduction** (target ~40 tok/s)
- Fuse expert gathers (E_18432 × 2 + E_4 → 1)
- Fuse topk + gate into single schedule
- Fuse norms into adjacent matmuls
- Target: ~20 kernels/block × 40 = 800 kernels (from 1550)

**Phase 4: BEAM + further optimization** (target ~50 tok/s)
- BEAM on remaining hot kernels
- Dispatch overhead: start JIT_BATCH_SIZE higher, reduce ICBs
- Per-kernel overhead from 53us to ~25us
- Target: ~500 kernels × 25us = 12.5ms dispatch + 12.4ms compute ≈ 25ms → 40 tok/s
- Remaining gap: eliminate more kernels or reduce per-kernel overhead further

### Success Criteria

| Phase | tok/s | ms/tok | Kernels | Key metric |
|-------|-------|--------|---------|------------|
| Current | 12.15 | 82.5 | 1550 | Fused dequant at 13 GB/s |
| Phase 1 | ~25 | ~40 | 1550 | Expert dequant at 25+ GB/s |
| Phase 2 | ~30 | ~33 | 1550 | All dequant at 25+ GB/s |
| Phase 3 | ~40 | ~25 | ~800 | Fewer dispatches |
| Phase 4 | ~50 | ~20 | ~500 | BEAM + dispatch reduction |

### Theoretical Budget (20ms target)
- Memory bandwidth limited: 1.24 GB / 100 GB/s = **12.4ms compute**
- Dispatch budget: 20 - 12.4 = **7.6ms overhead**
- At 500 kernels × 15us = 7.5ms → barely fits
- At 200 kernels × 38us = 7.6ms → more comfortable
- Kernel count reduction is essential even with faster kernels

---

## Journal — Feb 7, 2026 (night): Why MoE Quantized Models Are Slow — Root Cause Analysis

### The Core Problem

Dense quantized models (llama, qwen) are fast. MoE quantized models (GLM, deepseek, qwen3:30b-a3b) are 4-7x slower than theoretical. Same quantization format (Q4K), same optimizer, completely different performance.

### Root Cause: Dequant-Cache vs On-The-Fly Dequant

| Path | Used By | How | Kernel Quality |
|------|---------|-----|----------------|
| **Dequant-cache** | `QuantizedLinear` (dense) | Dequant to fp16 once at load → `x @ W.T` | Standard matmul, MV matches, 70+ GB/s |
| **On-the-fly dequant** | `QuantizedExpertWeights` (MoE) | Gather selected expert Q4K blocks → fuse dequant+matmul | Terrible: 2-15 GB/s |

Dense models CAN cache because total weights fit. GLM's 64 experts at fp16 = 30GB+ → doesn't fit in 36GB RAM.

### Bench: Simplified Dequant Formats (2048×5120 matvec)

| Format | Time | BW | Kernel | Opts | Why |
|--------|------|-----|--------|------|-----|
| Q8 repack (scale×int8) | **146us** | **72 GB/s** | `r_2048_16_20_16` | GROUPTOP(1,16) | Simple MUL, clean shape |
| cached fp16 | 456us | 46 GB/s | standard | MV | 2x more data than Q8 |
| Q4 repack (scale×nib+off) | 2622us | 2 GB/s | `r_16_32_4_20_4_2_32` | UPCAST+UNROLL+LOCAL | Nibble unpack fragments kernel |
| Q4K full (8 sub-scales) | ~1400us | 14 GB/s | `r_88_32_3_8_4_2_32_8_4_2_32` | GROUPTOP(1,16) | Complex UOp graph |

**Key finding**: Q8 repack is **faster than fp16** (reads half the data). But Q8 doesn't fit for GLM (31GB).

### Why Q4 Nibble Unpack Kills Performance

Q4K dequant does: `Tensor.stack(qs.bitwise_and(0xF), qs.rshift(4), dim=2)` to unpack nibbles.
This creates a **new dimension** in the UOp graph. The kernel shape decomposes into many small factors
(e.g., `16_32_4_20_4_2_32`) instead of clean `[output, reduce]` shape.

Result: optimizer assigns only 16 workgroups instead of 2000+. Each thread does serial reduction. 2 GB/s.

Q8 avoids this because `int8` values need no unpacking — direct cast to fp16.

### GLM-4.7-Flash Context
- **30B parameter MoE model** (not 4.7B — that's just the name)
- 64 experts, 4 selected per token, 47 blocks
- Active params/token: ~1.24 GB → theoretical 80 tok/s at 100 GB/s
- Current: ~12 tok/s (15% efficiency)
- Q4K_M file: ~17GB. Q8_0: 31GB (doesn't fit in 36GB RAM)
- Must stay at ~4-bit quantization

### MV Heuristic Change (deepseek-v2-lite only)
Relaxed `MUL(INDEX, INDEX)` check to allow `MUL(INDEX, dequant_chain)` → 19.2 → 21.3 tok/s.
Does NOT help GLM because GLM's MoE kernel top-level MUL is `routing × expert_result`, not `activation × weight`.

### Paths Forward

1. ~~**Simpler quantization format**~~: **DONE** — Q4_0 is 11% faster than Q4_K (23.4 vs 21 tok/s on deepseek-v2-lite).
   One scale per 32 elements, simple `(nibble - 8) * scale` formula.
   Key: avoid `q_to_uint8` expansion, use direct `bitwise_and + rshift + cat`.

2. **Avoid nibble expansion**: Process packed bytes against activation pairs (llama.cpp style).
   `lo_nib * x[j] + hi_nib * x[j+128]` per byte → no 2x tensor expansion.
   Needs new dequant formulation in `QuantizedExpertWeights.__call__`.

3. **Fix optimizer for fused dequant**: Teach GROUPTOP/MV to handle kernels with extra sub-block
   dimensions. Hard — requires deep changes to heuristic pattern matching.

4. **Hybrid approach**: Repack Q4K → simplified per-block (scale, offset, packed_nibbles) at load time.
   Loses sub-block precision (~0.5% quality loss). Gets simpler dequant.

### Key Insight: Kernel Count is the Secondary Problem

840 kernels/tok is bad, but the PRIMARY problem is individual kernel efficiency (2-15 GB/s vs 100 GB/s).
With 840 kernels at 100 GB/s, theoretical: 840 × 20us = 16.8ms → 60 tok/s. That's plenty.
Reducing to tens of kernels only matters if we ALSO fix kernel efficiency.

---

## Journal — Feb 7, 2026 (late): Q4_0 Is Actually Faster Than Q4_K!

### Discovery
Testing Q4_0 quantization revealed it's **11% faster** than Q4_K for deepseek-v2-lite MoE:
- **deepseek-v2-lite Q4_0**: 23.4 tok/s (43ms/tok)
- **deepseek-v2-lite Q4_K_M**: 21.0 tok/s (47ms/tok)

This is surprising given Q4_0 has **8x more blocks** (32 vs 256 elements per block).

### Why Q4_0 Is Faster

**Dequant complexity comparison:**

Q4_0 (simple):
```python
d = blocks[:,:2].bitcast(float16).cast(float32)  # 1 scale
lo = blocks[:,2:].bitwise_and(0xF)
hi = blocks[:,2:].rshift(4)
return (cat(lo, hi, dim=-1).cast(float32) - 8) * d
```

Q4_K (hierarchical):
```python
d, dmin = blocks[:,[0,2]].bitcast(float16).cast(float32)  # 2 base scales
sc, mn = <complex 8-sub-block scale extraction>  # 8 sub-scales each
q = stack(bitwise_and(0xF), rshift(4), dim=2).reshape(8,32)  # nibble unpack
return d * sc * q - dmin * mn  # hierarchical formula
```

**Key differences:**
1. **Simpler UOp graph**: Q4_0 has ~5 ops, Q4_K has ~20 ops with complex bit manipulation
2. **Better fusion**: Simpler dequant chain fuses better with matmul
3. **More uniform structure**: Optimizer handles Q4_0's regular pattern better than Q4_K's hierarchical structure

Even with 8x more blocks, the **per-block simplicity** creates a net win.

### Critical Fix: Replace `q_to_uint8`

The original Q4_0 dequant used `q_to_uint8(blocks, 4)` which does:
```python
expand + idiv + bitwise_and + transpose + flatten
```

This created massive UOp bloat (8x blocks × expand/idiv overhead = JIT failure).

New version uses direct bit ops:
```python
bitwise_and(0xF) + rshift(4) + cat
```

Much simpler → scheduler handles it efficiently.

### GLM-4.7-Flash Q4_0 Results

Confirmed Q4_0 is faster than Q4_K for GLM too:
- **GLM Q4_K_M**: ~12 tok/s
- **GLM Q4_0 (unsloth)**: **14.6 tok/s** (+22%)

**IMPORTANT**: The bartowski Q4_0 GGUF was broken (0.02 tok/s), but the unsloth Q4_0 works fine.
Different GGUF producers can have different tensor layouts/metadata — always use unsloth for GLM.

### Implications

1. **Q4_0 > Q4_K for MoE models**: Simpler dequant trumps larger blocks
2. **Quantization complexity matters**: Hierarchical structures hurt fusion
3. **GGUF source matters**: bartowski vs unsloth can be dramatically different

### Tokenizer Fix Required

DeepSeek Q4_0 GGUF had bad unicode ranges in tokenizer regex:
- `Ὗ-ώ` (U+1F5F to U+03CE) — backwards range
- `ῐ-ΐ` (U+1FD0 to U+0390) — backwards range

Fixed by replacing ranges with individual characters in `_DEEPSEEK_LETTERS`.

### Files Modified
- `tinygrad/nn/state.py`: Simplified `dequantize_q4_0` (no q_to_uint8), added to GGML_QUANT_INFO
- `tinygrad/apps/llm.py`: Fixed tokenizer unicode ranges, added Q4_0/Q6_K model URLs

---

## Journal — Feb 7, 2026: Remove Hand-Coded MSL Kernels — Pure Tinygrad Baseline

### Result
- **deepseek-v2-lite: 19.2 tok/s** (52ms/tok), 840 kernels/tok, 5 ICBs
- Down from 25 tok/s with MSL kernels — 23% regression from losing hand-coded Metal
- Fast load time (no Q8 repack, no heavy precomputation)
- GLM: not yet tested at this baseline

### What We Did
Removed all hand-coded Metal kernels (metal_q4k.py, metal_q6k.py). Pure tinygrad tensor ops only.

**QuantizedLinear**: Dequant-cache — dequant to fp16 at first use, cache, MV-matched matmul.
**QuantizedExpertWeights**: On-the-fly — gather selected expert Q4K blocks, fuse dequant with matmul.
No precomputation for experts (RAM can't handle it for GLM's 17GB).

### Approaches Tried and Rejected

| Approach | Result | Why |
|----------|--------|-----|
| Q8 repack (scale*int8) for Linear | 20 tok/s | MV matched but load time ~1min (full dequant+requantize) |
| Q8 repack for Experts | OOM | Dequanting all 64 experts to fp32 intermediate blows RAM |
| Q8 chunked per-expert | Too slow | 64 sequential dequant+requantize calls |
| fp16 dequant-cache for Experts | OOM | 64 experts x O x I x 2 bytes = 30GB+ |
| .contiguous() split (dequant then fp16 matmul) | 12 tok/s (2x worse!) | Extra 33MB write+read per expert call |
| Fused dequant+matmul (no .contiguous()) | **19.2 tok/s** | Best option without precomputation |

### Profile (deepseek-v2-lite, 19.2 tok/s)

Top warm-up kernels:
| Kernel | Calls/tok | Avg us | BW | Description |
|--------|-----------|--------|-----|-------------|
| `r_88_32_3_8_4_2_32_8_4_2_32` | 3 | 1373 | 15 GB/s | Fused Q4K MoE gate-up (THE bottleneck) |
| `r_16_32_6_4_352_4` | 3 | 1265 | 44 GB/s | MoE down proj (dequant-cache fp16) |
| `r_22_32_4_2048_512_4` | 3 | 665 | 36 GB/s | Large reduce |
| `r_2048_16_6_176` | 3 | 150 | 80 GB/s | Shared expert (already fast) |

**Key insight**: `r_88_32_3_8_4_2_32_8_4_2_32` runs at 15 GB/s because MV heuristic doesn't match
fused dequant pattern. Teaching MV to handle `MUL(dequant_chain(INDEX), INDEX)` would fix this.

### Next Step: MV Heuristic Dequant Chain Support
The #1 kernel at 15 GB/s is fixable. MV heuristic at `heuristic.py:67` requires `MUL(INDEX, INDEX)`.
Fused dequant creates `MUL(dequant_chain(INDEX), INDEX)`. Fix: (1) walk backward through chains to
find INDEX, (2) relax range subset check for weight's extra sub-block ranges.

---

## Journal — Feb 6, 2026 (night 2): Selective .contiguous() Split — 11→14 tok/s (27%)

### Result
- **GLM-4.7-Flash: ~11 → 14.0 tok/s (27% speedup), ~90ms → 71ms/tok**
- No MSL kernels used (Q6K_MOE_MSL=0)
- Applied selectively to Q6K (type 14) and Q5K (type 13) expert weights only

### What We Did
Added `.contiguous()` after `_dequant_fn()` in `QuantizedExpertWeights.__call__` for Q6K/Q5K only.
This breaks the fused dequant+matmul into two well-parallelized kernels: (1) dequant, (2) GEMV.

### Method: Isolated JIT Steady-State Benchmarking

**Key principle**: Measure in JIT steady state, not raw wall time. Raw wall time is dominated by
scheduling overhead (~1.3ms wall for a 78us kernel). JIT steady state amortizes this.

**Test harness** (`test_split_all.py`): For each quant type, create random expert weights matching
GLM dimensions, wrap in `@TinyJit`, warm up 5 iterations (JIT capture), then measure 30 iterations
of JIT replay with Metal synchronize barriers.

### Results Per Quant Type (JIT steady-state, 6 experts selected from 64)

| Type | Dims | Fused (ms) | Split (ms) | Delta | Winner |
|------|------|-----------|-----------|-------|--------|
| Q4K gate_exps | 64×1536×2048 | **1.15** | 1.32 | +15% worse | Fused |
| Q4K up_exps | 64×1536×2048 | **0.89** | 1.22 | +37% worse | Fused |
| Q6K down_exps | 64×2048×1536 | 1.81 | **1.51** | -17% better | Split |
| Q5K shared_gate | 64×1536×2048 | 1.95 | **1.50** | -23% better | Split |

### Why Q4K Prefers Fusion, Q6K/Q5K Prefer Split

**Compression ratio determines the trade-off:**
- Q4K: 144 bytes/256 elements → 3.56× compression. Split pays 3.56× memory amplification to
  materialize fp16 weights. The fused kernel, despite poor parallelism, wins because it reads
  3.56× less data.
- Q6K: 210 bytes/256 elements → 2.44× compression. Lower amplification penalty. Split wins because
  the GEMV kernel achieves ~60 GB/s vs ~8 GB/s for the fused dequant+matmul kernel.
- Q5K: 176 bytes/256 elements → 2.91× compression. Middle ground, but split still wins due to
  very poor fused kernel parallelism (complex dequant arithmetic serializes the reduction).

**Insight**: The crossover point is roughly where `compression_ratio × fused_bandwidth > split_bandwidth`.
For Q4K: 3.56 × 14 GB/s ≈ 50 GB/s > 60 GB/s split bandwidth — marginal, but fused wins.
For Q6K: 2.44 × 2 GB/s ≈ 5 GB/s << 60 GB/s split bandwidth — split wins easily.

### Full Model Benchmark

Selective split (Q5K + Q6K only) applied to `quantized.py`:
```
Steady-state: 14.0 tok/s (71 ms/tok), down from ~90 ms/tok without split
Savings: ~21 ms/tok from Q6K (8.4ms) + Q5K (12.6ms) selective split
```

### Gap Analysis

| Component | Est. ms/tok | Notes |
|-----------|-------------|-------|
| Q4K expert gate/up (fused) | ~28ms | Kept fused (split is worse) |
| Q6K expert down (split) | ~42ms | Improved from ~50ms |
| Q5K shared expert (split) | ~42ms | Improved from ~55ms |
| Q4K attention linears | ~5ms | Q4K MSL runners, already fast |
| Attention + other | ~15ms | RMSNorm, softmax, routing, etc. |
| **Total** | **~71ms** | **14.0 tok/s** |
| **Target** | **20ms** | **50 tok/s** |

The remaining 3.5× gap cannot be closed by split alone. Even the split path only achieves ~60 GB/s
effective bandwidth (vs 81 GB/s for clean GEMV) due to the memory amplification of materializing
decompressed weights. The fundamental path forward requires either:
1. Teaching tinygrad's optimizer to generate good fused dequant+matmul kernels (MV heuristic fix)
2. MSL kernels that read compressed data inline (like Q6K_MOE_MSL, but for all types)
3. Reducing dispatch overhead (kernel count, ICB barriers)

### Files Changed
- `tinygrad/apps/quantized.py`: Added `if self.ggml_type in (13, 14): w_flat = w_flat.contiguous()`
  after `_dequant_fn()` in `QuantizedExpertWeights.__call__` fallback path (line ~288)

### Test Scripts Created
- `test_split_all.py` — JIT steady-state fused vs split for all quant types
- `test_split_jit.py` — JIT steady-state Q6K fused vs split (initial validation)
- `test_split_gpu.py` — DEBUG=2 single-shot comparison (for kernel inspection)

---

## Journal — Feb 6, 2026 (night): Q6K MoE MSL Kernel — 12→18 tok/s (48%)

### Result
- **GLM-4.7-Flash: 12.16 → 17.97 tok/s (48% speedup), 82ms → 55.6ms/tok**
- deepseek-v2-lite: ~25.8 tok/s (no regression, Q4K experts unaffected)

### What We Did
Wrote a Q6K MoE MSL Metal kernel (`metal_q6k.py`) using llama.cpp's proven coalesced access pattern.
This directly replaces the fused dequant+matmul path for **Q6K expert down projections** — the #1
bottleneck at 28.4% of GPU time.

**Key correction**: The expert down projections in GLM are **Q6K (type 14)**, not Q5K as previously
documented. The Q5K tensors are only shared expert gate/up (tiny, 199MB total). The quant breakdown:
- Q4_K (type 12): 260 tensors — attention projections, expert gate/up
- Q6_K (type 14): 70 tensors — expert down projections, output head
- Q5_K (type 13): 92 tensors — shared expert gate/up only (small)

### Kernel Details
- llama.cpp Q6K GEMV pattern: `tid=tiisg/2`, `ix=tiisg%2` (interleaved block processing)
- NSG=2 simdgroups × 32 threads = 64 threads, NR0=2 rows per simdgroup = 4 rows per TG
- Expert indexing: `w + expert_id * BLOCKS_PER_EXPERT + row * nb`
- Q6K dequant inline: `(ql & 0xF) | ((qh & mask) << shift)` with 4 kmasks for 4 sub-groups
- `CompiledRunner` subclass for JIT ICB batching

### Files Changed
- `tinygrad/nn/metal_q6k.py`: Added `_make_q6k_moe_src`, `Q6KMoERunner`, `q6k_moe_linear_msl`
- `tinygrad/apps/quantized.py`: Added Q6K MoE MSL path for `ggml_type == 14` + `expert_first_in_memory`
  - Handles SHRINK UOp (sliced blocks): `.cast(uint8).contiguous().realize()` before buffer extraction
  - `_moe_blocks_tensor` keeps contiguous tensor alive to prevent buffer GC

---

## Journal — Feb 6, 2026 (late): .float() Casts Are Load-Bearing + Per-Block Kernel Map

### GLM Baseline (profile_model.py)
- **11.66 tok/s**, 1703 kernels/token, 85.8 ms/tok, 6 ICBs (32+64+128+256+512+535)
- 50.4 us avg overhead per kernel
- After JIT capture (token 4+), no schedule() calls — all 1703 kernels replayed as one batch

### Experiment: Remove .float() Casts
**Result: REGRESSION — 10.11 tok/s (down from 11.66). Same 1703 kernels. Reverted.**

Removed `.float()` from `mla.py` (lines 207, 208, 213) and `quantized.py` (lines 282-283).
The casts don't create separate kernels — they're fused into adjacent operations. But removing
them changes accumulation from fp32 to fp16, making fused Q5K dequant+matmul kernels slower:
- MoE down: 4201us → 5977us (+42%)
- MoE gate·up: 957us → 1149us (+20%)

**Why**: Q5K dequant computes in fp32 internally, outputs fp16. With `.float()`, the chain is
`fp32→fp16→fp32` which the compiler optimizes away (keeping fp32 accumulation). Without `.float()`,
the matmul runs in fp16 accumulation — fewer bits = less precision = slower convergence in
the reduction loop. The `.float()` casts are load-bearing for performance.

### Per-MoE-Block Kernel Map (37 dispatches, from DEBUG=2 trace)

Each of the 46 MoE blocks dispatches exactly 37 kernels. Dense block 0 adds ~46 more.
46 × 37 + ~46 = ~1748, minus shared first/last kernels ≈ **1703 total**.

**Attention phase (19 kernels):**

| # | Kernel | Operation |
|---|--------|-----------|
| 1 | `r_16_48` | q_a RMSNorm reduce |
| 2 | `E_23040_32_3` | q_b weight reshape |
| 3 | `E_6_32_4n47` | q_a norm elementwise (cast + mul) |
| 4 | `q4k_linear_msl_5120x768` | Q4K q_b projection |
| 5 | `r_16_128n1` | kv_a RMSNorm reduce |
| 6 | `r_576_16_128n1` | kv_a absorbed matmul |
| 7 | `r_16_32` | RoPE/norm reduce |
| 8 | `E_9_32_2n1` | RoPE prep |
| 9 | `E_9_16_4n1` | RoPE prep |
| 10 | `r_16_128n1` | kv absorbed RMSNorm reduce |
| 11 | `r_512_16_128n1` | kv absorbed matmul |
| 12 | `r_16_32n1` | norm reduce |
| 13 | `E_4_32_4n48` | norm elementwise |
| 14 | `r_5_18_4_16_2_48_4` | QK matmul |
| 15 | `r_5_4_28` | softmax reduce |
| 16 | `r_20_28_16_36` | attention V matmul |
| 17 | `q4k_linear_msl_2048x5120` | Q4K output projection |
| 18-19 | `E_768...` × 2 | Q5K expert weight dequant (gate, up) — one-time cache |

**MoE phase (18 kernels):**

| # | Kernel | Operation | Time |
|---|--------|-----------|------|
| 20 | `E_3072...` | Q5K expert weight dequant (down) — one-time cache | — |
| 21 | `r_16_128n2` | FFN RMSNorm reduce | ~12us |
| 22 | `E_9216_32_3` | Router weight prep | ~12us |
| 23 | `E_16_32_4n97` | FFN norm elementwise | ~11us |
| 24 | `r_64_16_128` | Router matmul + sigmoid | ~16us |
| 25 | `r_1536_16_128_2048` | Shared expert fused (gate·silu·up·down) | **406us** |
| 26 | `r_64_16_4` | Pairwise topk comparison | ~11us |
| 27 | `r_4_16_4` | Pairwise topk selection | ~9us |
| 28 | `E_18432_32_4_3` | Expert gather (gate weights) | ~25us |
| 29 | `E_18432_32_4_3n1` | Expert gather (up weights) | ~25us |
| 30 | `E_26880_32_4_3` | Expert gather (down weights) | ~35us |
| 31 | `E_4` | Routing probs gather | ~14us |
| 32 | `r_64_32_3_8_4_2_32_8_4_2_32` | Expert gate·silu·up fused (Q5K) | **1150us** |
| 33 | `r_4` | Routing normalization | ~9us |
| 34 | `r_32_4_16_4_6_4_2_32` | **Expert down fused (Q5K) — THE BOTTLENECK** | **5977us** |
| 35 | `r_2048_16_4_96` | Shared expert + residual + weighted sum | ~92us |
| 36 | `r_16_128n1` | Next block's attn RMSNorm reduce | ~12us |
| 37 | `E_16_32_4n96` | Next block's attn norm elementwise | ~12us |

Plus `q4k_linear_msl_768x2048` (next block's q_a) — custom runner, overlaps with next block.

### Kernel Budget Analysis

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
| Weight prep / dequant | 3 | 138 | q_b reshape, Q5K dequant cache (one-time) |
| **Total** | **37** | **1702** | +dense block ≈ 1703 |

### Optimization Targets (kernel count reduction)

1. **Eliminate cache_v** — v = cache_k[:, :, :pos, :kv_lora_rank]. Saves 1 realize per block = ~47 kernels
2. **Fuse RMSNorm reduce + elementwise** — Currently 2 kernels per norm. 5 norms × 46 blocks = 230 kernels → 0
3. **Fuse QK + softmax** — 2 separate kernels → 1. Saves 46 kernels
4. **Fuse absorbed matmul pairs** — kv_a + kv_absorbed could be 1 kernel. Saves 46 kernels
5. **Expert gather consolidation** — 4 separate gathers → 1. Saves 138 kernels

Target: 1703 → ~1200 kernels (reduce dispatch overhead by ~30%)

---



### Critical Discovery: Steady-State ICB Profiling

All previous analysis was based on `VIZ=-1` aggregate totals (warm-up + steady). When filtering to
**steady-state only** (tokens 5-10, after JIT), the picture changes completely:

**Every kernel shows ~37us per call in steady state — regardless of size.**

| Kernel | Steady avg | Work | Notes |
|--------|-----------|------|-------|
| `E_6` (6 elements!) | 37.9us | trivial | Same as below |
| `r_16_4` (16×4 reduce) | 37.9us | trivial | Same timing |
| `r_88_32_3_8_4_2_32_8_4_2_32` (gate+up Q4K) | 37.9us | huge | Same timing! |
| `r_16_32_6_4_352_4` (down fp16) | 37.9us | huge | Same timing! |
| `q4k_linear` (MSL kernel) | 36.0us | medium | Same timing! |

**Conclusion**: The Metal profiler divides total ICB time evenly among kernels. It cannot measure
individual kernel durations within an ICB. The ~37us is `total_ICB_time / num_kernels`.

### What This Means

- **948 kernels/tok × 37us = 35.1ms/tok** — this accounts for nearly ALL of the 37.5ms wall time
- Individual kernel optimization (MV heuristic, GROUPTOP, custom kernels) was targeting the WRONG thing
- The bottleneck is **per-kernel overhead** in the ICB, likely from `setBarrier()` on every kernel
- Every approach that INCREASED kernel count (PER_EXPERT, DEQUANT_BREAK, Q4K_MOE_MSL) regressed
  because it added more 37us-overhead slots, not because the kernels themselves were slower

### Warm-Up vs Steady-State Comparison

| Kernel | Warm-up (pre-JIT) | Steady (in ICB) | Speedup |
|--------|-------------------|-----------------|---------|
| Gate+up Q4K | 1300us @ 15 GB/s | 37.9us (uniform) | 34× |
| Down fp16 | 1300us @ 290 GB/s | 37.9us (uniform) | 34× |
| q4k_linear | 32-228us | 36.0us | ~same |

The warm-up numbers (15 GB/s) reflected REAL per-kernel bandwidth. But once inside ICBs, the
per-kernel profiler resolution is lost. The dramatic warm-up → steady improvement comes from
ICB batching, not from the GPU suddenly reading memory 34× faster.

### Steady-State Breakdown

| Category | ms/tok | Pct | Details |
|----------|--------|-----|---------|
| q4k_linear (attn) | 4.92 | 14% | 137 calls/tok |
| RMSNorm + elementwise | 5.46 | 15% | ~72 + 72 calls/tok |
| Other small reductions | ~8 | 23% | ~200 calls/tok |
| MoE gate+up | 0.91 | 2.6% | 24 calls/tok |
| MoE down | 0.91 | 2.6% | 24 calls/tok |
| Attention ops | ~5 | 14% | ~130 calls/tok |
| Shared expert + routing | ~2 | 6% | ~50 calls/tok |
| ICB transition overhead | ~7 | 20% | Between ICB batches |
| **Total GPU** | **35.3** | | **948 kernels/tok** |
| Python/JIT overhead | ~2.2 | | |
| **Wall time** | **37.5** | | **26.6 tok/s** |

### Path to 50 tok/s (20ms/tok) — REVISED

The old plan (fix individual Q4K kernel at 15 GB/s) was wrong. The new plan:

**Strategy: Reduce kernel count from 948 to ~500 and/or reduce per-kernel overhead from 37us to ~20us.**

1. **Remove ICB barriers for independent kernels** (`metal.py:50`)
   - Current: `setBarrier()` on every kernel → GPU serializes all 948 kernels
   - Proposed: only barrier between kernels that share buffers with write access
   - Potential savings: if GPU can pipeline independent kernels, per-kernel overhead drops
   - Risk: Metal ICB barrier semantics may be required for correctness

2. **Reduce kernel count via fusion**
   - RMSNorm + elementwise: ~144 calls/tok → could fuse norm+first_linear
   - Many tiny reductions (`r_16_4`, `r_16_32`) = unnecessary kernel boundaries
   - Target: 948 → 500 kernels/tok → save ~16ms

3. **Increase ICB batch size** (`JIT_BATCH_SIZE`)
   - Currently starts at 32, doubles: 32+64+128+256+468 = 5 ICBs
   - Starting at 128: 128+256+564 = 3 ICBs → less ICB transition overhead

### Per-Token Kernel Count Breakdown (est.)

| Component | Kernels/tok | Notes |
|-----------|-------------|-------|
| Q4K attention linears | ~137 | q4k_linear MSL |
| RMSNorm (r_16_128) | ~72 | 27 blocks × ~3 norms |
| Elementwise (E_16_32) | ~72 | Cast, residual add |
| Other reductions | ~200 | r_16_32, r_16_4, etc. |
| Attention ops | ~130 | QK, softmax, V, proj |
| MoE gate+up | ~24 | Fused Q4K dequant |
| MoE down | ~24 | fp16 matmul |
| MoE routing + gather | ~50 | Topk, expert select |
| Expert add/residual | ~50 | Sum, shared expert |
| Misc | ~89 | Initialization, etc. |
| **Total** | **~948** | |

---

## Journal — Feb 6, 2026 (DeepSeek 50 tok/s push, no BEAM)

### Run Constraints
- `BEAM=0` only.
- Never run more than one benchmark at once.
- Use `timeout 60` for DeepSeek runs.
- Pipe benchmark output to log files for later comparison.

### Current Status
- Best recent decode tail from exact command:
  - `50.50 ms` (`19.80 tok/s`)
  - `48.23 ms` (`20.73 tok/s`)
  - Log: `deepseek_beam0_bench6_after.log`
- Earlier run in this session reached:
  - `44.24 ms` (`22.60 tok/s`)
  - `45.55 ms` (`21.95 tok/s`)
- We are still far from the `50 tok/s` target.

### Hot Kernels (from `DEBUG=3`, `BEAM=0`)
- `r_88_32_3_8_4_2_32_8_4_2_32` (fused Q5K gate·silu·up matmul path): ~13-43 GB/s depending on token.
- `r_16_32_6_4_352_4` (matmul path): highly variable, ~20-86 GB/s.
- `r_2048_16_6_176` (shared expert linear·silu·mul): often ~3-13 GB/s and already using `GROUPTOP(1,16)`.

### Best Ideas (ranked)
1. **Custom Q5K GEMV in tinygrad UOp DSL** with explicit threadgroup reduction and coalesced dequant reads.
2. **Targeted expert pre-dequant** (only selected experts each token), then fast fp16 matmul.
3. **Restructure/fission of fused Q5K MoE kernels** when fusion forces serialized dequant+matmul loops.
4. **Dispatch cleanup only after Q5K MoE wins** (dispatch overhead is secondary to Q5K kernel inefficiency right now).

### Probe Status
- Heuristic experiments in `tinygrad/codegen/opt/heuristic.py` (`MV_FUSED`, `GROUPTOP_FUSED_LIMIT`) are **not yet validated** as a meaningful speedup.
- `MOE_FUSED=1` did not complete useful benchmark output within 60s timeout.
- Raising global `GROUPTOP_LIMIT` aggressively also failed to complete useful output within 60s.
- `MV=0` probe looked worse on first token, so not a direction to pursue.

# GLM-4.7-Flash → 50 tok/s Plan

## Target
- **GLM-4.7-Flash**: 1.24 GB params, 47 blocks, 64 MoE experts (4 active), 1 shared expert
- Theoretical at 100 GB/s: **80 tok/s** (12.4ms/tok)
- Current: **~12.2 tok/s** (82ms/tok) — 6.6x gap (was 9.3 before pairwise topk + MoE fusion break)
- Target: **50 tok/s** (20ms/tok) — 7.6ms overhead budget

## Completed Wins
- **CompiledRunner** for Q4K/Q6K: 31→55.5 tok/s on youtu (1.8x)
- **Pairwise topk**: 29→3 kernels per topk (~800 fewer kernels for GLM)
- **MoE fusion break**: 18→25.5 tok/s on deepseek-v2-lite (41%)

## GLM Kernel Profile — Feb 5 2026 (12.2 tok/s baseline)

**Total GPU time**: ~847ms across benchmark run. Steady-state: **12.21 tok/s** (81.89ms/tok).

### Top 5 Kernels (72% of GPU time)

| # | Kernel | Total | Calls | Per-call | % GPU | BW | Description |
|---|--------|-------|-------|----------|-------|-----|-------------|
| 1 | `r_9_32_4_16_4_6_4_2_32` | 240ms | 88 | 2.73ms | 28.4% | **2 GB/s** | MoE down proj (Q5K, 92.9MB weights) |
| 2 | `r_64_32_3_8_4_2_32_8_4_2_32` | 128ms | 184 | 0.70ms | 15.1% | 14 GB/s | MoE gate·silu·up fused (Q5K) |
| 3 | `r_1536_16_128_2048` | 56.7ms | 184 | 0.31ms | 6.7% | 29 GB/s | Shared expert linear·silu·mul (fp16) |
| 4 | `r_9_32_4_16_4_6_4_2_32n1` | 44.6ms | 96 | 0.46ms | 5.3% | ~3 GB/s | MoE gate/up proj (Q5K, 63.7MB weights) |
| 5 | `q4k_linear` | 34.4ms | 572 | 0.06ms | 4.1% | ~80 GB/s | Q4K attention linears |

### Kernel #1 Deep Analysis: `r_9_32_4_16_4_6_4_2_32` (28.4% — THE bottleneck)

**What it does**: MoE down projection — dot product of activated intermediate (fp32) against Q5K expert weights. Computes `output[9, 4, 2048] = dequant_Q5K(weights[9, 4, 1536]) @ act[9, 4, 1536]`.

**Structure**:
- Grid: gidx0=32, gidx1=9 → 288 workgroups
- Threadgroup: lidx0=4, lidx1=16 → 64 threads
- 4 accumulators via UPCAST (each thread writes 4 output elements)
- Loop: Ridx0_0[6] × Ridx0_1_0[4] × Ridx0_1_1_0[2] × 32(unrolled) = 1536 inner iterations
- Opts: `UPCAST(1,4), UNROLL(3,0), LOCAL(0,4), LOCAL(1,16)` — **NO GROUPTOP**

**Why 2 GB/s (should be ~80-100 GB/s)**:
1. **Too few workgroups**: 288 total. Apple Silicon needs ~2000+ for bandwidth saturation.
2. **Fully serial reduction**: Each thread loops 1536 times alone. No threadgroup cooperation. No SIMD reduction. Q4K kernel uses 32×8=256 threads with SIMD reduction for comparison.
3. **Scattered byte reads**: Adjacent threads read from locations separated by `lidx1*5040` bytes (5KB apart). Each Q5K element requires ~3 scattered byte reads (nibbles + high bits + scales at different offsets). Not coalesced.
4. **Massive weight stride**: `gidx1*10,321,920` bytes between expert groups = 10MB jumps in a 92.9MB buffer.

**Approaches (prefer tinygrad DSL)**:

**A. Add GROUPTOP to parallelize reduction** — Multiple threads cooperate on each dot product via threadgroup memory. Currently 1 thread does all 1536 multiplies. With GROUPTOP=32, 32 threads split the work (48 iterations each) then reduce via threadgroup memory. Requires: figuring out why the optimizer didn't choose GROUPTOP and forcing it.

**B. Pre-dequant selected experts (targeted QK_DEQUANT_CACHE)** — Currently `QK_DEQUANT_CACHE` dequants ALL non-Q4K weights once. For MoE experts, we only use 4 of 64, so dequanting all 64 is wasteful (64 × 6.3MB = 403MB). Instead: dequant only the 4 selected experts per token into fp16, then run a standard fp16 matmul with good parallelism. Trade-off: 4 × 6.3MB = 25MB extra memory + dequant overhead, but matmul runs at 80+ GB/s.

**C. Restructure expert computation in mla.py** — Instead of stacking all expert outputs into one `[9, 4, 2048]` tensor (forcing one fused kernel), compute each expert's down proj separately with `.contiguous()` between them. Each gets its own kernel with potentially better parallelism for its size.

**D. Custom Q5K GEMV kernel via tinygrad UOp DSL** — Write a proper GEMV with coalesced access pattern (interleaved thread→weight mapping like Q4K), SIMD reduction within threadgroup, and enough workgroups. Model on existing `q4k_linear_uop` but for Q5K format. Expert indexing built into the kernel (mul_mat_id style).

### Kernel #2: `r_64_32_3_8_4_2_32_8_4_2_32` (15.1%)

MoE gate·silu·up — fused Q5K dequant + 2 matmuls (gate, up) + silu + elementwise mul. 184 calls ≈ 4/block.
14 GB/s is better than #1 but still far from theoretical. Grid is 64×32=2048 workgroups (vs 288 for #1), explaining the 7x better bandwidth. Same Q5K dequant overhead applies.

**Approach**: If kernel #1 is fixed with a custom Q5K GEMV, same fix applies here. Alternatively, break gate and up into separate kernels if tinygrad's fusion is suboptimal.

### Kernel #3: `r_1536_16_128_2048` (6.7%)

Shared expert linear·silu·mul (fp16 weights, not quantized). 29 GB/s with 500 GB/s LDS suggests good LDS utilization but bandwidth-limited on device reads. 1536×16=24,576 workgroups × 128 threads = healthy parallelism.

**Approach**: Already reasonable. Could improve to ~50-60 GB/s with better memory access patterns or wider loads. Lower priority than #1.

### Time Budget Analysis

| Component | Est. time/tok | Notes |
|-----------|---------------|-------|
| Kernel #1 (MoE down) | ~24ms | 2 calls/block × ~12ms |
| Kernel #2 (gate·silu·up) | ~13ms | 4 calls/block |
| Kernel #3 (shared expert) | ~6ms | 4 calls/block |
| Kernel #4 (MoE proj) | ~4.5ms | 2 calls/block |
| Q4K attention linears | ~3.5ms | 572 calls |
| ~2700 other small kernels | ~20ms | dispatch overhead dominant |
| **Total GPU** | **~71ms** | |
| Python/framework | ~11ms | |
| **Total wall** | **~82ms** | 12.2 tok/s |

### Path to 50 tok/s (20ms/tok)

1. Fix kernel #1: 2→80 GB/s would be ~0.6ms/call → saves ~19ms → **63ms/tok**
2. Fix kernel #4: similar → saves ~3.5ms → **59ms/tok**
3. Fix kernel #2: 14→60 GB/s → saves ~10ms → **49ms/tok**
4. Reduce dispatch overhead for small kernels → saves ~10ms → **39ms/tok**
5. Fix kernel #3: marginal → ~37ms/tok → **~27 tok/s**

Fixing #1 alone is the biggest lever: **12.2 → ~16 tok/s**. Fixing #1+#2+#4 (all Q5K MoE): **~20 tok/s**.

---

## Road to 50 tok/s

### Phase 0: Baseline GLM with current fixes — DONE
- [x] Run GLM benchmark with pairwise topk + MoE fusion break: **12.2 tok/s** (was 9.3)
- [x] Profile with `VIZ=-1` to get kernel landscape
- [x] Measure actual tok/s improvement: **31% gain** from topk + fusion break
- See "GLM Kernel Profile — Feb 5 2026" section above for full analysis

### Phase 1: Fix Q5K MoE kernel parallelism (~12→20 tok/s)
The #1 bottleneck (`r_9_32_4_16_4_6_4_2_32`) runs at **2 GB/s** — only 288 workgroups with
64 threads each, fully serial reduction over 1536 elements, scattered byte-level Q5K reads.
This one kernel is 28.4% of total GPU time.

**Options (prefer tinygrad DSL):**
- **GROUPTOP for reduction**: Force threadgroup cooperation on the dot product
- **Pre-dequant selected experts**: Dequant 4 selected experts → fp16, then standard matmul
- **Custom Q5K GEMV (UOp DSL)**: Proper SIMD reduction + coalesced reads, like Q4K runner
- **Separate expert kernels**: `.contiguous()` per expert for better per-kernel parallelism

### Phase 2: Fuse MoE gate·up into single kernel (~20→25 tok/s)
Currently gate and up projections for experts run as one fused Q5K dequant kernel but with
two sequential GEMV loops (83ms total, #2 bottleneck). Consider:
- Single-pass gate·silu·up kernel that dequants once, computes both projections
- Pre-interleave gate+up weights so one memory scan produces both
- This is the `r_88_32_3_8_4_2_32_8_4_2_32` kernel

### Phase 3: Reduce small kernel overhead (~25→35 tok/s)
~700 tiny elementwise kernels (RMSNorm outputs, residuals, casts) at ~5-10us each = 3-7ms.
- Fuse RMSNorm + first linear: norm output feeds directly into matmul
- Fuse residual + norm: `h + attn_out` → `rmsnorm(h + attn_out)` in one kernel
- Fuse cast chains: `float().silu()` → `.cast()` sequences

### Phase 4: Dispatch overhead reduction (~35→50 tok/s)
At ~2000 kernels × ~20us dispatch overhead = ~40ms of pure overhead.
- **Remove ICB barriers**: `metal.py:50` puts `.setBarrier()` on every kernel.
  Many are independent (q_a/kv_a, gate/up). Selective barriers could save 5-15ms.
- **Increase JIT_BATCH_SIZE**: Start at 64 or 128 instead of 32
- **Reduce ICB count**: Fewer command buffer submissions = less CPU-GPU sync

### Phase 5: Custom MoE kernel (stretch → 50+ tok/s)
Write a Metal `mul_mat_id` equivalent: one kernel dispatches per-expert threadgroups
with proper GEMV parallelism. Eliminates expert gather, expert loop, and scattered access.
Combine with Q4K dequant inline (like existing Q4KRunner pattern).

---

# Youtu-LLM 2B Q4 Speed History

## Dev Harness Goal
- Use Youtu (`youtu-llm:2b-Q4`) as the fast iteration harness for MLA/QK codepaths.
- Push Youtu to high throughput first, then transfer wins to GLM-4.7-Flash.

## Benchmark Protocol
- Command:
```bash
.venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4" --benchmark 20
```
- Correctness gate:
```bash
.venv2/bin/python tinygrad/apps/smoke_test.py youtu-q4
```
- Kernel profile:
```bash
DEBUG=2 .venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4" --benchmark 5
```

## Current Status (Feb 5, 2026)
- Baseline steady-state (median tail, `--benchmark 20`): **~20.68 tok/s**
- Previous steady-state: **~31.78 tok/s**
- **Current steady-state (median tail, `--benchmark 20`): ~55.5 tok/s**
- Net gain from baseline: **~2.7x**
- Per-token time: **~18ms** (down from ~32ms)
- Effective param bandwidth: **~38.6 GB/s** (of 100 GB/s theoretical)

Recent run (`bench_after_20.log`):
- Tokens 13-20 stable at 55.0-56.3 tok/s
- Occasional dips to ~41-48 tok/s (GPU thermal/scheduling variance)

## Landed Changes
1. `tinygrad/apps/llm.py`
- Greedy decode now uses `argmax(logits)` directly (no redundant `softmax`).
- Standard attention KV write changed from `Tensor.stack(k, v)` to two assigns.
- Removed trailing `.contiguous()` on standard block return to reduce a fusion barrier.

2. `tinygrad/apps/quantized.py`
- Added `QK_DEQUANT_CACHE` (default `1`) for non-Q4 linears.
- Q5/Q6 weights now dequantize once and are reused, instead of repeated per-token dequant work.
- This was the largest reliable speedup (20→31 tok/s).

3. `tinygrad/nn/metal_q4k.py` — **Q4K CompiledRunner (NEW, largest win)**
- Q4KRunner now extends `CompiledRunner` instead of `Runner`.
- O, K dimensions baked as compile-time `#define` constants (per-shape kernel compilation).
- Proper `ProgramSpec` with `globals=[0,1,2], outs=[0], ins=[1,2], vars=[]`.
- `__call__` override handles dynamic batch for non-JIT prefill.
- **Result**: Q4K dispatches now batch into MetalGraph ICBs. JIT replay items dropped from **417 → 5**.
- **Impact**: **31 → 55.5 tok/s** (~1.8x speedup, ~18ms per token).

4. `tinygrad/nn/metal_q6k.py` — **Q6K CompiledRunner (NEW)**
- Same Runner→CompiledRunner conversion as Q4K.
- Still defaulted to off (`Q6K_MSL=0`), but now JIT-batchable when enabled.

5. `tinygrad/apps/mla.py`
- Added optional `Q4K_DUAL_GATE_UP` path for fused gate/up projection (default `0`, experimental).

## JIT Batching Analysis

### Before (Q4KRunner extends Runner)
- 975 total kernels captured by JIT
- **417 dispatch items** per token: 209 tiny MetalGraph ICBs (avg 3.7 kernels) + 208 individual Q4K dispatches
- Each Q4K dispatch created its own Metal command buffer
- Q4K dispatches between compiled kernels forced flush + new ICB, fragmenting batches

### After (Q4KRunner extends CompiledRunner)
- 975 total kernels captured by JIT
- **5 dispatch items** per token: 5 MetalGraph ICBs (32+64+128+256+495 kernels)
- All Q4K dispatches now inside ICBs
- **83x reduction** in command buffer submissions

## Current Hotspots (from DEBUG=2)
Top Q4 kernels by total time:
- `q4k_linear_msl_6144x2048` (FFN gate/up): dominant
- `q4k_linear_msl_3072x1536`
- `q4k_linear_msl_1536x2048`
- `q4k_linear_msl_2048x2048`
- `q4k_linear_msl_2048x6144`
- `q4k_linear_msl_576x2048`

Non-Q4 kernels still significant:
- `r_16_128n2`, `r_16_32`, `r_16_96`, and related reduction-heavy kernels.

## What Did Not Hold Up
- Replacing MLA manual attention with `scaled_dot_product_attention` regressed correctness/perf in this path.
- Removing MLA KV cache write `.realize()` broke correctness (read-after-write ordering issue).
- Enabling `Q6K_MSL=1` by default regressed end-to-end Youtu throughput (keep default off for now).
- `Q4K_DUAL_GATE_UP` microbench looked promising but end-to-end behavior was unstable; keep default off.

### Q6K MSL Kernel (`metal_q6k.py`) — Assessment
- **Numerics are fixed**: bit extraction now matches `q6k_moe_fused` and dequant reference tests.
- **Default stays off** (`Q6K_MSL=0`) because current end-to-end throughput is lower than cached dequant path.
- **Relevant to Youtu**: Youtu Q4 model still has Q6K tensors (e.g. `output` and some FFN down projections).
- **Performance is still weak**: naive `lane * 8 + i` mapping means adjacent threads read 8 addresses apart (uncoalesced).
  The Q4K kernel uses interleaved `ix=tiisg/8, it=tiisg%8` for coalesced 16-bit weight reads + batch uint16 accumulation.
  Q6K kernel would need the same interleaving treatment to match Q4K's ~400 GB/s; as-is probably ~60-100 GB/s.
- **Now JIT-batchable** (extends CompiledRunner), so enabling it won't break ICB batching.
- **Verdict**: correct but not yet faster. Revisit when Q6K throughput matters.

---

## Deep Analysis: Remaining 2.6x Gap

### Theoretical Limit
- 0.69 GB params at 100 GB/s Metal bandwidth = **~145 tok/s** (6.9ms per token)
- Current ~55.5 tok/s = ~18ms per token → **2.6x gap** to theoretical

### Where Time Goes Per Token (~18ms)
| Source | Est. time | Notes |
|--------|-----------|-------|
| Actual GPU compute | ~7ms | 0.69 GB / 100 GB/s |
| ICB dispatch + barrier overhead | ~5-8ms | 975 kernels in 5 ICBs, all serialized by `.setBarrier()` |
| Python/JIT framework overhead | ~3-5ms | JIT replay loop (5 items), buffer management |

### Why 975 Kernels Still Matters
Even inside ICBs, every kernel has `.setBarrier()` serializing execution. The GPU can't overlap independent operations. And 975 kernel launches within ICBs still have per-dispatch overhead (pipeline state switches, buffer bindings).

**llama.cpp uses ~50-100 graph nodes** for the same model because it fuses aggressively into mega-kernels.

### Key Architectural Insight
For T=1 token generation, every operation is **memory-bound** (tiny batch, large weight matrices). Intermediate results are tiny (~2048 × fp16 = 4KB) and fit in threadgroup memory or registers. There is no compute bottleneck — only memory reads and launch overhead.

---

## Experiment Log (Idea -> Result -> Evidence)

| Idea | Result | Evidence |
|------|--------|----------|
| Greedy `argmax` without `softmax` | Panned out | `tinygrad/apps/llm.py`, smoke pass |
| QK dequant cache for non-Q4 | Panned out (largest reliable win, 20→31) | `tinygrad/apps/quantized.py` |
| Q4KRunner → CompiledRunner | **Panned out (31→55.5 tok/s)** | 417→5 dispatch items, `bench_after_20.log` |
| Q6KRunner → CompiledRunner | Panned out (consistent change) | `tinygrad/nn/metal_q6k.py` |
| Remove MLA cache assign `.realize()` | Did not pan out (correctness break) | `smoke_test.py youtu-q4` mismatch |
| MLA SDPA swap in absorbed path | Did not pan out (correctness/perf regression) | smoke mismatch + slow decode |
| Q4 dual gate/up fused path | Did not pan out yet (unstable end-to-end) | microbench good, full benchmark inconsistent |
| Q6K MSL numerics fix | Panned out for correctness, not speed | q6k numeric test |
| Pairwise topk (replace bitonic sort) | **Panned out (29→3 kernels, 4.4x faster)** | `topk_test.py`, wired into `mla.py` as `_topk_pairwise` |
| Custom kernel topk (UOp DSL) | Did not pan out (linearizer assertion / wrong results) | 8-register reduce loop too complex for compiler |
| Break MoE down proj fusion | **Panned out (18→25.5 tok/s on ds2-lite, 41%)** | `.contiguous()` after down proj, now default in `mla.py` |

## Journal — Feb 5 2026 (evening): MV Heuristic and MoE Kernel Optimization Attempts

### Context
After establishing the GLM baseline at 12.2 tok/s and profiling the top kernels, I tried several
approaches to improve the fused Q5K dequant+matmul MoE kernels (which account for ~49% of GPU time).
Used deepseek-v2-lite as the fast iteration target (26 tok/s baseline, ~10s per benchmark).

### Approach 1: Remove unnecessary .float() casts
**Result: Neutral (26.14 vs 25.94 tok/s), kept for cleanliness.**

- Removed `.float()` no-op casts in `mla.py` (ExpertWeights already returns fp32→fp16)
- Removed `.float()` casts inside `QuantizedExpertWeights.__call__` — matmul now runs in fp16
  (dequant's native output type) instead of unnecessarily casting to fp32
- Smoke test passes on all 3 models (youtu, deepseek, glm)
- Fewer ops = fewer kernels = cleaner code, even if no measurable speedup

### Approach 2: `.contiguous()` after dequant in ExpertWeights
**Result: REGRESSION (14.70 tok/s, 68ms). Reverted.**

- Idea: Break the fused dequant+matmul so the matmul gets proper parallelism
- Problem: Extra memory traffic from materializing full fp16 weights dominates.
  Fused kernel reads compact Q4K blocks (144 bytes/256 elements); breaking forces
  fp16 write + read (1024 bytes/256 elements) = 7x more memory traffic
- Lesson: For on-the-fly expert dequant, fusion is correct — the problem is the
  PARALLELISM of the fused kernel, not the fusion itself

### Approach 3: Relax MV heuristic in `heuristic.py`
**Result: Only 1 extra MATVEC match (from 0 to 1). MoE kernels still unmatched.**

The MV (matvec) heuristic at `heuristic.py:65-81` requires `MUL(INDEX, INDEX)` — both operands
of the multiply must be direct INDEX ops. For fused dequant+matmul, the pattern is
`MUL(dequant_chain(INDEX(blocks)), INDEX(activation))` — fails the check.

**Attempt**: Walk `backward_slice` to find INDEX ops through dequant chains, use `.ranges`
instead of `split_uop(Ops.ADD)`, try both orderings.

**Why it still fails for MoE kernels**:
1. The fused dequant creates MULTIPLE reduce ranges (block index + element-within-block).
   The activation only has the combined feature range. `all(r in idx_b.ranges for r in idx_a.ranges
   if r in reduce_rngs)` fails because both reduce ranges need to appear in both indices.
2. The weight INDEX has EXTRA ranges (dequant sub-block indices) not present in the activation INDEX.
   The `ranges_subset` check fails in both orderings.
3. The fundamental issue: MV heuristic assumes simple `y[i] = sum(a[j] * b[i,j])` structure.
   Fused dequant creates `y[i] = sum(f(blocks[expert, block_j, :]) * x[j])` — multi-dimensional
   reduce with complex index expressions that don't decompose into MV's expected pattern.

**Reverted** — the 1 extra match (a non-MoE matmul `[64, 2048]`) gave negligible speedup.

Also checked: GROUPTOP doesn't apply because `prod(output_shape[upcastable_dims]) = 73,728 > 2048`.
The 2048 limit exists to prevent threadgroup memory overflow. Can't easily raise it.

### Approach 4: MOE_FUSED custom kernels
**Result: 6x SLOWER (4.25 tok/s, 235ms). Not viable.**

Existing `q4k_moe_fused`, `q5k_moe_fused`, `q6k_moe_fused` kernels in `quant_kernels.py`
(gated behind `MOE_FUSED=0`). These are hand-written UOp DSL kernels with proper parallelism
(64 output features per workgroup, cooperative input loading, register accumulation).

Why they're slow:
- Custom kernels go through normal compilation but may not JIT-batch efficiently
- Byte-level scattered reads within the kernel (each Q4K/Q5K element requires 3+ scattered reads)
- The existing fused-by-tinygrad path actually does better despite poor parallelism, because
  tinygrad's fusion avoids the overhead of the custom kernel dispatch chain
- Warm-up takes 5+ iterations (4333ms → 235ms) suggesting JIT compilation overhead

### Key Insight: The Correct Approach

The problem is NOT that the fused kernel exists — it's that tinygrad's optimizer can't apply
GROUP/GROUPTOP to it because:
1. MV heuristic: pattern match too restrictive for fused dequant
2. GROUPTOP: output too large (73K elements > 2048 limit)

**What would actually help**:
- Teach the heuristic to recognize fused dequant+matmul as a reduction pattern
  that benefits from GROUP. This requires matching the MUL-reduce through arbitrary
  dequant chains, not just through direct INDEX ops.
- OR: restructure the expert computation so each expert's matmul is a separate kernel
  with manageable output size (fits GROUPTOP's 2048 limit), then sum results.
  But `.contiguous()` break already showed this regresses due to memory traffic.
- OR: write a PatternMatcher rewrite that normalizes fused dequant+matmul into a form
  the existing MV heuristic can recognize (e.g., replace the dequant chain with a
  single LOAD+CAST before matching).

The real path forward for tinygrad native: make the OPTIMIZER smarter about fused dequant
patterns, not write custom kernels around it.

### Approach 5: BEAM=2 on isolated MoE expert kernel
**Result: No improvement (7.50ms vs 7.20ms baseline). BEAM can't fix this.**

Isolated the fused dequant+matmul expert down projection (`moe_beam_test.py`):
64 experts, top-6, 2048×1536 Q4K. BEAM=2 selected `r_6_128_16_6_4_2_8_4` with
opts `(LOCAL axis=1 16, UNROLL axis=3 4)`.

Per-kernel GPU times (Metal profiler):
- `E_2_1728_4_16_4_4_3` (expert gather): ~160us
- `r_6_128_16_6_4_2_8_4` (fused dequant+matmul): ~460us
- **Total GPU time: ~620us** — but wall-clock is 7.50ms (no JIT in test)

The kernel achieves **23 GB/s** (23% of Metal's ~100 GB/s) regardless of opt choices.
BEAM explored the search space and couldn't find better options. The bottleneck is
structural: scattered byte-level Q4K reads and complex dequant arithmetic per element.

### Deepseek-v2-lite Benchmarks Summary

| Configuration | tok/s | ms/tok | Notes |
|--------------|-------|--------|-------|
| Baseline (current code) | 25.94 | 38.5 | No MV, no MOE_FUSED |
| .float() removal only | 26.14 | 38.3 | Cleaner, kept |
| .contiguous() dequant break | 14.70 | 68.0 | **REGRESSION**, reverted |
| Relaxed MV heuristic | 26.43 | 37.8 | 1 extra MATVEC, reverted |
| MOE_FUSED=1 | 4.25 | 235.0 | **6x SLOWER**, not viable |
| BEAM=2 isolated MoE kernel | — | 7.50ms* | *standalone test, no JIT overhead |
| PER_EXPERT=1 (separate per-expert matmuls) | 21.7 | 46.1 | **REGRESSION -18%**, reverted |
| FUSED_GROUP=1 (fallback GROUPTOP) | 27.3 | 36.6 | +3.4%, too blunt, reverted |
| Q4K MoE MSL CompiledRunner | 8.24* | 121* | *battery mode, needs re-test plugged in |
| **Q6K MoE MSL (llama.cpp pattern)** | **17.97** | **55.6** | **+48% on GLM, #1 bottleneck fixed** |
| **Selective .contiguous() split (Q5K+Q6K)** | **14.0** | **71** | **+27% on GLM, no MSL needed** |

## Journal — Feb 6 2026: MoE Kernel Optimization — Heuristics & MSL CompiledRunner

### Context
Continuing from the Feb 5 analysis that identified `r_9_32_4_16_4_6_4_2_32` (MoE down proj) as
the #1 bottleneck at 2 GB/s. Tried three approaches on deepseek-v2-lite (26.4 tok/s baseline).

### Approach 6: Per-Expert Dispatch (PER_EXPERT=1)
**Result: REGRESSION — 21.7 tok/s (-18%). Reverted.**

Split the batched 6-expert matmul into a `for i in range(n_sel)` loop of individual expert matmuls.
Each expert's matmul has output size 2048, fitting GROUPTOP's 2048 threshold.

Why it regressed:
- 6× more kernel dispatches per MoE layer (6 individual matmuls vs 1 batched)
- Dispatch overhead dominates at this scale — each dispatch costs ~20us
- 6 × 2 MoE projections × ~20us = 240us extra overhead per block
- The batched kernel, despite poor parallelism, amortizes dispatch overhead

### Approach 7: FUSED_GROUP Fallback GROUPTOP
**Result: +3.4% (27.3 tok/s). Reverted — too blunt.**

Added fallback in `heuristic.py`: if no MV or GROUPTOP was applied AND the kernel has a large
reduction (>= 512 elements), apply GROUPTOP=16 unconditionally.

Why it was reverted:
- The 2048 output threshold for GROUPTOP exists to limit threadgroup memory usage.
  `GROUPTOP=16` allocates `output_elements × 16 × sizeof(float)` in threadgroup memory.
  For batched experts (output = 6 × 2048 = 12,288), this is 12,288 × 16 × 4 = 768KB — way over
  Metal's 32KB threadgroup memory limit.
- The +3.4% came from applying GROUP to a few other kernels, not the target MoE kernel.
- A more nuanced approach would need to: (a) check actual threadgroup memory requirement
  before applying, (b) pick GROUP size based on available memory, (c) handle the fused dequant
  pattern specifically.

### Why Heuristics Don't Apply GROUP Natively (Root Cause Analysis)

Two independent failures block the optimizer:

1. **MV heuristic** (`heuristic.py:65-81`): Pattern-matches `MUL(INDEX, INDEX)`.
   Fused dequant creates `MUL(dequant_chain(INDEX), INDEX)` — fails the match.
   Even relaxing to walk through dequant chains, the reduce ranges don't align:
   weight has extra sub-block ranges (block_idx × elements_per_block) that activation lacks.
   `all(r in idx_b.ranges for r in idx_a.ranges)` fails in both orderings.

2. **GROUPTOP threshold** (`heuristic.py:83-89`): `prod(output_shape[upcastable_dims]) <= 2048`.
   Batched expert kernel: 6 experts × 2048 output features = 12,288 > 2048. Fails threshold.
   The threshold prevents threadgroup memory overflow. Can't simply raise it.

The correct fix would be a new heuristic that:
- Recognizes fused dequant+matmul as a reduction pattern (not just `MUL(INDEX, INDEX)`)
- Calculates actual threadgroup memory needed based on GROUP size × output per workgroup
- Applies GROUP where it fits within Metal's 32KB limit

### Approach 8: Q4K MoE MSL CompiledRunner
**Result: Smoke test passes. Benchmark inconclusive (8.24 tok/s — laptop on battery).**

Wrote a Metal shader that adapts the proven `q4k_linear` kernel for MoE expert indexing:
- Same SIMD reduction, NR=8 rows/workgroup, 4 simdgroups × 32 threads = 128 threads
- Added expert index lookup: `expert_indices[sel_idx]` → offset into weight buffer
- Wraps as `CompiledRunner` (extends `Q4KRunner` pattern) for JIT ICB batching
- Wired into `QuantizedExpertWeights.__call__` for Q4K + Metal + `expert_first_in_memory`

Key difference from tinygrad-fused path:
- Q4K MSL kernel achieves ~80 GB/s on regular linears (proven)
- For MoE: same access pattern but with expert offset added to weight pointer
- Expected: ~60-80 GB/s (expert offset adds one indirection but access within expert is sequential)

Benchmark result of 8.24 tok/s is unreliable — user confirmed laptop was on battery (GPU throttled).
**Needs re-test plugged in to get real numbers.**

Files changed:
- `tinygrad/nn/metal_q4k.py`: Added `_make_q4k_moe_src`, `Q4KMoERunner`, `q4k_moe_linear_msl`
- `tinygrad/apps/quantized.py`: Added Q4K MoE MSL code path in `QuantizedExpertWeights.__call__`

### Current State
- Q4K MoE MSL kernel written, smoke test passes, awaiting plugged-in benchmark
- If it works: Q4K MoE experts run at ~60-80 GB/s (vs 2 GB/s for tinygrad-fused)
- Q5K MoE experts (GLM's main bottleneck) still need the same treatment
- Topk_pairwise confirmed optimal for MoE routing (3 kernels, same algorithm as sort_v3_rank)

---

## Next Prioritized Work (Feb 7 — Q4_0 packed-dot approach)

See "Plan: GLM Q4_0 → 50 tok/s" at top of file.

### Phase 1: Packed-dot Q4_0 for QuantizedExpertWeights
Restructure `QuantizedExpertWeights.__call__` to use packed-byte dot products for Q4_0:
- `lo_nib * x_lo + hi_nib * x_hi` per packed byte, avoiding 2x tensor expansion
- Offset correction: `scale * (dot(nibs, x) - 8 * sum(x_block))` per block
- Pure tinygrad tensor ops, targets 25+ GB/s (from 13 GB/s)

### Phase 2: Packed-dot for QuantizedLinear (attention, shared expert)
Same approach for cached-dequant linears. Currently these dequant to fp16 then matmul.
Packed-dot skips fp16 materialization: reads packed Q4_0 bytes + does dot inline.

### Later: Kernel count reduction, ICB barrier removal, BEAM optimization
See plan phases 3-4 at top of file.

---

## GLM-4.7-Flash Analysis (Target Model)

### Stats
- **1.24 GB params**, 47 blocks, deepseek2 MLA, **64 experts** + 1 shared expert
- **2899 kernels** per token, batched into **7 ICBs** (32+64+128+256+512+1024+883)
- Steady-state: **~9.3 tok/s** (~106ms per token)
- Theoretical at 100 GB/s: 1.24 GB / 100 GB/s = 12.4ms = **~80 tok/s**
- Gap: **~8.6x**
- Q4K shapes: 141× `768x2048`, 141× `5120x768`, 141× `2048x5120`, 6× `10240x2048` (429 total)

### Where Time Goes Per Token (~106ms)
| Source | Est. time | Notes |
|--------|-----------|-------|
| Actual GPU compute | ~12ms | 1.24 GB / 100 GB/s |
| ICB dispatch + barrier overhead | ~60-70ms | 2899 kernels × ~22us avg overhead |
| Python/JIT overhead | ~20-25ms | JIT replay (7 items) + scheduling |

### TopK Bitonic Sort — SOLVED with Pairwise Ranking

**Problem**: Each MoE block has a `topk` operation for expert routing (select top-k from 64 experts).
tinygrad compiled this as a **bitonic sort**: ~20 tiny kernels per topk (`E_32_2`, `E_16_2_2`, `E_8_4_2`, etc.).
- 47 blocks × ~20 topk kernels = **~940 kernels** (32% of total!)
- Each topk kernel takes 6-9us → ~160us per block just for routing
- 47 blocks × 160us = **~7.5ms** purely in topk dispatch overhead

**Fix**: `_topk_pairwise` in `mla.py` — O(n²) pairwise comparison to compute stable ranks, then extract top-k by rank matching.
- Pure tinygrad ops, no custom kernel DSL, works on any backend
- **29 → 3 kernels** per topk call, **4.4x faster** in isolation
- Expected GLM kernel reduction: ~940 → ~141 topk kernels (3 per block × 47 blocks)
- Enabled by default (`PAIRWISE_TOPK=1`), falls back to `Tensor.topk` when off

Approaches tried:
| Approach | Kernels | Time (us) | Status |
|----------|---------|-----------|--------|
| Bitonic sort (`Tensor.topk`) | 29 | 20,318 | baseline |
| Argmax iterations (`_topk_simple`) | 11 | 12,115 | works, too many kernels |
| Scatter iterations | 11 | 13,151 | same as argmax |
| **Pairwise ranking** | **3** | **4,624** | **winner, landed** |
| Pairwise v2 (eps tiebreak) | 4 | 4,778 | not tie-safe |
| Custom kernel (UOp DSL) | — | — | linearizer assertion failure |

#### Sort Algorithm Analysis (Feb 6 2026)

Benchmarked replacing bitonic sort with alternative algorithms for MoE topk (N=6-64):

| Algorithm | N=64 | N=256 | N=1024 | N=4096 | Kernels | Notes |
|-----------|-------|--------|--------|--------|---------|-------|
| Bitonic (current `tensor.sort`) | 19.8ms | 31.6ms | 47ms | 67ms | 9-15 | O(N log²N), `.contiguous()` per substage |
| Rank-based (pairwise) | 2.4ms | 2.4ms | 2.5ms | 2.6ms | 2-3 | O(N²) compute, constant kernel count |
| Bitonic no-contiguous | 4.7ms | 33.8ms | — | — | 1 | Graph rewrite explodes at N>64 |
| Bitonic stage-contiguous | — | 29.9ms | — | — | fewer | Scheduling explodes (28s at N=256) |

**Key findings**:
- Rank-based O(N²) dominates up to N~200K because kernel count is constant (2-3 vs O(log²N))
- tinygrad fuses the O(N²) comparison matrix into the reduction — no O(N²) memory materialized
- Removing `.contiguous()` from bitonic achieves 1 kernel but graph rewrite time explodes exponentially
- The bottleneck for bitonic is Python loop rebuilding UOp graph (~15-30ms), not kernel dispatch
- Crossover (rank vs bitonic) is ~200K-250K elements, hardware-dependent (GPU speed)
- For MoE routing (N=6-64), rank-based is optimal — same algorithm as `_topk_pairwise`

### MoE Down Projection Fusion — SOLVED with `.contiguous()` Break

**Problem**: tinygrad fused MoE down projection + weighted expert sum + shared expert + residual
into ONE kernel (`r_2048_16_1408_6_176`). The compiler chose `GROUPTOP=16` for the shared expert
reduction, but the MoE down projection (6 experts × 1408 inner dim = 8,448 iterations) ran fully
serial — all 16 threads redundantly computed the same result. Result: 4.5ms at 31 GB/s.

**Root cause**: When tinygrad fuses reductions with different parallelism needs, the compiler picks
ONE threadgroup size. The shared expert needed 16 threads for its 2816-element reduction. The MoE
down projection needed many more threads for its 8,448-element reduction. Fusing them forced the
MoE part into the 16-thread straitjacket.

**Fix**: `.contiguous()` after `ffn_down_exps()` breaks the fusion. The down projection gets its
own kernel with proper parallelization; the weighted sum + shared expert + residual become a
separate kernel. Now default in `mla.py`.

**Result**: deepseek-v2-lite 55→39ms/tok, **18→25.5 tok/s (41% speedup)**.

**Lesson**: llama.cpp never fuses MoE. Separate `mul_mat_id` per projection (gate/up/down),
explicit adds for weighted sum, shared expert fully independent. See `llamacpp.md`.

### Per-MoE-Block Kernel Breakdown (~62 kernels)
1. RMSNorm: 2 kernels
2. Q-LoRA: q4k_768x2048 → norm → q4k_5120x768 → 5 kernels
3. KV compression: norm → linear → norm → cast → 4 kernels
4. Attention: qk → softmax(3) → v → v_proj → 7 kernels
5. Output proj: q4k_2048x5120 → 1 kernel
6. Residual + RMSNorm: 2 kernels
7. **Router**: linear → **topk (~3 kernels with pairwise, was ~20)** → expert select
8. Expert gather: E_18432×2 + E_26880 → 3 kernels
9. Fused expert compute: r_64_32_3_... → 1 kernel
10. MoE reduce + shared expert: 4-5 kernels
11. Total: ~45 per block × 47 blocks ≈ 2100 (was ~62 × 47 ≈ 2900 before pairwise topk)

### GLM vs Youtu Comparison
| | Youtu 2B Q4 | GLM-4.7-Flash |
|--|-------------|---------------|
| Params | 0.69 GB | 1.24 GB |
| Blocks | 32 | 47 |
| Experts | shared only | 64 + 1 shared |
| Kernels/token | 975 | 2899 |
| ICBs | 5 | 7 |
| tok/s | 55.5 | 9.3 |
| Theoretical | 145 | 80 |
| Gap to theory | 2.6x | 8.6x |

### GLM-Specific Priorities
1. ~~**Fuse topk into 1 kernel**~~ — **DONE** via `_topk_pairwise` (29→3 kernels per topk)
2. ~~**Break MoE down proj fusion**~~ — **DONE** via `.contiguous()` (18→25.5 tok/s on ds2-lite)
3. **Reduce MoE kernel count** — expert gather + compute + reduce still many kernels per block
4. Transfer youtu optimizations (already done: CompiledRunner, dequant cache)

---

## What Did NOT Change (Lessons)
- Individual Q4 kernel optimization (already at 400 GB/s — near memory bandwidth limit)
- Attention algorithm changes (MLA absorbed path is correct and efficient)
- Quantization format changes (Q4_K is the right format for this model size)

## Milestones (Revised)
### Youtu 2B Q4
- M1: 35 tok/s ← **DONE**
- M2: 50 tok/s ← **DONE (55.5 tok/s)**
- M3: 70 tok/s — barrier removal + initial fusion
- M4: 100+ tok/s — fused MLA block kernel or mega-kernel

### GLM-4.7-Flash (Q4_0 path — current focus)
- G0: 12.2 tok/s — Q4_K_M baseline with pairwise topk + MoE fusion break (DONE, Feb 5)
- G0.5: 18.0 tok/s — Q6K MoE MSL kernel for Q4_K expert down projections (DONE, Feb 6) — MSL removed
- G0.7: 14.0 tok/s — selective .contiguous() split for Q5K/Q6K experts (DONE, Feb 6) — Q4_K path
- G1: **14.6 tok/s** — Q4_0 (unsloth), pure tinygrad, no MSL (**DONE, Feb 7**) — current baseline
- G2: ~25 tok/s — packed-dot Q4_0 for QuantizedExpertWeights (Phase 1)
- G3: ~30 tok/s — packed-dot Q4_0 for QuantizedLinear (Phase 2)
- G4: ~40 tok/s — kernel count reduction (Phase 3)
- G5: 50 tok/s — BEAM + dispatch optimization (Phase 4)

---

## Youtu Q4_0 Sprint Results (Feb 7, 2026)

### Status: 52 tok/s (was 26.5 baseline, was 55.5 with MSL)

**No MSL kernels.** Pure tinygrad tensor DSL. Rule: no custom kernels, no BEAM.

### Changes Made & Impact

| # | Change | tok/s | Kernels | Delta | Status |
|---|--------|-------|---------|-------|--------|
| 0 | Baseline (fp16 dequant cache) | 26.5 | 780 | — | was |
| 1 | Q4_0 packed-dot QuantizedLinear (v1, 3 reduces) | 45.5 | 1004 | +224 kernels, +19 tok/s | replaced |
| 2 | Q4_0 packed-dot v2 (subtract 8 inline, 1 reduce) | 52.0 | 780 | 0 kernels, +25 tok/s | **CURRENT** |
| 3 | Eliminate cache_v (V = slice of K) | 52.0 | 650 | -130 kernels | **CURRENT** |
| 4 | Remove .float() casts in dense FFN | 52.0 | 650 | 0 | **CURRENT** |
| 5 | Remove .float() from softmax | 51.8 | 650 | 0 | **CURRENT** |

### Key Decisions

1. **Packed-dot Q4_0**: Read quantized blocks directly, compute matvec inline. 3.56x less bandwidth than fp16 cache. The trick is `(nib - 8) * x` instead of `nib * x - 8 * sum(x)` — avoids separate x_block_sum kernel.

2. **Single K cache, no V**: V is `cache_k[:,:,:,:kv_lora_rank]`. Saves 4 kernels/block (cache write + associated ops) × 32 blocks = 128 kernels. Matches llama.cpp's MLA approach.

3. **Absorbed MLA is correct**: Researched naive (HuggingFace expand-at-cache), absorbed (llama.cpp/DeepSeek), and split-scores (DeepSeek reference). Absorbed wins for batch=1: smaller cache, fewer bytes, MQA attention. See `mla.md` for full analysis.

### Where Time Goes (650 kernels, 19.2ms/tok)

| Category | Kernels/tok | % of kernels |
|----------|-------------|-------------|
| Large reduce (matmul/MoE) | 273 | 42% |
| Elementwise | 139 | 21% |
| Small reduce (norm/softmax/topk) | 238 | 37% |

At 29.7 us/kernel average overhead, **dispatch cost = 19.3ms**. GPU compute time is ~0.1ms in ICBs. The entire bottleneck is dispatch overhead.

### What Didn't Help
- JIT_BATCH_SIZE=780 (single ICB): 50.3 tok/s — marginal
- GROUPTOP_LIMIT increase: no change
- Remove .contiguous() from FFN silu: -32 kernels but slower (kernel explosion risk)
- Remove .float() casts: 0 kernel change (tinygrad already optimizes away)

### Gap Analysis

- **Current**: 52 tok/s = 19.2ms = 650 kernels × 29.5us
- **Target**: 80 tok/s = 12.5ms
- **Need to cut**: 6.7ms → ~227 fewer kernels at same overhead, or same kernels at 19.2us/kernel

### Next Steps (Youtu Q4_0)
1. **Profile with DEBUG=5** to see actual Metal kernel source and identify fusion opportunities
2. **Reduce attention kernel count**: 20/layer → 12/layer would save 256 kernels → ~14.5ms → 69 tok/s
3. **Fuse RMSNorm into following matmul**: saves 2 kernels/layer = 64 total
4. **Fuse absorbed einsums with surrounding ops**: K absorb + cat, V absorb + output proj
5. **Consider replacing einsum with matmul+reshape**: may fuse better with tinygrad scheduler
