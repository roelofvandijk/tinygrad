# Final Conclusion: Youtu 2b Q4_0 Optimization (Feb 8, 2026)

## The Hard Truth

**Current**: 56 tok/s (586 kernels, 17.8ms/token)
**Target**: 70 tok/s (14ms/token)
**Gap**: 3.8ms

### What We Tried

1. ✅ **Softmax fp16 no-max**: 2 kernels in isolation vs 5 baseline
   - Full model: 586 kernels (same as baseline)
   - The scheduler already fuses optimally in context

2. ✅ **Builtin softmax()**: 3 kernels in isolation vs 4 manual
   - Full model: 618 kernels (+32 from `.max()`)
   - Slower, not better

3. ✅ **Cat elimination**: 4.6x faster in isolation
   - Full model: 652 kernels (+66), timeout risk
   - Split cache creates more barriers than it saves

### The Pattern

**Isolated micro-benchmarks don't predict full-model behavior**. The scheduler's fusion decisions depend on the entire graph structure, cache assignments, JIT patterns, and symbolic variables.

##The Real Bottleneck

**Dispatch overhead dominates**:
- 586 kernels × 34us = 19.9ms
- Compute (weight reads) = 7ms @ 100 GB/s
- Total: ~27ms, but JIT batching gets us to 17.8ms actual

**To reach 14ms**:
- Option A: 586 kernels × 24us = 14.1ms (need 10us/kernel improvement in Metal)
- Option B: 412 kernels × 34us = 14.0ms (need to cut 174 kernels)

### Why Model Tweaks Don't Work

The 586-kernel structure is actually OPTIMAL given tinygrad's current scheduler:
- 18 kernels/layer × 32 layers + 10 overhead
- Removing barriers → 3600+ kernels (scheduler can't handle large graphs)
- Adding barriers → 652+ kernels (fragments fusion opportunities)
- The current structure hits a local optimum

### What WOULD Work

1. **Metal Runtime Optimization** (per-kernel dispatch overhead)
   - Current: 34us/kernel average
   - Target: 24us/kernel
   - How: Better ICB batching, reduce barrier overhead, optimize Metal command encoder

2. **Scheduler Fusion Patterns** (cross-operation fusion)
   - Recognize: element-wise + Q4_0 matmul → inline
   - Recognize: matmul + softmax → online computation
   - Risk: Complex, could break other models

3. **Ushort Codegen** (Q4_0 bandwidth)
   - Generate `ushort` loads in Metal code (not tensor ops)
   - 103-147 GB/s → 200+ GB/s
   - Cuts compute from 7ms → 3.5ms
   - But dispatch still 20ms, so total only 23.5ms = 42 tok/s
   - **Not sufficient alone!**

### The Math

Even with ALL improvements:
- Ushort: 7ms → 3.5ms compute
- Scheduler fusion: 586 → 450 kernels
- Metal dispatch: 34us → 24us
- Total: 450 × 0.024 + 3.5 = **14.3ms = 70 tok/s ✓**

This requires changes across 3 subsystems (codegen, scheduler, Metal runtime).

## Recommendation

**Focus on Metal dispatch optimization first** (biggest bang for buck):
- 586 kernels × (34 → 24us) = 14ms immediately
- No scheduler risk
- Benefits ALL models, not just MLA

Then add ushort codegen for headroom (70 → 90+ tok/s).
# Q4_0 Optimization — youtu-llm:2b-Q4_0

## Current State: 56 tok/s (Feb 8, 2026)
- Baseline tensor path: 54 tok/s
- GROUPTOP=32 tuning: 56 tok/s (+4%)
- Theoretical limit: 145 tok/s (0.69GB / 100GB/s bandwidth)
- llama.cpp reference: ~70-80 tok/s

**Bottleneck**: 586 kernels/token × ~20us dispatch overhead = ~12ms. Weight reads at 100GB/s = 7ms. Total ≈ 19ms/token.

## Root Cause Analysis

### What Tinygrad Generates
Looking at the fused Q4_0 dequant+matmul Metal kernel (from VIZ=-1 + DEBUG=5):

```metal
// Line 8319 from r_64_32_3_64_16_64_16 kernel
unsigned char val7 = (*(data1_1176501056+(alu6+235920194)));  // READ 1 BYTE
// Extract nibbles
half lo = (half)((val7 & 15u)) - 8.0f;  // bits [0:4]
half hi = (half)((val7 >> 4u)) - 8.0f;  // bits [4:8]
```

**Pattern**: Read 1 byte → extract 2 nibbles → 2 multiply-adds.

**Bandwidth**: The Q4_0 kernels achieve 103-147 GB/s, which is 45-64% of the 229 GB/s MSL reference kernel.

### What llama.cpp Does
From `metal_q4_0.py` MSL reference kernel:

```metal
device const ushort *qs = (device const ushort *)(row + b * 18 + 2);  // READ 2 BYTES
ushort packed = qs[j];  // Read 2 bytes = 4 nibbles at once
float n0 = float(packed & 0xFu) - 8.0f;          // bits [0:4]
float n1 = float((packed >> 4) & 0xFu) - 8.0f;  // bits [4:8]
float n2 = float((packed >> 8) & 0xFu) - 8.0f;  // bits [8:12]
float n3 = float(packed >> 12) - 8.0f;           // bits [12:16]
```

**Pattern**: Read 2 bytes (ushort) → extract 4 nibbles → 4 multiply-adds.

**Result**: 229 GB/s peak (in isolation, GPU warm).

### Why Tinygrad Uses Byte Reads

The Q4_0 tensor path (quantized.py:79-87):
```python
packed = blocks[:, :, 2:]  # (O, bpr, 16) uint8 tensor
lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
```

Since `packed` is a **uint8 tensor**, the codegen generates byte-level loads. The UOp graph sees operations on uint8 elements, so it emits `unsigned char` reads.

To get ushort reads, the UOp graph would need to see operations on **uint16 elements**.

## Solution Path 1: Reduce Kernel Count (Scheduler-Level)

**Goal**: 586 kernels → ~300 kernels = 12ms dispatch → 6ms dispatch. Total: 7ms + 6ms = 13ms = 77 tok/s.

**Approach**: Improve fusion in the scheduler so Q4_0 dequant+matmul+residual+norm fuse into larger kernels.

**Why it's hard**:
- The scheduler already does a good job (103-147 GB/s for Q4_0 kernels)
- Further fusion requires understanding complex dataflow patterns
- Risk of breaking other models

**Where to look**:
- `tinygrad/engine/schedule.py` — schedule_uop converts tensor UOps to kernel UOps
- `tinygrad/schedule/` — the scheduler that decides kernel boundaries
- Pattern: contiguous() barriers, realize() barriers break fusion

**Estimated effort**: 3-5 days of deep scheduler debugging + risk of regressions.

**Expected gain**: 13ms → 13ms (no change unless kernel count drops significantly). Reducing from 586 to 300 kernels would give 77 tok/s.

## Solution Path 2: Generate Ushort Reads (Codegen-Level) ⭐ RECOMMENDED

**Goal**: Make tinygrad generate `ushort` reads instead of `uchar` reads for Q4_0 packed data.

**Result**: Individual Q4_0 kernels go from 103-147 GB/s → 200+ GB/s. With 586 kernels still: weight reads 7ms → 3.5ms. Total: 3.5ms + 12ms = 15.5ms = 65 tok/s.

### Implementation Approach

The fix needs to happen where the Q4_0 tensor operations are defined. Currently:

```python
# quantized.py:79-87 (current)
packed = blocks[:, :, 2:]  # (O, bpr, 16) uint8
lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
```

**Option 2A: Bitcast to ushort in tensor ops** (attempted, failed — shape mismatch issues)
```python
packed_u16 = blocks[:, :, 2:].bitcast(dtypes.ushort)  # (O, bpr, 8)
# Need to carefully reshape x to match the 4-nibbles-per-ushort access pattern
```

**Issue**: The nibble access pattern from ushorts doesn't map cleanly to the current `x_pairs` reshape. The MSL kernel uses indices: `x[base + 2*j]`, `x[base + 2*j + 16]`, `x[base + 2*j + 1]`, `x[base + 2*j + 17]`. Expressing this in tensor ops is awkward.

**Option 2B: Codegen pattern recognition** ⭐
Add a pattern matcher in the Metal codegen that recognizes:
```
LOAD(uint8) → BITWISE_AND/RSHIFT → CAST → arithmetic
```

And transforms it to:
```
LOAD(ushort) → extract 4 nibbles → arithmetic
```

**Where to implement**:
- `tinygrad/codegen/kernel.py` — the Kernel class that generates code
- `tinygrad/renderer/metal.py` — Metal-specific code generation
- Add a rewrite pattern that recognizes byte-level bitwise ops on quantized weights

**Steps**:
1. In Metal codegen, detect when a LOAD reads from a uint8 buffer with Q4_0 pattern (bitwise_and, rshift)
2. Check if 2 consecutive byte reads can be combined into 1 ushort read
3. Emit `ushort` load instead of 2 `uchar` loads
4. Emit the 4-nibble extraction inline

**Pseudo-code**:
```python
# In metal codegen, when we see:
#   byte0 = load(addr)
#   byte1 = load(addr+1)
#   lo0 = (byte0 & 0xF) - 8
#   hi0 = (byte0 >> 4) - 8
#   lo1 = (byte1 & 0xF) - 8
#   hi1 = (byte1 >> 4) - 8
#
# Replace with:
#   ushort packed = *(ushort*)(addr)
#   n0 = (packed & 0xF) - 8
#   n1 = ((packed >> 4) & 0xF) - 8
#   n2 = ((packed >> 8) & 0xF) - 8
#   n3 = (packed >> 12) - 8
```

**Estimated effort**: 1-2 days to implement the pattern matcher + testing.

**Expected gain**: 147 GB/s → 200+ GB/s per kernel. Total model: 54 tok/s → 65+ tok/s.

**Risk**: Low — only affects Metal backend, Q4_0-specific pattern. Easy to guard behind env var.

### Option 2C: Custom UOp kernel (like QL_CUSTOM)

Similar to the existing `custom_q4_0_linear` in quantized.py:7-30, but improved:
1. Use `UOp.range()` instead of `UOp.special()` so BEAM can optimize
2. Read as ushort: create a UOp that loads uint16 and does the 4-nibble extraction
3. Let the optimizer apply GROUPTOP, LOCAL, UPCAST

**Issue**: The UOp DSL may not have primitives for ushort reads with the required access pattern. And previous attempts with UOp DSL only achieved 24 GB/s.

**Verdict**: Not recommended unless we can prove the UOp DSL can express efficient ushort reads.

## Current Wins

1. **GROUPTOP=32** (heuristic.py:93): Try size=32 before size=16 for Metal SIMD groups.
   - Result: 54 → 56 tok/s (+4%)
   - Change: `for axis, sz in itertools.product((0, 1, 2), (32, 16)):`

## Implementation Plan: Ushort Reads (Option 2B)

**Status**: IN PROGRESS

### Architecture Understanding

Tinygrad's rendering pipeline:
```
UOps (abstract IR)
  → graph_rewrite(PatternMatcher) - UOp graph transformations
  → Renderer._render() - string generation
    → string_rewrite.rewrite(uop) - device-specific patterns
  → Device.compile() - to binary
```

**Key insight**: Pattern matching happens at TWO levels:
1. **UOp graph level** (before rendering): Simplifies abstract operations
2. **String rendering level** (during codegen): Device-specific code generation

For ushort optimization, we need **string-level patterns** in `MetalRenderer.string_rewrite`.

### Critical Files

**Primary (to modify)**:
- `/Users/rvd/src/rvd/tinygrad/tinygrad/renderer/cstyle.py` (lines 367-369)
  - `MetalRenderer.string_rewrite` — add Q4_0 ushort load pattern

**Reference**:
- `/Users/rvd/src/rvd/tinygrad/tinygrad/uop/ops.py` — UPat, PatternMatcher API
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/metal_q4_0.py` — llama.cpp reference kernel

### Implementation Strategy

**Challenge**: At rendering time, each UOp is processed independently. Can't easily detect "2 consecutive byte loads" without state tracking.

**Solution**: Add stateful tracking in MetalRenderer to:
1. Detect uint8 buffers from Q4_0 quantized weights
2. When rendering LOAD(uint8), check if it's part of Q4_0 packed data
3. Transform byte loads to ushort loads (reading 2 bytes at once)
4. Transform nibble extraction (BITWISE_AND, RSHIFT) to use ushort values

**Pattern to add** (in MetalRenderer.string_rewrite, BEFORE generic LOAD pattern):

```python
# Detect Q4_0 buffer characteristics:
# - dtype == uint8
# - Used in bitwise_and(0xF) and rshift(4) operations
# - From quantized weight blocks (metadata check)

# Transform:
#   uchar val = *(data + byte_offset);
#   lo = (val & 0xF) - 8;
#   hi = (val >> 4) - 8;
#
# Into:
#   ushort packed = *((device ushort*)(data + byte_offset));
#   n0 = (packed & 0xF) - 8;
#   n1 = ((packed >> 4) & 0xF) - 8;
#   n2 = ((packed >> 8) & 0xF) - 8;
#   n3 = (packed >> 12) - 8;
```

### Implementation Steps

**Phase 1: Detection** ✓
- Understand current Q4_0 Metal kernel structure (from DEBUG=5)
- Identify UOp patterns for byte loads + nibble extraction
- Confirmed pattern: `LOAD(uint8) → BITWISE_AND(0xF) / RSHIFT(4)`

**Phase 2: Pattern Matcher Design** (NEXT)
- Add method to MetalRenderer to detect Q4_0 buffers
- Design UPat patterns for byte load sequences
- Handle state tracking for paired byte loads

**Phase 3: Code Generation**
- Implement ushort load emission
- Handle 4-nibble extraction (shifts: 0, 4, 8, 12)
- Ensure 2-byte alignment (Q4_0 blocks: 2-byte scale + 16-byte packed)

**Phase 4: Testing**
- Correctness: Compare outputs with baseline
- Performance: Measure tok/s (target: 56 → 65+)
- Bandwidth: Verify Q4_0 kernels reach 180+ GB/s
- Inspect generated Metal source (DEBUG=5)

**Phase 5: Refinement**
- Handle edge cases (alignment, complex INDEX patterns)
- Gate behind env var: `Q4_0_USHORT=1`
- Test other quantized models for regressions

### Expected Outcome

**Success metrics**:
1. Identical model outputs (same token IDs)
2. 56 → 65+ tok/s (15%+ improvement)
3. Q4_0 kernel bandwidth: 103-147 GB/s → 180-200 GB/s
4. Generated Metal uses `ushort` loads instead of `uchar`

**Risk mitigation**:
- Metal-specific, isolated to one backend
- Gated behind environment variable
- Easy to revert if issues arise
- No UOp graph changes needed

### Testing Commands

```bash
# Baseline (current byte loads)
.venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4_0" \
  --benchmark 20 > baseline_bench.log

# With ushort optimization
Q4_0_USHORT=1 .venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4_0" \
  --benchmark 20 > ushort_bench.log

# Verify Metal source
DEBUG=5 Q4_0_USHORT=1 .venv2/bin/python tinygrad/apps/llm.py \
  --model "youtu-llm:2b-Q4_0" --benchmark 3 > ushort_debug5.log 2>&1
grep "device ushort" ushort_debug5.log  # Should see ushort loads
```

## Next Steps

Currently implementing **Phase 2** (Pattern Matcher Design) of Option 2B.

## Appendix: Profiling Data

From `profile_model.py youtu-llm:2b-Q4_0 10`:

```
--- Performance (10 tokens) ---
  Token          ms    tok/s  param GB/s
  5            21.6    46.32        32.1
  6-10         20.5    48.92        33.9  (steady-state)

--- Kernels: 586 per token, 5 ICBs ---
  Avg overhead per kernel: 34.9 us

--- Top Q4_0 kernels (with bitwise ops) ---
  r_64_32_3_64_16_64_16   549us   117 GB/s   (Q4_0 dequant+matmul+SiLU fused)
  r_2048_16_4_16          147 GB/s           (Q4_0 dequant+matmul)
  r_288_2_16_4_4_8         83 GB/s           (Q4_0 attn_kv_a 576×2048)
  r_128_16_4_4_32n1        65 GB/s           (Q4_0 attn_out 2048×2048)
```

**Observation**: Q4_0 kernels already achieve 65-147 GB/s (45-64% of theoretical). The gap to 229 GB/s is the byte→ushort optimization.

## Cat Elimination Failed (Feb 8 Evening)

Attempted to replace .cat() with sum-of-matmuls:
- Isolated test: 4.6x faster (53ms vs 245ms, same 9 kernels)
- Full model: **TIMEOUT** - split cache creates scheduling explosion
- Reverted. See cat_elimination_failed.md for analysis.

Conclusion: Local optimizations don't translate. Need scheduler-level changes.

## Split Cache Attempt - Measured (Feb 8)

**Change**: Split KV cache to enable sum-of-matmuls (avoid cat)

**Result**:
- 652 kernels vs 586 baseline (+66, +11%)
- 55.5 tok/s vs 56 tok/s (slightly slower)
- 137 schedule calls vs 73 baseline

**Why more kernels**: Two .realize() barriers per layer (cache_k_nope + cache_k_pe) fragments the 18-kernel scheduling region into smaller chunks (17+9+3 pattern). The extra barriers prevent cross-operation fusion that the single-cache approach enables.

**Isolated test showed 4.6x speedup, but full model adds +66 kernels overhead**. The cat operations in the original code allow better fusion ACROSS the cat boundary than split barriers allow.

**Conclusion**: The scheduler fuses THROUGH cat operations better than it handles split barriers. Reverted.
# MLA Optimization Sprint: youtu-llm 2b Q4_0 → 70 tok/s (Feb 8, 2026)

## Current Performance
- **56 tok/s** (17.8ms/token steady-state)
- **Target**: 70 tok/s (llama.cpp parity)
- **Gap**: 3.5ms/token = 25% speedup needed
- **586 kernels/token** (18.3/layer × 32 layers)
- **34.4us avg overhead/kernel** = 20.1ms dispatch overhead
- **Bottleneck**: Kernel count, not bandwidth (Q4_0 kernels achieve 103-147 GB/s)

---

## Per-Layer Kernel Breakdown (18 kernels from DEBUG=2)

### ATTENTION (11-12 kernels)
1. `r_32_64n1` - attn_norm RMSNorm reduce (~14us)
2. `r_1536_32_2_16n1` - q_a Q4_0 matmul (2048→1536, 258 GB/s)
3. `r_32_48` - q_a_norm reduce (~12us)
4. `r_32_32_3_48_16n1` - q_b Q4_0 matmul (1536→3072, 66 GB/s)
5. `r_288_2_16_4_4_8` - kv_a Q4_0 matmul (2048→576+288, 83 GB/s)
6. `r_(start_pos+1)_16_4_4_36` - QK matmul + scale (~21us)
7. `r_4_4_(start_pos+1)` - softmax sum (~9us)
8. `r_8_4_16_4_4_(start_pos+1)` - softmax div + attn@V (~13us)
9. `r_128_16_4_4_32n1` - absorbed V projection (V@v_b^T, 67 GB/s)
10. `r_2048_32_2_16n1` - attn_output Q4_0 matmul + residual (303 GB/s)
11. `r_32_64` - post-attn RMSNorm reduce

### FFN DENSE (6-7 kernels)
12. `r_64_32_3_64_16_64_16n1` - gate+up Q4_0 fused + silu + mul (545us, 120 GB/s)
13. `r_2048_32_6_16n1` - down Q4_0 matmul + residual (132 GB/s)
14. Plus RMSNorm (2 kernels) and misc elementwise

### OVERHEAD
15-18. Contiguous copies, cache ops, RoPE element-wise

---

## Theoretical Minimum (10-11 kernels/layer)

### ATTENTION (7 kernels ideal)
1. attn_norm reduce
2. q_a matmul
3. **q_a_norm + q_b matmul** ← FUSE (save 1 kernel)
4. kv_a matmul + kv_a_norm ← FUSE (save 1 kernel)
5. **QK + softmax** ← FUSE online (save 2 kernels)
6. attn@V + V absorption ← FUSE (save 1 kernel)
7. attn_output + residual

### FFN (3-4 kernels ideal)
8. ffn_norm reduce
9. gate+up+silu+mul (already well-fused)
10. down+residual

**Gap**: 18 → 10-11 = **7-8 kernels/layer × 32 = 224-256 kernels saved**
At 34us/kernel = 7.6-8.7ms saved → **~9-10ms/token = 100+ tok/s**

But this is theoretical. Actual achievable depends on scheduler capabilities.

---

## Experiments Conducted

### 1. Softmax fp16 (NEUTRAL)
**Change**: Use `qk.softmax(-1)` instead of manual `.float().exp().sum().div().half()`
**Result**: 618 kernels vs 586 baseline (+32 from `.max()` operation), same speed
**Conclusion**: Reverted. The `.max()` adds kernels without speedup benefit.

### 2. Cat Elimination (FAILED)
**Change**: Split KV cache into `cache_k_nope` and `cache_k_pe`, decompose QK matmul into sum
```python
# Before: q.cat(q_pe) @ k.cat(k_pe).T
# After: q@k.T + q_pe@k_pe.T
```

**Isolated test** (`test_mla_fusion.py`):
- Cat: 9 kernels, 245ms
- Sum: 9 kernels, 53ms
- **4.6x speedup**!

**Full model**:
- **652 kernels** vs 586 baseline (+66 kernels, +11%)
- **55.5 tok/s** vs 56 tok/s baseline (slightly slower)
- 137 schedule calls during warmup (vs ~73 baseline)

**Why it failed**:
- Split cache creates 2 `.realize()` barriers per layer (vs 1)
- Doubles cache assignment operations: 64 barriers instead of 32
- More unique schedules = more compilation work = slower warmup
- The 4.6x isolated speedup is eaten by +66 kernels of scheduling overhead

**Conclusion**: Reverted. The `.cat()` overhead (seen in isolation) is masked by good fusion in the full model. Split cache adds net overhead (652 vs 586 kernels).

### 3. Profiling Deep Dive
Created `profile_model.py` wrapper that shows:
- Kernel categories: Q4_0 matmuls (237/tok, 83%), elementwise (137, 12.5%), reductions (213, 4.6%)
- Per-kernel overhead: 34.4us average
- Scheduling patterns: 18-kernel blocks repeat 32 times

---

## Why Simple Fixes Don't Work

### Can't Remove `.contiguous()`
- Current with barriers: 586 kernels, 56 tok/s
- Without barriers: 3624 kernels, 3.7 tok/s (15x slower!)
- Scheduler partitions large cross-block graphs poorly

### Can't Eliminate Cat Operations
- Cat serves as controlled fusion point
- Removing it requires split cache → 2× barriers → scheduling explosion
- Sum-of-matmuls is faster in isolation but breaks JIT

### Can't Inline RMSNorm into Q4_0 Matmul
- RMSNorm outputs fp16
- Q4_0 matmul reads quantized weights
- Scheduler doesn't recognize element-wise → quantized matmul as fusable pattern

### Can't Use Ushort at Tensor Level
- Tensor ops with `bitcast(ushort)` create wrong nibble access patterns
- Slower (4.0ms vs 2.3ms) AND incorrect (produces NaN/inf)
- Must be fixed at codegen/renderer level, not tensor expression level

---

## Viable Paths Forward

### Path A: Scheduler Fusion Patterns (Estimated 52-60 tok/s)

**Goal**: Teach scheduler to recognize and fuse:
1. **Element-wise → Q4_0 matmul** (RMSNorm scale into matmul input)
   - Current: norm reduce (1k) + norm scale (1k) + matmul (1k) = 3 kernels
   - Optimal: norm reduce + (scale+matmul fused) = 2 kernels
   - Savings: 4 occurrences/layer × 32 = 128 kernels

2. **Matmul → online softmax** (QK + max/exp/sum/div in one kernel)
   - Current: QK matmul (1k) + exp (1k) + sum (1k) + div (1k) = 4 kernels
   - Optimal: QK matmul + online softmax = 1-2 kernels
   - Savings: 2-3 kernels/layer × 32 = 64-96 kernels

**Total savings**: 192-224 kernels = 6.6-7.6ms @ 34us/kernel
**New perf**: 10.2-11.2ms = **89-98 tok/s**

**Implementation**:
- Add patterns to `tinygrad/schedule/` or `tinygrad/engine/schedule.py`
- Extend `found_contiguous()` optimization to recognize more fusable patterns
- Risky: Could break other models

### Path B: Ushort Codegen (Estimated 65 tok/s)

**Goal**: Generate `ushort` loads instead of `uchar` for Q4_0 packed nibbles at Metal codegen level.

**Current Metal pattern**:
```metal
unsigned char val = *(data + byte_offset);
half lo = (half)(val & 15u) - 8.0f;
half hi = (half)(val >> 4u) - 8.0f;
```

**Target Metal pattern** (llama.cpp):
```metal
ushort packed = *((device ushort*)(data + byte_offset));
float n0 = (float)(packed & 0xFu) - 8.0f;
float n1 = (float)((packed >> 4) & 0xFu) - 8.0f;
float n2 = (float)((packed >> 8) & 0xFu) - 8.0f;
float n3 = (float)(packed >> 12) - 8.0f;
```

**Expected**: 103-147 GB/s → 200-229 GB/s (llama.cpp reference)
**Result**: 7ms compute → 3.5ms compute (same 586 kernels)
**New perf**: 3.5ms + 20.1ms = 23.6ms = **42 tok/s**

Wait, that's WORSE than current! Let me recalculate...

Actually: If we improve bandwidth 2x, the compute portion (7ms) becomes 3.5ms, but dispatch stays 20ms. New total: 23.5ms = 42 tok/s. **This doesn't help enough alone.**

### Path C: Both Scheduler + Ushort (70+ tok/s)

Combine approaches:
- Scheduler fusion: 586 → 362 kernels (save 224)
- Ushort reads: 100 GB/s → 200 GB/s (2x)

**Compute**: 7ms / 2 = 3.5ms (weight reads)
**Dispatch**: 362 kernels × 34us = 12.3ms
**Total**: 15.8ms = **63 tok/s**

Still short! Need BOTH plus lower per-kernel overhead:
- Better Metal dispatch: 34us → 20us per kernel
- 362 kernels × 20us + 3.5ms = 10.7ms = **93 tok/s**

---

## Critical Constraints

### JIT Cache Stability
Removing `.contiguous()` or changing cache structure breaks JIT:
- Different context lengths create different graph shapes
- Scheduler produces different kernel counts (548 vs 586)
- JIT cache misses on every token

**Solution needed**: Make schedule cache robust to variable-length patterns OR ensure all fusions are stable regardless of symbolic vars.

### Scheduling Complexity
The full 32-layer graph is too complex for scheduler:
- Without barriers: 420-548 kernels (unpredictable)
- With barriers: 586 kernels (stable but suboptimal)

**The raw `.schedule()` (no JIT, no barriers) produces just 15 kernels!**
But this is not achievable with autoregressive decode (need cache writes between layers).

### Quantization Prevents Fusion
Q4_0 dequant inline prevents standard fusion patterns:
- Element-wise ops (RMSNorm scale) can't inline into Q4_0 matmul reads
- Matmul output can't fuse with Q4_0 matmul input (different data types)
- The MATVEC heuristic requires `MUL(INDEX, INDEX)` but Q4_0 creates `MUL(dequant_chain(INDEX), INDEX)`

---

## Isolated Operation Test Results

Created `test_mla_fusion.py` for incremental testing:

| Operation | Kernels | Time (warmup) | Bandwidth | Notes |
|-----------|---------|---------------|-----------|-------|
| RMSNorm | 5 | 450ms | N/A | Reduce + elementwise |
| Q4_0 matmul (2048→1536) | 3 | 80ms | ~180 GB/s | Well-fused dequant+matmul |
| Cat method QK | 9 | 245ms | Poor | `.cat()` forces expensive copies |
| Sum method QK | 9 | 53ms | Good | **4.6x faster, same kernels!** |

**Key finding**: Cat overhead is in data movement, not kernel count. But can't eliminate in full model without breaking JIT.

---

## What We Know Doesn't Work

1. ❌ **Removing `.contiguous()`**: 586 → 3624 kernels, 15x slower
2. ❌ **Packed-dot QuantizedLinear**: +142 kernels, slower (fragments too much)
3. ❌ **Split KV cache**: Timeout, scheduling explosion
4. ❌ **Ushort tensor ops**: Slower + incorrect (nibble reordering issues)
5. ❌ **Deferred cache + no contiguous**: 63s to schedule, no benefit
6. ❌ **FP16 softmax with `.max()`**: +32 kernels, same speed

---

## The Real Problem

**Kernel count is too high (586), but reducing it breaks things**:
- Remove barriers → scheduler produces 420-3624 kernels (unpredictable)
- Keep barriers → stable 586 kernels but can't reach 70 tok/s
- Theoretical raw graph schedules to 15 kernels (not achievable with decode)

**To reach 70 tok/s** (14.3ms/token) from 56 tok/s (17.8ms):
- Need to save 3.5ms
- At 34us/kernel: need to eliminate ~103 kernels (586 → 483)
- OR: cut per-kernel overhead 34us → 20us (586 kernels × 20us = 11.7ms + 7ms = 18.7ms = 53 tok/s) — still short!

**The math**: Even with BOTH improvements (483 kernels @ 20us + 3.5ms ushort):
- 483 × 0.020 = 9.7ms dispatch
- 3.5ms compute (2x bandwidth from ushort)
- Total: 13.2ms = **76 tok/s**

This is achievable but requires:
1. Scheduler fusion patterns (save 100 kernels)
2. Ushort codegen (2x Q4_0 bandwidth)
3. Better Metal dispatch (34us → 20us per kernel)

---

## Next Steps (Ranked by Feasibility)

### 1. Use GlobalCounters for Precise Profiling
User suggestion: Use GlobalCounters to track exact kernel counts per operation.

**Action**: Add GlobalCounters to test_mla_fusion.py to see:
- Which operations generate which kernels
- Where fusion actually happens vs where it doesn't
- Precise before/after counts for changes

### 2. Ushort Codegen Pattern (2-3 days, 65 tok/s potential)
Implement at `MetalRenderer.string_rewrite` level (cstyle.py):
- Detect Q4_0 `LOAD(uint8) → BITWISE_AND/RSHIFT` pattern
- Transform to `LOAD(ushort) → extract 4 nibbles`
- Test with isolated Q4_0 matmul first (in test_mla_fusion.py)

**Expected**: Q4_0 kernels 103-147 GB/s → 200+ GB/s

### 3. Scheduler Fusion Patterns (4-5 days, high risk)
Add patterns to recognize:
- Element-wise post-reduce → matmul input (for RMSNorm+Q4_0)
- Matmul output → reduction (for online softmax)

**Where**: `tinygrad/schedule/rangeify.py` or `tinygrad/engine/schedule.py`
**Risk**: Could break other models, requires deep scheduler understanding

---

## Tools Created

1. **`profile_model.py`** - One-command profiling with DEBUG=2 + PROFILE=1 parsing
2. **`test_mla_fusion.py`** - Iterative operation testing framework
3. **Various .md docs** - Analysis and failed attempt documentation

---

## Key Insights

### 1. Isolated Tests Are Misleading
Sum-of-matmuls was 4.6x faster in isolation but **timed out** in full model. The cache assignment patterns and JIT compilation change the dynamics completely.

### 2. Cat Operations Are Structural
`.cat()` isn't just an operation - it's a **scheduling control mechanism**. Removing it makes the graph too complex for the scheduler to handle efficiently.

### 3. Barriers Are Necessary Evils
The `.contiguous()` and `.realize()` calls create scheduling barriers that:
- Limit fusion opportunities (bad)
- Keep scheduling tractable (essential)
- Enable stable JIT cache hits (critical)

Without them, the scheduler either:
- Takes 60+ seconds to schedule (unusable)
- Produces 3600+ kernels (15x slower)

### 4. The Scheduler Can't See Through Quantization
Q4_0 packed-dot operations create complex UOp patterns:
- `MUL(dequant_chain(INDEX), INDEX)` instead of `MUL(INDEX, INDEX)`
- Breaks MATVEC heuristic pattern matching
- Prevents fusion with surrounding element-wise ops

### 5. Dispatch Overhead Dominates
Even with perfect bandwidth (200+ GB/s):
- 586 kernels × 34us = 20ms (just dispatch)
- Compute is only 7ms (weight reads @ 100 GB/s)
- **Can't reach 70 tok/s without reducing kernel count OR dispatch overhead**

---

## Conclusion

To reach 70 tok/s, we need **scheduler-level changes** or **Metal dispatch optimization**, not model code tweaks.

The most promising path:
1. **Add ushort codegen pattern** (Metal renderer level) → 65 tok/s
2. **Reduce per-kernel overhead** (Metal runtime improvements) → 70+ tok/s

Scheduler fusion alone maxes out at ~52-60 tok/s due to remaining dispatch overhead.

---

## References

- `mla.md` - MLA architecture and cache management analysis
- `speed.md` - Historical optimization attempts and bottleneck analysis
- `experiments.md` - Ongoing experiment log
- `test_mla_fusion.py` - Isolated operation testing
- `profile_model.py` - One-command profiling tool

---

## CRITICAL INSIGHT: Scheduler Fuses Through Cat (Feb 8 Evening)

### The Discovery

Split cache attempt showed:
- Isolated test: sum-of-matmuls 4.6x faster than cat (53ms vs 245ms)
- Full model: 652 kernels vs 586 baseline (+66), 55.5 tok/s vs 56 tok/s (slower!)

**Why**: The `.cat()` operations DON'T create fusion barriers - the scheduler optimizes through them!
The split cache creates REAL barriers (`.realize()` calls) that PREVENT fusion.

### What Actually Happens

**With cat (current)**:
```python
q = q_nope.cat(q_pe)  # NOT a barrier - scheduler sees through this
k = k_nope.cat(k_pe)
cache.assign(k).realize()  # ONE barrier per layer
# Result: 18-kernel schedule that fuses across cat boundaries
```

**With split cache**:
```python
cache_nope.assign(kv).realize()  # Barrier 1
cache_pe.assign(k_pe).realize()  # Barrier 2
# Result: 17+9+3 kernel pattern, fragments the fusion
```

The cat is just a tensor operation that gets optimized during scheduling. The `.realize()` is an actual execution barrier that forces materialization.

### Implication

**We've been optimizing the wrong thing**. The cat operations are cheap in the full model because:
1. Scheduler sees the full dataflow graph
2. Fusions happen across cat boundaries
3. The single `.realize()` barrier per layer keeps scheduling regions large enough for good fusion

Splitting the cache to "avoid cat" actually **creates barriers where there were none**, fragmenting the fusion opportunities.

### The Real Path Forward

Instead of eliminating cat, we need to:
1. **Let the scheduler fuse more aggressively** within the existing 18-kernel regions
2. **Reduce per-kernel dispatch overhead** (34us → 20us)
3. **Improve Q4_0 kernel bandwidth** (ushort reads at codegen level)

The 18-kernel-per-layer structure is actually good - it's stable, predictable, and allows reasonable fusion. The problem is:
- 18 kernels × 32 layers = 576 kernels (close to current 586)
- At 34us overhead = 19.6ms
- Need to get to 14ms = either 412 kernels @ 34us OR 586 kernels @ 24us

