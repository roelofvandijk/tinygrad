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
