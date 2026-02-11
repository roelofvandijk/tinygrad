# GLM Optimization Experiment Log

## Current Status (Feb 10, 2026)
- **Baseline**: 20.0 tok/s (GLM-4.7-Flash Q4_0, pure tinygrad DSL)
- **Target**: 35 tok/s (llama.cpp reference in 10GB RAM)
- **Model**: 47 blocks (1 dense + 46 MoE), 64 experts, 4 sel/tok, kv_lora_rank=512

## Fundamental Optimization Principle

**Merge operations that access the same indices/data to reduce memory bandwidth.**

### Evidence: 14 tok/s → 20 tok/s Jump
- **14 tok/s**: Separate `ffn_gate_exps(sel, x)` and `ffn_up_exps(sel, x)` calls
  - 2 expert gathers (21% × 2 = 42% of time!)
  - 2 dequant operations, 2 matmuls
- **20 tok/s**: Merged `ffn_gate_up_exps(sel, x)` single call
  - 1 expert gather (21% of time)
  - 1 dequant, 1 matmul producing both gate and up
  - **Saved: 21% of per-token time** ✅

## Experiments Timeline

### Exp 1: Remove .contiguous() on Line 141
**Date**: Feb 10, 2026
**Status**: ❌ FAILED (noise)
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > exp1a_no_contig.log 2>&1`

**Hypothesis**: `.contiguous()` prevents scheduler fusion
**Result**: 19.74 tok/s (baseline: 20.0 tok/s, diff: -1.3% = noise)
**Conclusion**: Removing contiguous() doesn't help. Scheduler needs it for correctness.
**Reverted**: Yes

---

### Exp 2: Merge RoPE for q_pe and k_pe
**Date**: Feb 10, 2026
**Status**: ❌ FAILED (no improvement)
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > both_rope_merged.log 2>&1`

**Hypothesis**: Lines 103-104 call `_rope_interleaved()` twice with same start_pos, T. Memory bandwidth waste loading same cos/sin cache twice.

**Implementation**:
- Created `_rope_interleaved_both(q_pe, k_pe, ...)` function
- Loads cos/sin cache ONCE, processes both q_pe and k_pe

**Result**: 20.00 tok/s (baseline: 20.0 tok/s, diff: 0%)
**Conclusion**: RoPE cache loads are NOT a bottleneck. Data is small and already cached.
**Insight**: Merge principle only helps when the operation is bandwidth-limited, not for small cached data.
**Reverted**: Yes

---

### Exp 3: Merge Shared Expert gate/up
**Date**: Feb 10, 2026
**Status**: ❌ NO BENEFIT (scheduler already fuses)
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > merge_shexp.log 2>&1`

**Hypothesis**: Shared expert gate+silu*up = **22% of per-token time** (TOP BOTTLENECK!). Currently separate `ffn_gate_shexp(x)` and `ffn_up_shexp(x)` calls. Same anti-pattern as 14 tok/s version.

**Implementation**:
- Created `merge_gate_up_shared_expert()` function
- Concatenates gate and up weight matrices: (out,in) cat (out,in) -> (2*out,in)
- Single merged call, split output, apply activation

**Result**: 20.33 tok/s (baseline: 20.0 tok/s, diff: +1.65% = noise)
**Kernel count**: Still 1404 kernels/token (no reduction)

**Why it didn't work**:
- Scheduler was ALREADY fusing gate+silu*up into a single kernel!
- glm_kernel_map line 31: `r_96_16_4_4_128_2048` labeled "shared expert gate+silu*up **fused** Q4_0"
- The scheduler fuses: Q4_0 dequant (gate) + silu + Q4_0 dequant (up) + multiply → ONE kernel
- Weight-level merge provides no additional benefit when scheduler already does kernel-level fusion

**Key Learning**:
**Scheduler fusion can be as effective as weight-level merging.** Manual merging only helps when:
1. Scheduler can't fuse the ops (e.g., different expert selection paths)
2. Merging reduces memory movement (e.g., expert gathers benefit from merged blocks)

**Reverted**: No (kept for code cleanliness, no harm)

---

### Exp 4: Remove .contiguous() in MoE Path
**Date**: Feb 10, 2026
**Status**: ❌ MAJOR REGRESSION (-17%)
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > bench_no_contig.log 2>&1`

**Changes tested**:
- Line 141: `gated = gate.silu() * up` (removed .contiguous())
- Line 142: `expert_out = self.ffn_down_exps(sel, gated)` (removed .contiguous())

**Result**:
- **Baseline**: 20.33 tok/s (49.2ms, 1404 kernels)
- **After removal**: 16.82 tok/s (59.5ms, 1312 kernels)
- **Change**: **-17% slower!** (-3.51 tok/s)

**Why it failed**:
- Kernel count DECREASED (1404 → 1312, -6.6%)
- But per-kernel time INCREASED (35.0us → 45.3us avg, +29%)
- **.contiguous() is NOT always a fusion barrier!**
- It materializes tensors in optimal memory layout for next operation
- Without it, scheduler creates fewer but SLOWER kernels

**Key Learning**:
**.contiguous() can be NECESSARY for performance.** It ensures:
1. Memory layout matches next operation's access pattern
2. No strided/broadcasted views that slow down kernels
3. Expert gather/matmul patterns work efficiently

**Reverted**: Yes (immediately)

---

## Results Summary

| Experiment | Status | Baseline | Result | Δ | Keep? | Notes |
|------------|--------|----------|--------|---|-------|-------|
| Baseline (merge_gate_up) | ✅ | 20.0 | - | - | - | Already in master |
| Exp1: No contiguous L141 | ❌ | 20.0 | 19.74 | -1.3% | No | Noise |
| Exp2: Merge RoPE | ❌ | 20.0 | 20.00 | 0% | No | Not a bottleneck |
| Exp3: Merge shexp | ❌ | 20.0 | 20.33 | +1.65% | No | Scheduler already fuses |
| Exp4: No contiguous MoE | ❌ | 20.33 | 16.82 | **-17%** | No | Major regression |

---

## Key Insights

### 1. When Manual Merging Helps
**merge_gate_up_experts (regular experts)**: 14 → 20 tok/s (+43%)
- **Why it worked**: Reduced expert gathers from 2 to 1 (42% → 21% of time)
- Gathers can't be fused by scheduler (different expert selection paths)

### 2. When Scheduler Already Fuses
**merge_gate_up_shared_expert**: 20 → 20.33 tok/s (+1.65% = noise)
- **Why it didn't help**: Scheduler already fuses gate+silu*up into ONE kernel
- Weight-level merge provides no additional benefit

### 3. .contiguous() Is NOT Always a Fusion Barrier!
**Tested removal from MoE path**: -17% REGRESSION
- **.contiguous() materializes tensors in optimal memory layout**
- Without it: fewer kernels but SLOWER due to strided/broadcasted access patterns
- Expert gathers and matmuls need contiguous layout for efficient memory access

**When .contiguous() is necessary**:
1. Before expert gathers/matmuls (ensures coalesced memory access)
2. After operations that create strided views (split, reshape, transpose)
3. When next operation expects specific memory layout

**When to remove .contiguous()**:
- Only when profiling shows it's redundant (scheduler already materializes)
- Never remove speculatively - always benchmark!

---

## Decision Rules

### Use Manual Merging When
1. Operations can't be scheduler-fused (e.g., expert gathers with fancy indexing)
2. Merging reduces memory movement (e.g., concatenated blocks = fewer loads)

### Trust Scheduler Fusion When
1. Operations are in same execution path (no branching)
2. Data types/shapes are compatible
3. No explicit fusion barriers (.contiguous(), .realize())

---

## Split Shared Expert Results (Historical)

### Q4_0 Model
- **Baseline**: 17.40 tok/s
- **With split_shexp**: 19.52 tok/s
- **Speedup**: +12.2% ✅

### Q4_K_M Model
- **Baseline**: 9.57 tok/s
- **With split_shexp**: 9.60 tok/s
- **Speedup**: +0.3% (within noise)

**Conclusion**: split_shexp works for Q4_0 quantization (inline dequant kernels where fusion helps) but no benefit for Q4_K_M (custom mul_mat_id kernels already optimized).

---

## GROUPTOP Threshold Fix (Historical)

**File**: `tinygrad/codegen/opt/heuristic.py:113`
**Change**: Increased GROUPTOP threshold from `2048` to `65536`

**Problem**: MoE expert kernels have output shape product > 2048 (e.g., 6 experts × large output dim). Old threshold blocked GROUPTOP, resulting in 3 GB/s bandwidth with poor cooperative reduction.

**Results** (Token 5 steady state):
- **10.05 tok/s** (vs baseline ~0.35 tok/s)
- **17 GB/s bandwidth** (vs 3 GB/s before)
- **~28x speedup!**

**Why This Works**: GROUPTOP enables cooperative reduction across threadgroup - threads split work and cooperate using shared memory, improving memory coalescing and cache utilization.

---

## Next Steps

Based on failed experiments, the path forward is NOT:
- ❌ Removing .contiguous() calls speculatively
- ❌ Merging operations that scheduler already fuses
- ❌ Optimizing non-bottleneck operations (RoPE cache)

Instead, focus on:
1. **Expert gathers** (21% of time) - pure data copying waste
2. **Expert Q4_0 matmul** (16% of time) - low bandwidth (51 GB/s)
3. **Attention path optimizations** - the mysterious "both_fused" (+23%)
4. **Heuristic improvements** - MV pattern matching for fused dequant
5. **E-graph optimization** - automatic kernel fusion discovery
