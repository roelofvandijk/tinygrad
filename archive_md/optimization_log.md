# GLM Optimization Log

## Baseline Performance
- **Current**: 20.0 tok/s (Q4_0) — **pure tinygrad DSL, no custom Metal/MSL kernels**
- **Commit**: 1f0804962 "comfortable 20.9 tok/s"
- **Key optimization already in place**: merge_gate_up for regular experts
- **Target**: 40 tok/s (llama.cpp does 35 tok/s in 10GB RAM)

## Fundamental Principle Discovered
**Merge operations that access the same indices/data to reduce memory bandwidth**

### Evidence from 14 tok/s → 20 tok/s jump:
- **14 tok/s**: Separate `ffn_gate_exps(sel, x)` and `ffn_up_exps(sel, x)` calls
  - 2 expert gathers (21% × 2 = 42% of time!)
  - 2 dequant operations
  - 2 matmuls
- **20 tok/s**: Merged `ffn_gate_up_exps(sel, x)` single call
  - 1 expert gather (21% of time)
  - 1 dequant
  - 1 matmul producing both gate and up
  - **Saved: 21% of per-token time**

## Optimization Experiments

### Exp 1: Remove contiguous() on line 141
**Status**: ❌ FAILED (measurement noise)
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > exp1a_no_contig.log 2>&1`
**Result**: 19.74 tok/s (baseline: 20.0 tok/s, diff: -1.3% = noise)
**Conclusion**: Removing contiguous() doesn't help. Scheduler needs it for correctness.
**Reverted**: Yes

---

### Exp 2: Merge RoPE for both q_pe and k_pe (both_fused candidate)
**Status**: ⏳ RUNNING
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > both_rope_merged.log 2>&1`

**Hypothesis**:
- Lines 103-104 call `_rope_interleaved()` twice with same start_pos, T
- Both load same cos/sin cache: `freqs_cos_cache[:, :, start_pos:start_pos+T]`
- Memory bandwidth waste - loading same data twice

**Implementation**:
- Created `_rope_interleaved_both(q_pe, k_pe, ...)` function
- Loads cos/sin cache ONCE
- Processes both q_pe and k_pe with same cache
- Mirrors merge_gate_up pattern

**Expected**: +5-15% speedup if memory bandwidth is limiting factor

**Code changes**:
```python
# OLD (2 cache loads):
q_pe = self._rope_interleaved(q_pe, start_pos, T)  # Load cos/sin
k_pe = self._rope_interleaved(k_pe, start_pos, T)  # Load cos/sin AGAIN

# NEW (1 cache load):
q_pe, k_pe = self._rope_interleaved_both(q_pe, k_pe, start_pos, T)  # Load ONCE
```

**Waiting for result...**

---

### Exp 3: Merge shared expert gate/up (PLANNED)
**Status**: 📋 NEXT

**Analysis**:
- Shared expert gate+silu*up = **22% of per-token time** (TOP BOTTLENECK!)
- Current code (line 145): Separate calls like 14 tok/s version:
  ```python
  self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm)
  ```
- Two separate Linear forward passes
- Two separate weight loads

**Plan**:
1. Create merge function similar to `merge_gate_up_experts` for shared expert
2. Concatenate ffn_gate_shexp and ffn_up_shexp weights at model load
3. Single forward pass, split output
4. Mirror regular expert optimization

**Expected**: +10-20% speedup (proportional to 22% bottleneck reduction)

**Implementation approach**:
- Check llm.py for where shared expert weights are loaded
- Add merge step after model initialization
- Update mla.py line 145 to use merged weights

---

## Results Summary

| Experiment | Status | Baseline | Result | Δ | Keep? | Notes |
|------------|--------|----------|--------|---|-------|-------|
| Baseline   | ✅     | 20.0     | -      | - | -     | merge_gate_up already in |
| Exp1: No contiguous | ❌ | 20.0 | 19.74 | -1.3% | No | Noise |
| Exp2: Merge RoPE | ⏳ | 20.0 | ? | ? | ? | Running now |
| Exp3: Merge shexp | 📋 | 20.0 | ? | ? | ? | Planned next |

---

## Timeline
- 2026-02-10 17:28: Started systematic optimization
- 2026-02-10 17:30: Exp1 failed (noise)
- 2026-02-10 17:31: Exp2 launched (merged RoPE)
- 2026-02-10 17:32: Planning Exp3 (merge shexp)

---

### Exp 2: Merge RoPE for both q_pe and k_pe
**Status**: ❌ FAILED (no improvement)
**Result**: 20.00 tok/s (baseline: 20.0 tok/s, diff: 0%)
**Conclusion**: RoPE cache loads are NOT a bottleneck. Data is small and already cached.
**Insight**: Merge principle only helps when the operation is bandwidth-limited, not for small cached data.
**Reverted**: Yes

---

### Exp 3: Merge shared expert gate/up
**Status**: ⏳ RUNNING  
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > merge_shexp.log 2>&1`

**Hypothesis**:
- Shared expert gate+silu*up = **22% of per-token time** (TOP BOTTLENECK!)
- Currently: Separate `ffn_gate_shexp(x)` and `ffn_up_shexp(x)` calls
- Two Linear forward passes, two weight matrix loads
- Same anti-pattern as 14 tok/s version

**Implementation**:
- Created `merge_gate_up_shared_expert()` function
- Concatenates gate and up weight matrices: (out,in) cat (out,in) -> (2*out,in)
- Updated forward pass to:
  1. Single merged call: `ffn_gate_up_shexp(h_norm)`
  2. Split output: `gate, up = output[..., :half], output[..., half:]`
  3. Apply activation: `gate.silu() * up`
  
**Expected**: +10-20% speedup (proportional to 22% bottleneck reduction)

**Code changes**:
- mla.py: Added `merge_gate_up_shared_expert()` function
- mla.py lines 145-150: Updated forward pass to use merged weights
- llm.py line 9: Import new function
- llm.py line 414: Call merge at model load time

**Waiting for result...**


**Result**: ❌ FAILED (no improvement)
- Benchmark: 20.33 tok/s (baseline: 20.0 tok/s, diff: +1.65% = noise)
- Kernel count: Still 1404 kernels/token (no reduction)

**Why it didn't work**:
- Scheduler was ALREADY fusing gate+silu*up into a single kernel!
- glm_kernel_map line 31 shows: `r_96_16_4_4_128_2048` labeled "shared expert gate+silu*up **fused** Q4_0"
- The scheduler fuses: Q4_0 dequant (gate) + silu + Q4_0 dequant (up) + multiply → ONE kernel
- Weight-level merge provides no additional benefit when scheduler already does kernel-level fusion

**Key Learning**:
**Scheduler fusion can be as effective as weight-level merging.** Manual merging only helps when:
1. Scheduler can't fuse the ops (e.g., different expert selection paths)
2. Merging reduces memory movement (e.g., expert gathers benefit from merged blocks)

**Reverted**: No (kept for code cleanliness, no harm)

---

### Exp 4: Remove .contiguous() calls in MoE path
**Status**: ❌ FAILED (major regression!)
**Command**: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > bench_no_contig.log 2>&1`

**Changes tested**:
- Line 141: `gated = gate.silu() * up` (removed .contiguous())
- Line 142: `expert_out = self.ffn_down_exps(sel, gated)` (removed .contiguous())

**Result**:
- **Baseline**: 20.33 tok/s (49.2ms, 1404 kernels)
- **After removal**: 16.82 tok/s (59.5ms, 1312 kernels)
- **Change**: -17% slower! (-3.51 tok/s)

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

