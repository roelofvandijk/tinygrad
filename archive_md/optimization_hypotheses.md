# Optimization Hypotheses

## 1. both_fused (+23% speedup) - HIGHEST PRIORITY

### Evidence
- bench_both_fused.log: 19.43 → 23.95 tok/s (+23%)
- Most significant single optimization

### Hypothesis A: Fuse q_nope matmul + q_pe RoPE + concatenation
**Location**: mla.py lines 98-106

**Current code:**
```python
q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
# ...
q_pe = self._rope_interleaved(q_pe, start_pos, T)
k_pe = self._rope_interleaved(k_pe, start_pos, T)
q_nope = q_nope @ self.attn_k_b.weight.transpose(-1, -2)
q = q_nope.cat(q_pe, dim=-1)
```

**Optimization**: Eliminate split/cat overhead by:
- For decode (T=1), fuse the split → transform → cat into single operations
- Or avoid split/cat entirely by indexing into q directly

### Hypothesis B: Fuse both q and k path concatenations
**Location**: Lines 106 and 108

**Current**:
```python
q = q_nope.cat(q_pe, dim=-1)  # Line 106
k_new = kv_normed.cat(k_pe, dim=-1)  # Line 108
```

**Optimization**: Scheduler fuses both concatenation operations together

### Hypothesis C: Decode-specific fast path
**Location**: Lines 117-122 have T>1 vs T==1 branches

**Current**: Different softmax path for decode (lines 120-122)
**Optimization**: Add T==1 specific path for lines 98-108 that avoids split/cat

### Testing Strategy
1. Try removing `.contiguous()` calls that might prevent fusion
2. Add conditional for T==1 decode path
3. Experiment with reordering operations to allow scheduler fusion

---

## 2. split_shexp (+8-12% speedup) - ✅ IMPLEMENTED

**Location**: mla.py lines 145-149
**What**: Split shared expert gate/up computation into separate ops
**Status**: Implemented, awaiting benchmark confirmation

---

## 3. merged (+9% speedup)

### Evidence
- bench_merged.log: 19.43 → 21.21 tok/s (+9%)

### Hypothesis: Scheduler-level kernel merging
**Possible locations:**
1. `tinygrad/schedule/rangeify.py` - Pattern matchers for operation fusion
2. `tinygrad/codegen/opt/` - Kernel optimization passes
3. Adjacent operations in MoE or attention that can merge

**What to look for:**
- Pattern matchers that merge adjacent ops (e.g., add+mul, reshape+transpose)
- Fusion opportunities in the 1358 kernels/token
- Specific patterns in RMSNorm (2 kernels) or softmax (3 kernels) that could merge

**Testing approach:**
1. Run with DEBUG=3 to see kernel fusion decisions
2. Compare kernel counts: merged should have fewer kernels
3. Look for patterns like "fused X+Y" in kernel names

---

## 4. kernel_opt (+7% speedup)

### Evidence
- bench_kernel_opt.log: 19.43 → 20.75 tok/s (+7%)

### Hypothesis: Heuristic optimizer improvements
**Possible locations:**
1. `tinygrad/codegen/opt/heuristic.py` - MV heuristic, GROUP/GROUPTOP
2. `tinygrad/codegen/opt/postrange.py` - Optimization passes
3. `tinygrad/codegen/opt/search.py` - Beam search parameters

**What it might be:**
- Improved upcasting/threading for matmul kernels
- Better GROUP/GROUPTOP decisions for reduces
- Tensor core utilization improvements
- Better vectorization (ALLOW_HALF8, etc.)

**Testing approach:**
1. Compare kernel bandwidth: kernel_opt should show higher GB/s on bottleneck kernels
2. Check if threadgroup memory usage increased (better GROUP)
3. Look for changes in kernel shapes (better tiling)

---

## 5. FOLD_ON (-9.5% when ENABLED) - SHOULD DISABLE

### Evidence
- bench_q4_0_fold_on.log: 21.20 → 19.18 tok/s (-9.5%)
- Enabling this HURTS performance

### Hypothesis: Constant folding or operation folding that prevents fusion
**Possible sources:**
- Eager constant evaluation that prevents later optimizations
- Folding operations together that should stay separate for scheduler
- Over-aggressive fusion that creates large inefficient kernels

**Action**: Find and DISABLE this for GLM/MoE models

---

## 6. group5 (+56% speedup, RISKY)

### Evidence
- profile_group5.log: 17.40 → 27.08 tok/s (+56%)
- 1404 → 780 kernels/token (-44% kernel count)

### What it is
- Aggressive GROUP/GROUPTOP schedule optimization
- Pushes more reductions into shared memory
- Risks exceeding Metal's 32KB threadgroup memory limit

### Current guards (already in place)
1. `indexing.py:75-83` - Cap individual LOCAL buffers at 32KB
2. `postrange.py:153-156` - Check total DEFINE_LOCAL bytes before GROUP

### Why it works
- GLM has many small reduces (RMSNorm, softmax)
- GROUP moves per-thread reductions into shared memory
- Fewer syncs, better locality, massive kernel count reduction

### Implementation
- Already has safety guards
- Needs testing on full benchmark suite
- May need model-specific tuning

---

## Implementation Order

### Week 1: Identify & implement both_fused
1. Experiment with attention path modifications
2. Test split/cat elimination for decode
3. Measure kernel count and bandwidth changes
4. Target: 19.43 → 23+ tok/s

### Week 2: Find merged & kernel_opt
1. Analyze schedule/codegen code for optimization flags
2. Compare DEBUG output between baseline and optimized
3. Implement identified optimizations
4. Target: 23 → 25+ tok/s combined

### Week 3: group5 with safety
1. Enable aggressive GROUP with existing guards
2. Full regression testing (all models, all quant types)
3. Monitor for Metal crashes
4. Target: 25 → 30+ tok/s if stable
