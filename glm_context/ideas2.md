# GLM Optimization Ideas — Ranked by Expected Impact

**Date**: 2026-02-12
**Current**: ~27-30 tok/s (recent improvements per optimization_journal.md)
**Target**: 40 tok/s (comfortable, exceeding llama.cpp's 35 tok/s)
**Gap**: ~33-48% improvement needed

---

## Tier 1: Highest Impact (10-100% improvement each)

### 1. E-graph Optimization for MoE ⭐⭐⭐⭐⭐

**Expected Impact**: 2-3x speedup (MoE is 54% of decode time) → **40-80 tok/s**
**Confidence**: Medium (experimental, PR #14628)
**Risk**: High (cutting-edge feature, may need stability work)
**Effort**: 3-5 days

**Concept**: Apply `egraph_saturate()` to entire MoE layer's UOp graph instead of per-expert scheduling. Let it discover equivalent expressions through pattern matching, then use `egraph_extract()` with cost model favoring fewer memory loads.

**Why It Works**:
- MoE has many algebraically equivalent paths (different expert orderings)
- Current: 1358 kernels/token, each expert invocation scheduled separately
- E-graph discovers common subexpressions across experts, batched matmuls that share memory loads
- Weight loading once instead of 4x per token

**Implementation**:
```python
# In tinygrad/apps/mla.py, _feed_forward method
# Apply egraph_saturate to full MoE expression before scheduling
moe_uop = (expert_out * probs).sum(axis=2)  # Current
saturated = egraph_saturate(moe_uop)
optimal = egraph_extract(saturated, cost_model=prefer_fewer_loads)
```

**Files**:
- `tinygrad/uop/egraph.py` (import from PR #14628)
- `tinygrad/apps/mla.py` (apply to MoE forward pass)

**Test**:
```bash
# Baseline
.venv2/bin/python bench_block.py 30 --model "glm-4.7:flash-unsloth-Q4_0"

# With E-graph (if PR merged)
EGRAPH=1 .venv2/bin/python bench_block.py 30 --model "glm-4.7:flash-unsloth-Q4_0"
```

**Blockers**: PR #14628 may still be experimental, need to check stability.

---

### 2. PARAM Normalization for Expert Cache Hits ⭐⭐⭐⭐

**Expected Impact**: 1.5-2x speedup → **40-60 tok/s**
**Confidence**: High (PR #14665 already landed in schedule.py)
**Risk**: Low (just needs extension to MoE patterns)
**Effort**: 2-3 days

**Problem**:
```python
# Each expert has different BUFFER UOp → different schedule cache key
# Even though computation pattern is IDENTICAL:
expert_0_out = matmul(x, expert_weights[offset_0])  # Cache miss
expert_1_out = matmul(x, expert_weights[offset_1])  # Cache miss
# ... 64 cache misses per layer!
```

**Solution**:
```python
# Replace BUFFER with PARAM before cache key computation
# pm_pre_sched_cache replaces BUFFER with PARAM(id, dtype, shape, device)
# Now all experts hit SAME cache entry

def expert_matmul_normalized(x, expert_base, byte_offset):
    # Bake offset as #define instead of buffer view
    kernel = f"""
    #define EXPERT_BYTE_OFF {byte_offset}
    const float* expert_ptr = (float*)(expert_base + EXPERT_BYTE_OFF);
    """
    return Tensor.call(x, expert_base, fxn=kernel)
```

**Impact**:
- Schedule cache hit rate: 0% → 98% for expert matmuls
- Eliminates 64x redundant kernel compilation per layer
- Runtime speedup from better cache utilization

**Files**:
- `tinygrad/engine/schedule.py` (extend PARAM normalization)
- `tinygrad/apps/quantized.py` (apply to QuantizedExpertWeights)

**Note**: Already partially implemented in metal_mul_mat_id.py byte_off approach.

---

### 3. Custom mul_mat_id MSL Kernel for Q4_0 Experts ⭐⭐⭐⭐

**Expected Impact**: -17.5ms/tok → **50-65 tok/s**
**Confidence**: High (pattern proven with Q4K/Q6K)
**Risk**: Medium-high (custom Metal code, JIT integration)
**Effort**: 5-7 days

**Problem**: Q4_0 packed-dot kernels hit **~50 GB/s ceiling** from scattered byte reads in tinygrad-generated code. Expert gate/up: 51 GB/s, down: 47 GB/s. llama.cpp achieves 145-229 GB/s.

**Solution**: Hand-written MSL kernel with:
- Coalesced `uint16_t` reads (not scattered `unsigned char`)
- SIMD reduction within threadgroup
- Multi-row amortization
- Expert selection (gather) fused into kernel

**Expected Savings**:
- Gate/up: 14.2ms → ~4.5ms (-9.7ms)
- Down: 9.8ms → ~2ms (-7.8ms)
- **Total: -17.5ms/tok**

**Implementation Pattern** (from Q4K/Q6K runners):
```python
class Q4_0MulMatIdRunner(CompiledRunner):
    def __init__(self, expert_weights, n_experts, ...):
        # Define Metal kernel with:
        # 1. #define BYTE_OFF for each expert (baked at compile time)
        # 2. uint16_t coalesced loads
        # 3. SIMD sum reduction
        # 4. ProgramSpec with z=n_sel for ICB dispatch

    def __call__(self, x, expert_ids, ...):
        # JIT batching via CompiledRunner
```

**Files**:
- Create `tinygrad/nn/metal_mul_mat_id_q4_0.py`
- Integrate in `tinygrad/apps/quantized.py` QuantizedExpertWeights

**Critical Requirements**:
- Extend `CompiledRunner` (not `Runner`) for JIT batching
- Use `#define BYTE_OFF` for GGUF offsets (buffer views break JIT)
- Per-layer kernel cached via `lru_cache`

---

## Tier 2: High Impact (5-15% improvement each)

### 4. Fold q_nope @ attn_k_b Into Q Projection Weights ⭐⭐⭐

**Expected Impact**: 5-10% → +1.5-3 tok/s
**Confidence**: High (standard weight composition)
**Risk**: Medium (weight-shape surgery, needs careful parity check)
**Effort**: 1-2 days

**Problem**:
```python
# In glm.py:98 - runs EVERY token:
q_nope = q_nope @ self.attn_k_b.weight.transpose(-1, -2)
# Shape: (B, n_heads, T, qk_nope_head_dim) @ (n_heads, qk_nope_head_dim, kv_lora_rank)
```

For T=1 decode, k_b doesn't change. This is a per-token matmul that can be pre-composed.

**Solution**:
```python
# During model loading in GLMTransformerBlock.__init__:
# Pre-multiply q_nope projection with k_b once
if hasattr(self, 'attn_k_b'):
    # attn_q_b[:, :qk_nope] shape: (n_heads, qk_nope_head_dim, q_lora_rank)
    # attn_k_b.weight shape: (n_heads, kv_lora_rank, qk_nope_head_dim)
    self.attn_q_b_composed = precompose_k_projection(
        self.attn_q_b.weight[:, :qk_nope],
        self.attn_k_b.weight
    )

# In _attention:
# q_nope = self.attn_q_b_composed(q_a_normed)  # Direct to KV-rank
```

**Impact**: Removes one per-head matmul per layer per token (46 blocks × ~50us = ~2.3ms/tok).

**Files**:
- `tinygrad/apps/glm.py` GLMTransformerBlock.__init__ and _attention

---

### 5. Fold (attn_kv @ attn_v_b^T) → attn_output Into One Projection ⭐⭐⭐

**Expected Impact**: 8-12% → +2.4-3.6 tok/s
**Confidence**: High (standard weight composition)
**Risk**: Medium-high (weight algebra + load mapping)
**Effort**: 2-3 days

**Problem**: Two sequential matmuls in attention output path with transpose barrier:
```python
# attn_kv @ attn_v_b.weight^T  (per-head)
# then @ attn_output.weight    (heads → model_dim)
```

**Solution**: Pre-compose `attn_v_b` with `attn_output.weight` during loading:
```python
# In GLMTransformerBlock.__init__:
# attn_v_b.weight: (n_heads, v_head_dim, kv_lora_rank)
# attn_output.weight: (model_dim, n_heads * v_head_dim)
self.attn_output_composed = precompose_v_projection(
    self.attn_v_b.weight,
    self.attn_output.weight
)

# In _attention:
# out = (attn @ attn_kv) @ self.attn_output_composed  # One matmul
```

**Impact**: Removes one large per-token matmul + transpose/reshape barrier (46 blocks × ~60us = ~2.8ms/tok).

**Files**:
- `tinygrad/apps/glm.py` GLMTransformerBlock

---

### 6. Atomic Scatter for Expert Output Aggregation ⭐⭐⭐

**Expected Impact**: 1.3-1.5x on expert aggregation → +4-6 tok/s
**Confidence**: Medium (inspired by PR #14400)
**Risk**: Medium-high (Metal atomic syntax, memory ordering)
**Effort**: 3-4 days

**Problem**: Expert outputs accumulated sequentially:
```python
output = sum(routing_weight[i] * expert_out[i] for i in selected_experts)
# O(n_experts) serial kernels
```

**Solution**: Parallel atomic accumulation:
```python
def moe_output_aggregate(routing_weights, expert_outs, expert_ids):
    # Metal: atomic_fetch_add_explicit(&output[i], value, memory_order_relaxed)
    # All experts write to output buffer in parallel
    atomic = UOp(Ops.CUSTOM, dtypes.void,
                 (output.index(i, ptr=True),
                  routing_weights[i,j] * expert_outs[expert_id][i]),
                 arg="atomic_fetch_add_explicit({0}, {1}, memory_order_relaxed)")
    return atomic.end(i, j).sink(arg=KernelInfo(name="moe_aggregate"))
```

**Impact**:
- Eliminate sequential expert output accumulation
- Reduces kernel count by ~6-8 per MoE layer
- All experts run in parallel

**Concerns**:
- PR #14400 uses AMD-specific atomics
- Metal needs different syntax
- Memory ordering critical for correctness

**Files**:
- `tinygrad/nn/__init__.py` (following PR #14400 pattern)
- `tinygrad/apps/mla.py` (MoE aggregation)

---

### 7. Precompute RoPE cos/sin Tensors in Target Dtype ⭐⭐

**Expected Impact**: 3-5% → +0.9-1.5 tok/s
**Confidence**: High (simple caching)
**Risk**: Low
**Effort**: 0.5-1 day

**Problem**: Repeated tiny cast/reshape kernels for RoPE across all layers/tokens:
```python
freqs_cos, freqs_sin = precompute_freqs(...)  # Every call
# Repeated: reshape(...).cast(...) per layer per token
```

**Solution**:
```python
# In model __init__ or first call:
self.rope_cache = {
    'cos': freqs_cos.reshape(1, 1, max_context, rope_dim//2).cast(model_dtype).realize(),
    'sin': freqs_sin.reshape(1, 1, max_context, rope_dim//2).cast(model_dtype).realize()
}

# In _rope_interleaved:
cos = self.rope_cache['cos'][:, :, pos:pos+T, :]  # Just slice
```

**Impact**: Eliminates repeated reshape/cast kernels (46 blocks × 2 RoPE calls × ~8us = ~0.7ms/tok).

**Files**:
- `tinygrad/apps/mla.py` or `tinygrad/apps/glm.py`
- `tinygrad/apps/rope.py` (add caching helper)

---

## Tier 3: Medium Impact (2-5% improvement each)

### 8. Replace _topk_pairwise with Tensor.topk if Competitive ⭐⭐

**Expected Impact**: 0-3% (if topk improved) or code simplification
**Confidence**: Medium (needs benchmarking)
**Risk**: Low
**Effort**: 0.5 days

**Rationale**: `topk_pairwise` was written because `Tensor.topk` used to generate 29 kernels via `sort()`. May now be better optimized.

**Test**:
```python
# Benchmark both for n=64, k=4 (GLM) and n=64, k=6 (ds2-lite)
# Current: pairwise comparison, O(n²), ~3 kernels
values, indices = topk_pairwise(scores, k)

# Alternative: may be competitive now
values, indices = scores.topk(k, dim=-1)
```

If `Tensor.topk` is competitive in kernel count, delete `topk_pairwise` (saves 16 lines).

**Files**:
- `tinygrad/apps/glm.py` topk_pairwise
- Add benchmark to bench_block.py

---

### 9. Decode-Fast Top-K Path for T==1 ⭐⭐

**Expected Impact**: 2-4% on routing → +0.6-1.2 tok/s
**Confidence**: Medium
**Risk**: Medium (depends on stable top-k op)
**Effort**: 1-2 days

**Problem**: MoE routing runs every layer/token with O(n²) compare graph via _topk_pairwise.

**Solution**:
```python
def _moe_routing(self, x):
    if x.shape[1] == 1:  # T==1 decode path
        # Use direct top-k primitive/indexing, no O(n²) graph
        return fast_decode_topk(scores, self.n_experts_per_tok)
    else:  # Prefill
        return topk_pairwise(scores, self.n_experts_per_tok)
```

**Impact**: Launch-heavy routing becomes lighter for decode (most common path).

**Files**:
- `tinygrad/apps/mla.py` or `tinygrad/apps/glm.py`

---

### 10. Audit and Remove Unnecessary .contiguous() Calls ⭐⭐

**Expected Impact**: ~5% → +1.5 tok/s (fewer kernel dispatches)
**Confidence**: High
**Risk**: Medium (must profile each one)
**Effort**: 1-2 days

**Context**: Current `.contiguous()` calls in glm.py:135-140 were added to break problematic fusion. Some may be unnecessary.

**Approach**:
```python
# Profile with and without each .contiguous():
if self.split_moe_boundaries:
    gate, up = gate.contiguous(), up.contiguous()  # Test removing
...
if self.split_moe_boundaries:
    expert_out = expert_out.contiguous()  # Test removing
```

**Critical**: Never remove speculatively. Always benchmark. The -17% regression history proves this.

**Files**:
- `tinygrad/apps/glm.py` _feed_forward
- `tinygrad/apps/mla.py`

---

### 11. CALL-Based Custom Kernels (Refactoring) ⭐

**Expected Impact**: 1.1-1.2x (mainly code cleanliness) → +0.3-0.6 tok/s
**Confidence**: Low (more about maintainability)
**Risk**: Medium
**Effort**: 3-4 days

**Concept**: Use unified CALL interface (PR #14577, #14604) for cleaner MoE kernels.

**Current State**:
- `metal_mul_mat_id.py` uses hand-rolled `CompiledRunner`
- Requires careful `ProgramSpec` construction
- Hard to extend/modify

**CALL Solution**:
```python
def moe_fused_expert_call(x, weights, routing):
    # Define forward as UOp graph
    fwd = x.as_param(0) @ weights.as_param(1) * routing.as_param(2)

    # Define backward gradient function
    def grad_fn(grad, call):
        x, weights, routing = call.src[1:]
        return (grad @ weights.T, x.T @ grad, None)

    return Tensor.call(x, weights, routing, fxn=fwd, grad_fxn=grad_fn)
```

**Benefits**:
- Simpler implementation
- Better scheduler integration
- Easier to add new fused ops

**Files**:
- Refactor `tinygrad/nn/metal_mul_mat_id.py` to use CALL

---

## Tier 4: Low Impact or Uncertain (< 2% improvement)

### 12. Remove MoE Gate/Up Forced Contiguity Behind Flag

**Expected Impact**: 0-2% (needs testing)
**Risk**: Low-medium (already saw -17% regression)
**Effort**: 0.5 days

Gate `.contiguous()` / `.contiguous()` with env flag to test on/off safely without permanent regression risk.

---

### 13. Kernel Count Reduction via Scheduler Improvements

**Expected Impact**: ~17% dispatch overhead reduction → +5 tok/s
**Risk**: High (deep scheduler changes)
**Effort**: 10+ days (research project)

**Targets**:
- Fuse RMSNorm reduce+ewise: -230 kernels
- Fuse QK + softmax: -46 kernels
- Expert gather consolidation: -138 kernels

**Status**: Partially done (indexing.py:236 change). Remaining needs scheduler understanding.

---

## Implementation Roadmap

### Phase 1: Quick Wins (3-5 days) → +10-20% (33-36 tok/s)
1. ✅ Precompute RoPE tensors (lowest risk)
2. ✅ Fold q_nope @ attn_k_b (removes hot matmul)
3. ✅ Benchmark Tensor.topk vs topk_pairwise

**Expected**: 30 → 33-36 tok/s

### Phase 2: Weight Composition (3-5 days) → +8-12% (36-40 tok/s)
4. ✅ Fold attn_kv @ attn_v_b^T → attn_output
5. ✅ Audit .contiguous() calls

**Expected**: 33-36 → 38-42 tok/s ✅ **TARGET REACHED**

### Phase 3: Advanced (if needed) → +50-150% (45-75 tok/s)
6. ⚠️ PARAM normalization for expert cache hits
7. ⚠️ E-graph for MoE (if stable)
8. ⚠️ Custom Q4_0 mul_mat_id MSL kernel

---

## Testing Protocol

From optimization_journal.md:

```bash
# Micro benchmark (representative)
.venv2/bin/python bench_block.py 30 --model "glm-4.7:flash-unsloth-Q4_0"
# Primary metric: full block median (ms)
# Secondary: feed_forward median (ms)
# Use count=30 for stable results

# Full model verification
.venv2/bin/python tinygrad/apps/llm.py \
    --model "glm-4.7:flash-Q4_0" \
    --benchmark 20 > ./bench.log 2>&1

# Correctness check
.venv2/bin/python tinygrad/apps/llm.py \
    --model "glm-4.7:flash-Q4_0" \
    --prompt "Hello" \
    --count 10 \
    --benchmark > ./combo_stdout.log 2> ./combo_stderr.log
```

---

## Key Learnings (Don't Repeat Past Mistakes)

From bottlenecks.md and experiments.md:

✅ **What Worked**:
- merge_gate_up_experts: +43% (14→20 tok/s) - reduced gathers 2→1
- Weight-level pre-composition: proven pattern

❌ **What Failed**:
- Removing .contiguous() speculatively: -17% regression
- MV heuristic walk through dequant: blocked by range mismatches
- GROUPTOP threshold increase: -33% (applied to ALL kernels)
- ICB barrier removal: 0% (kernels are data-dependent)

⚠️ **Critical Rules**:
1. **Never remove .contiguous() without benchmarking** - fusion != always faster
2. **Benchmark sequentially only** - parallel GPU benchmarks crash machine
3. **Use count=30 for stability** - count=10 is too noisy
4. **Test correctness + performance** - combo run catches wrong output early

---

## References

- **[advanced_ideas.md](advanced_ideas.md)** - E-graph, PARAM, atomics (PR analysis)
- **[next.md](next.md)** - Weight composition priorities
- **[kernel_analysis.md](kernel_analysis.md)** - Per-kernel bottleneck breakdown
- **[bottlenecks.md](bottlenecks.md)** - 3 performance gaps, full experiment log
- **[optimization_journal.md](optimization_journal.md)** - Recent heuristic tuning results
- **MEMORY.md** - Historical learnings, JIT buffer view bug, pre-separation pattern
