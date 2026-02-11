# GLM Performance Ideas from geohot's Recent PRs

Analysis date: 2026-02-10
Focus: Embedding optimization, E-graph, and architectural improvements

## Executive Summary

After analyzing geohot's recent PRs (especially #14628 egraph, #14400 atomics for embedding, #14665 PARAM in schedule, #14604 remove CUSTOM_KERNEL, #14577 kernel is call), several architectural improvements could significantly improve GLM inference performance:

1. **E-graph optimization** (PR #14628) - New optimization framework for better pattern matching
2. **Atomic scatter for embedding backward** (PR #14400) - Could apply to MoE expert routing
3. **PARAM normalization in schedule** (PR #14665) - Better cache hits for similar operations
4. **CALL unification** (PR #14577, #14604) - Cleaner custom kernel integration
5. **Pattern matcher improvements** - Better rule saturation and extraction

## Key Ideas for GLM Optimization

### 1. E-graph for MoE Expert Selection (HIGHEST PRIORITY)

**Concept**: Use E-graph saturation to optimize the complex MoE expert routing expressions

**Current Problem**:
- GLM has 64 experts per MoE layer
- Expert routing involves: softmax(top_k(gate_logits)) → produces routing weights
- Current scheduler produces 129 kernels per token for MoE (83% of GPU time)
- Each expert path creates separate kernel invocations with minimal reuse

**E-graph Solution**:
```python
# From PR #14628, egraph_saturate() can find ALL equivalent expressions
# Then egraph_extract() picks the cheapest one

# Current: Each expert invocation is scheduled separately
expert_out[i] = mul_mat_id(x, expert_weights[i], expert_ids)

# E-graph could discover:
# 1. Common subexpressions across experts
# 2. Fused routing weight computation
# 3. Batched expert invocations that share memory loads
```

**Implementation Plan**:
1. Apply `egraph_saturate()` to the MoE layer's entire UOp graph (not per expert)
2. Let it discover equivalent formulations through pattern matching
3. Use `egraph_extract()` with cost model that favors:
   - Fewer memory loads (weights are loaded once, not 6x)
   - Batched matmuls
   - Fused routing + computation

**Expected Impact**:
- Reduce 129 kernels → ~20-30 kernels per MoE layer
- Current: 14.6 tok/s → Target: 30-40 tok/s (2-3x speedup)

**Why it works**:
- PR #14628 shows egraph can saturate with rebuild (e.g., `(a*0)+b` → `b`)
- MoE has many algebraically equivalent paths (different expert orderings)
- Cost-based extraction naturally picks most efficient kernel fusion

### 2. Atomic Scatter for Expert Routing (MEDIUM PRIORITY)

**Concept**: Use atomic operations for expert output aggregation (inspired by PR #14400)

**Current Problem**:
```python
# Expert outputs need to be accumulated:
output = sum(routing_weight[i] * expert_out[i] for i in selected_experts)

# Current: Sequential accumulation, each expert runs separately
# This is O(n_experts) serial kernels
```

**Atomic Solution** (from PR #14400 embedding backward):
```python
def moe_output_aggregate(grad_weight:UOp, routing_weights:UOp, expert_outs:UOp) -> UOp:
    # Parallel atomic accumulation like embedding backward
    i = UOp.range(n_tokens, 0)
    j = UOp.range(n_experts_selected, 1)
    expert_id = expert_ids[i, j]

    # Atomic add: output[i] += routing_weights[i,j] * expert_outs[expert_id][i]
    atomic_arg = "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);"
    atomic = UOp(Ops.CUSTOM, dtypes.void,
                 (output.index(i, ptr=True), routing_weights[i,j] * expert_outs[expert_id][i]),
                 arg=atomic_arg)
    return atomic.end(i, j).sink(arg=KernelInfo(name="moe_aggregate"))
```

**Expected Impact**:
- Eliminate sequential expert output accumulation
- All experts can write to output buffer in parallel
- Reduces kernel count by ~6-8 per MoE layer

### 3. PARAM Normalization for Expert Weights (HIGH PRIORITY)

**Concept**: Use PARAM-based cache normalization (PR #14665) for better expert weight reuse

**Current Problem**:
```python
# Each expert has different BUFFER UOp → different schedule cache key
# Even though the computation pattern is IDENTICAL:
expert_0_out = matmul(x, expert_weights[offset_0])  # Cache miss
expert_1_out = matmul(x, expert_weights[offset_1])  # Cache miss
expert_2_out = matmul(x, expert_weights[offset_2])  # Cache miss
# ... 64 cache misses per layer!
```

**PARAM Solution** (from PR #14665):
```python
# Replace BUFFER with PARAM before cache key computation
# pm_pre_sched_cache replaces BUFFER with PARAM(id, dtype, shape, device)
# Now all experts hit SAME cache entry:

def expert_matmul_normalized(x: Tensor, expert_base: Buffer, byte_offset: int):
    # Instead of: expert_base.view(offset=byte_offset)  # Creates unique BUFFER
    # Use: PARAM + baked offset (like metal_mul_mat_id.py already does)

    kernel = f"""
    #define EXPERT_BYTE_OFF {byte_offset}
    ... matmul kernel ...
    const float* expert_ptr = (float*)(expert_base + EXPERT_BYTE_OFF);
    """
    return Tensor.call(x, expert_base, fxn=kernel)
```

**Expected Impact**:
- Schedule cache hit rate: 0% → 98% for expert matmuls
- Eliminates 64x redundant kernel compilation per layer
- Already partially implemented in `metal_mul_mat_id.py` (byte_off approach)
- Need to extend to scheduler level

### 4. CALL-based Custom Kernels (MEDIUM PRIORITY)

**Concept**: Use unified CALL interface (PR #14577, #14604) for cleaner MoE kernels

**Current State**:
- `metal_mul_mat_id.py` uses hand-rolled `CompiledRunner`
- Requires careful `ProgramSpec` construction for JIT batching
- Hard to extend/modify

**CALL Solution**:
```python
# From PR #14604: CUSTOM_KERNEL deleted, use direct CALL construction
# From PR #14577: KERNEL op deleted, everything is CALL

def moe_fused_expert_call(x: Tensor, weights: Tensor, routing: Tensor) -> Tensor:
    # Define forward function as UOp graph
    fwd = x.as_param(0) @ weights.as_param(1) * routing.as_param(2)

    # Define backward gradient function
    def grad_fn(grad: UOp, call: UOp) -> tuple:
        x, weights, routing = call.src[1:]
        return (grad @ weights.T, x.T @ grad, None)

    # Single call that scheduler can optimize
    return Tensor.call(x, weights, routing, fxn=fwd, grad_fxn=grad_fn)
```

**Expected Impact**:
- Simpler custom kernel implementation
- Better integration with scheduler optimizations
- Easier to add new fused operations (e.g., routing + matmul + ReLU)

### 5. Devectorizer Improvements for MoE (LOW PRIORITY)

**Concept**: Better vectorization for scattered expert weight loads

**Current Problem** (from devectorizer.py analysis):
```python
# split_load_store() splits vectors based on alignment
# MoE expert weights at GGUF byte offsets often misaligned
# Results in scalar loads instead of vec4/vec8

# Example: Q4_K expert weight at offset 0x1234567
# offset % 16 != 0 → can't use float4 loads → 4x slower
```

**Solution**:
```python
def expert_aligned_load(buf: UOp, offset: int, dtype: DType):
    # Align expert weight buffers to 16-byte boundaries in GGUF loader
    # Or use unaligned vector loads on Metal (vload_half8 handles misalignment)

    if offset % 16 != 0:
        # Shift load address down to alignment
        aligned_off = (offset // 16) * 16
        shift = offset % 16
        # Load vector + shuffle to get correct elements
        return buf.index(aligned_off).load(vec=8)[shift:shift+needed]
```

**Expected Impact**:
- Limited (maybe 5-10% for Q4_K)
- Current byte_off approach already handles this reasonably well

## Implementation Priority Ranking

### Tier 1 - Do First (Highest ROI)
1. **E-graph optimization for MoE** (Expected: 2-3x speedup)
   - File: `tinygrad/uop/egraph.py` (PR #14628)
   - Apply to: `tinygrad/apps/mla.py` MoE forward pass
   - Test: GLM-4.7-flash with/without egraph

2. **PARAM normalization for expert cache hits** (Expected: 1.5-2x speedup)
   - File: `tinygrad/engine/schedule.py` (PR #14665 approach)
   - Extend: `metal_mul_mat_id.py` byte_off pattern
   - Test: Schedule cache hit rate before/after

### Tier 2 - Do Second (Good ROI)
3. **Atomic scatter for expert aggregation** (Expected: 1.3-1.5x speedup)
   - File: `tinygrad/nn/__init__.py` (PR #14400 pattern)
   - Add: `USE_ATOMICS=1` for MoE output accumulation
   - Test: Kernel count reduction

4. **CALL-based MoE kernels** (Expected: Code cleanliness, 1.1-1.2x speedup)
   - Refactor: `metal_mul_mat_id.py` to use CALL
   - Benefit: Easier to extend, better scheduler integration

### Tier 3 - Nice to Have
5. **Devectorizer alignment** (Expected: 1.05-1.1x speedup)
   - Only if profiling shows load bandwidth as bottleneck

## Testing Strategy

### Phase 1: Validation (Week 1)
```bash
# Baseline
python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 20 > baseline.log

# E-graph (with egraph.py from PR #14628)
EGRAPH=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 20 > egraph.log

# PARAM normalization
# (Requires implementing pm_pre_sched_cache changes to mla.py)
SCACHE_DEBUG=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --count 5 > cache.log

# Compare kernel counts
grep "kernel" baseline.log egraph.log cache.log
```

### Phase 2: Integration (Week 2)
- Combine E-graph + PARAM normalization
- Add atomic scatter for aggregation
- Full benchmark suite

### Phase 3: Optimization (Week 3)
- Profile with VIZ=-1 to find remaining bottlenecks
- Apply CALL refactoring if helpful
- Fine-tune pattern matchers

## Expected Overall Impact

**Conservative Estimate**:
- E-graph MoE: 2x speedup (129 → ~60 kernels)
- PARAM cache: 1.5x speedup (eliminate recompilation)
- Atomic scatter: 1.3x speedup (parallel aggregation)
- **Combined: 3.9x speedup** (14.6 → 57 tok/s)

**Optimistic Estimate**:
- E-graph MoE: 3x speedup (better fusion discovered)
- PARAM cache: 2x speedup (more cache hits than expected)
- Atomic scatter: 1.5x speedup (eliminate more serialization)
- **Combined: 9x speedup** (14.6 → 131 tok/s)

**Realistic Target**: 40-60 tok/s (3-4x speedup from current 14.6 tok/s)

## Technical Notes

### E-graph Concerns
- PR #14628 is in `test/null/test_egraph.py` - still experimental
- May need stability work before production use
- Cost model for extraction needs MoE-specific tuning

### Atomic Scatter Concerns
- Only works on AMD/CPU (from PR #14400)
- Metal would need different atomic syntax
- Memory ordering guarantees matter for correctness

### PARAM Normalization Concerns
- Already implemented in schedule.py (PR #14665)
- Just needs extension to MoE-specific patterns
- May need custom PatternMatcher rules for expert weight access

## Related Files to Modify

1. **Core Changes**:
   - `tinygrad/uop/egraph.py` - Import from PR #14628 (if not merged)
   - `tinygrad/engine/schedule.py` - Extend PARAM normalization
   - `tinygrad/apps/mla.py` - Apply E-graph to MoE

2. **Supporting Changes**:
   - `tinygrad/nn/__init__.py` - Atomic scatter aggregation
   - `tinygrad/nn/metal_mul_mat_id.py` - CALL refactoring
   - `tinygrad/apps/quantized.py` - Expert weight alignment

3. **Testing**:
   - `test/test_moe_egraph.py` - New test for E-graph MoE
   - `test/test_schedule.py` - Verify PARAM cache hits
   - `profile_model.py` - Benchmark harness

## Key Insights from PR Analysis

1. **E-graph is production-ready for experimentation** (PR #14628)
   - 502 lines of test coverage
   - Handles rebuild propagation correctly
   - Cost-based extraction works

2. **Atomic operations proven for parallel aggregation** (PR #14400)
   - Embedding backward: 20x speedup on MI300X (128ms)
   - Same pattern applies to MoE expert output accumulation

3. **PARAM normalization eliminates cache thrashing** (PR #14665)
   - Schedule cache hit rate crucial for LLMs
   - Different buffer offsets shouldn't break cache
   - Already working in core, just needs extension

4. **CALL unification simplifies custom kernels** (PR #14577, #14604)
   - Cleaner API for custom operations
   - Better gradient support
   - Easier scheduler integration

## Comparison to Current Approach

**Current** (metal_mul_mat_id.py):
- Hand-written Metal kernels per quantization type
- Byte offset baking for zero-copy weights
- CompiledRunner + ProgramSpec for JIT batching
- Result: 14.6 tok/s

**Proposed** (E-graph + PARAM + Atomics):
- E-graph discovers optimal kernel fusion automatically
- PARAM normalization ensures cache hits
- Atomic scatter for parallel aggregation
- Expected: 40-60 tok/s (3-4x faster)

**Key Difference**: Current approach optimizes individual kernels. Proposed approach optimizes the entire computation graph, finding opportunities invisible to local optimization.

## Next Steps

1. **Import E-graph code** from PR #14628 (if not merged)
2. **Profile baseline** with VIZ=-1 to confirm MoE is still bottleneck
3. **Implement E-graph for MoE** in mla.py
4. **Extend PARAM normalization** to expert weight access patterns
5. **Add atomic scatter** for expert output aggregation
6. **Benchmark** and iterate

## References

- PR #14628: E-graph implementation with saturation and extraction
- PR #14400: Atomic operations for embedding backward (128ms on MI300X)
- PR #14665: PARAM normalization in schedule for better cache hits
- PR #14604: Remove CUSTOM_KERNEL, use direct CALL construction
- PR #14577: KERNEL op deleted, replaced with CALL
- Memory: mul_mat_id performance (ds2-lite: 43 tok/s, GLM: 14.6 tok/s)
