# Advanced Optimization Ideas from Recent PRs

**Analysis date**: 2026-02-10
**Source**: geohot's recent PRs (#14628, #14400, #14665, #14604, #14577)
**Current performance**: GLM-4.7-Flash Q4_0 = 20 tok/s → **Target: 35 tok/s** (llama.cpp parity)

---

## Executive Summary

After analyzing geohot's recent architectural improvements, several high-impact optimizations could significantly improve GLM inference:

1. **E-graph optimization** (PR #14628) - Automatic kernel fusion discovery
2. **PARAM normalization** (PR #14665) - Better schedule cache hits
3. **Atomic scatter for MoE** (PR #14400) - Parallel expert aggregation
4. **CALL unification** (PR #14577, #14604) - Cleaner custom kernel integration

**Conservative estimate**: 3.9x speedup (20 → 78 tok/s)
**Realistic target**: 40-60 tok/s (2-3x speedup)

---

## 1. E-graph for MoE Expert Selection (HIGHEST PRIORITY)

### Concept
Use E-graph saturation to optimize the complex MoE expert routing expressions.

### Current Problem
- GLM has 64 experts per MoE layer
- Expert routing: softmax(top_k(gate_logits)) → routing weights
- Current: 1358 kernels/token, MoE = 54% of GPU time
- Each expert path creates separate kernel invocations with minimal reuse

### E-graph Solution
```python
# From PR #14628: egraph_saturate() finds ALL equivalent expressions
# Then egraph_extract() picks the cheapest one

# Current: Each expert invocation scheduled separately
expert_out[i] = mul_mat_id(x, expert_weights[i], expert_ids)

# E-graph could discover:
# 1. Common subexpressions across experts
# 2. Fused routing weight computation
# 3. Batched expert invocations that share memory loads
```

### Implementation Plan
1. Apply `egraph_saturate()` to entire MoE layer's UOp graph (not per expert)
2. Let it discover equivalent formulations through pattern matching
3. Use `egraph_extract()` with cost model favoring:
   - Fewer memory loads (weights loaded once, not 4x)
   - Batched matmuls
   - Fused routing + computation

### Expected Impact
- **Kernel reduction**: 1358 → ~800-1000 kernels per token
- **Performance**: 20 tok/s → **40-60 tok/s** (2-3x speedup)

### Why It Works
- PR #14628 shows egraph can saturate with rebuild (e.g., `(a*0)+b` → `b`)
- MoE has many algebraically equivalent paths (different expert orderings)
- Cost-based extraction naturally picks most efficient kernel fusion

---

## 2. PARAM Normalization for Expert Weights (HIGH PRIORITY)

### Concept
Use PARAM-based cache normalization (PR #14665) for better expert weight reuse.

### Current Problem
```python
# Each expert has different BUFFER UOp → different schedule cache key
# Even though computation pattern is IDENTICAL:
expert_0_out = matmul(x, expert_weights[offset_0])  # Cache miss
expert_1_out = matmul(x, expert_weights[offset_1])  # Cache miss
expert_2_out = matmul(x, expert_weights[offset_2])  # Cache miss
# ... 64 cache misses per layer!
```

### PARAM Solution
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

### Expected Impact
- **Schedule cache hit rate**: 0% → 98% for expert matmuls
- **Compile time**: Eliminates 64x redundant kernel compilation per layer
- **Runtime speedup**: 1.5-2x from better cache utilization
- **Note**: Already partially implemented in `metal_mul_mat_id.py` (byte_off approach)

---

## 3. Atomic Scatter for Expert Routing (MEDIUM PRIORITY)

### Concept
Use atomic operations for expert output aggregation (inspired by PR #14400).

### Current Problem
```python
# Expert outputs accumulated sequentially:
output = sum(routing_weight[i] * expert_out[i] for i in selected_experts)

# Current: Sequential accumulation, each expert runs separately
# This is O(n_experts) serial kernels
```

### Atomic Solution
```python
def moe_output_aggregate(grad_weight:UOp, routing_weights:UOp, expert_outs:UOp) -> UOp:
    # Parallel atomic accumulation like embedding backward
    i = UOp.range(n_tokens, 0)
    j = UOp.range(n_experts_selected, 1)
    expert_id = expert_ids[i, j]

    # Atomic add: output[i] += routing_weights[i,j] * expert_outs[expert_id][i]
    # Metal syntax: atomic_fetch_add_explicit(&output[i], value, memory_order_relaxed)
    atomic = UOp(Ops.CUSTOM, dtypes.void,
                 (output.index(i, ptr=True), routing_weights[i,j] * expert_outs[expert_id][i]),
                 arg="atomic_fetch_add_explicit({0}, {1}, memory_order_relaxed)")
    return atomic.end(i, j).sink(arg=KernelInfo(name="moe_aggregate"))
```

### Expected Impact
- Eliminate sequential expert output accumulation
- All experts can write to output buffer in parallel
- Reduces kernel count by ~6-8 per MoE layer
- **Speedup**: 1.3-1.5x

### Concerns
- PR #14400 uses AMD-specific atomics (`__hip_atomic_fetch_add`)
- Metal needs different atomic syntax (`atomic_fetch_add_explicit`)
- Memory ordering guarantees matter for correctness

---

## 4. CALL-based Custom Kernels (MEDIUM PRIORITY)

### Concept
Use unified CALL interface (PR #14577, #14604) for cleaner MoE kernels.

### Current State
- `metal_mul_mat_id.py` uses hand-rolled `CompiledRunner`
- Requires careful `ProgramSpec` construction for JIT batching
- Hard to extend/modify

### CALL Solution
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

### Expected Impact
- Simpler custom kernel implementation
- Better integration with scheduler optimizations
- Easier to add new fused operations (e.g., routing + matmul + ReLU)
- **Speedup**: 1.1-1.2x (mainly code cleanliness)

---

## Implementation Priority Ranking

### Tier 1 - Do First (Highest ROI)

#### 1. E-graph optimization for MoE
**Expected**: 2-3x speedup (20 → 40-60 tok/s)
- **Files**: `tinygrad/uop/egraph.py` (PR #14628)
- **Apply to**: `tinygrad/apps/mla.py` MoE forward pass
- **Test**: GLM-4.7-flash with/without egraph
```bash
# Baseline
python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --benchmark 20 > baseline.log

# E-graph (if PR #14628 merged)
EGRAPH=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --benchmark 20 > egraph.log
```

#### 2. PARAM normalization for expert cache hits
**Expected**: 1.5-2x speedup from better caching
- **File**: `tinygrad/engine/schedule.py` (extend PR #14665 approach)
- **Extend**: `metal_mul_mat_id.py` byte_off pattern
- **Test**: Schedule cache hit rate before/after
```bash
SCACHE_DEBUG=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --count 5 > cache.log
grep "cache hit" cache.log
```

### Tier 2 - Do Second (Good ROI)

#### 3. Atomic scatter for expert aggregation
**Expected**: 1.3-1.5x speedup
- **File**: `tinygrad/nn/__init__.py` (PR #14400 pattern)
- **Requires**: Metal atomic syntax adaptation
- **Test**: Kernel count reduction

#### 4. CALL-based MoE kernels
**Expected**: Code cleanliness, 1.1-1.2x speedup
- **Refactor**: `metal_mul_mat_id.py` to use CALL
- **Benefit**: Easier to extend, better scheduler integration

---

## Expected Overall Impact

### Conservative Estimate
- E-graph MoE: 2x speedup
- PARAM cache: 1.5x speedup
- Atomic scatter: 1.3x speedup
- **Combined: 3.9x speedup** (20 → 78 tok/s)

### Realistic Target
- E-graph MoE: 2.5x speedup (better fusion than expected)
- PARAM cache: 1.5x speedup
- Atomic scatter: 1.3x speedup
- **Combined: 4.8x speedup** (20 → 96 tok/s)

**Practical target: 40-60 tok/s** (2-3x speedup, exceeds llama.cpp's 35 tok/s)

---

## Technical Concerns

### E-graph Concerns
- PR #14628 in `test/null/test_egraph.py` - still experimental
- May need stability work before production use
- Cost model for extraction needs MoE-specific tuning

### PARAM Normalization Concerns
- Already implemented in schedule.py (PR #14665)
- Just needs extension to MoE-specific patterns
- May need custom PatternMatcher rules for expert weight access

### Atomic Scatter Concerns
- Only works on AMD/CPU in PR #14400
- Metal needs different atomic syntax
- Memory ordering guarantees critical for correctness

---

## Related Files to Modify

### Core Changes
1. `tinygrad/uop/egraph.py` - Import from PR #14628 (if not merged)
2. `tinygrad/engine/schedule.py` - Extend PARAM normalization
3. `tinygrad/apps/mla.py` - Apply E-graph to MoE

### Supporting Changes
1. `tinygrad/nn/__init__.py` - Atomic scatter aggregation
2. `tinygrad/nn/metal_mul_mat_id.py` - CALL refactoring
3. `tinygrad/apps/quantized.py` - Expert weight alignment

### Testing
1. `test/test_moe_egraph.py` - New test for E-graph MoE
2. `test/test_schedule.py` - Verify PARAM cache hits
3. `profile_model.py` - Benchmark harness

---

## Comparison to Current Approach

### Current (metal_mul_mat_id.py)
- Hand-written Metal kernels per quantization type
- Byte offset baking for zero-copy weights
- CompiledRunner + ProgramSpec for JIT batching
- **Result**: 20 tok/s

### Proposed (E-graph + PARAM + Atomics)
- E-graph discovers optimal kernel fusion automatically
- PARAM normalization ensures cache hits
- Atomic scatter for parallel aggregation
- **Expected**: 40-60 tok/s (2-3x faster)

**Key Difference**: Current approach optimizes individual kernels. Proposed approach optimizes the entire computation graph, finding opportunities invisible to local optimization.

---

## Next Steps

1. **Import E-graph code** from PR #14628 (if not merged)
2. **Profile baseline** with VIZ=-1 to confirm MoE is still bottleneck
3. **Implement E-graph for MoE** in mla.py
4. **Extend PARAM normalization** to expert weight access patterns
5. **Add atomic scatter** for expert output aggregation
6. **Benchmark** and iterate

---

## References

- **PR #14628**: E-graph implementation with saturation and extraction
- **PR #14400**: Atomic operations for embedding backward (128ms on MI300X)
- **PR #14665**: PARAM normalization in schedule for better cache hits
- **PR #14604**: Remove CUSTOM_KERNEL, use direct CALL construction
- **PR #14577**: KERNEL op deleted, replaced with CALL
- **MEMORY.md**: mul_mat_id performance history, optimization learnings
