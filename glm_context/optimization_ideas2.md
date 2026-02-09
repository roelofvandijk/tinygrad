# GLM-4.7-Flash Optimization Ideas

## Current Status (Feb 9 2026)

- **Q4_0 (no MUL_MAT_ID)**: 0.18 tok/s — 99.5% time in MoE dequant matmuls
- **Q4_K_M (MUL_MAT_ID=1)**: 14.6 tok/s — custom Metal kernels
- **Model**: 64 experts, kv_lora_rank=512, intermediate=10240
- **Kernels**: 1121 kernels/token, 6 ICBs. Batch 6 (129 kernels = MoE experts) = 83% of GPU time

## Key Bottlenecks

### 1. MoE Dequant Matmul Performance (2-15 GB/s)

**Problem**: Quantized MoE expert matmuls run at 2-15 GB/s instead of expected 200+ GB/s on M3 Max.

**Root Causes**:

1. **MV Heuristic Fails on Dequant Chains** ([heuristic.py:72-78](../tinygrad/codegen/opt/heuristic.py#L72-L78))
   - Requires `MUL(INDEX, INDEX)` pattern
   - Fused dequant+matmul has `MUL(scale_bitcast_chain, ADD(...))`
   - Activation INDEX buried 2+ levels deep in dequant operations
   - **Fix**: Added `find_index_through_chain()` helper (lines 64-74) to walk through dequant ops
   - **Status**: Now detects dequant patterns, but may need further tuning

2. **GROUPTOP Blocked for Large MoE Kernels** ([heuristic.py:134](../tinygrad/codegen/opt/heuristic.py#L134))
   ```python
   if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= 2048, False):
   ```
   - Threshold: 2048 elements max
   - MoE kernels: 6 experts × 2048 outputs = 12288 > 2048
   - **Fix Ideas**:
     - Increase threshold to 8192 or 16384 for MoE workloads
     - Make threshold adaptive based on kernel type
     - Add special case for MoE expert kernels

3. **No Cooperative Reduction** (DEBUG=5 analysis from MEMORY.md)
   - Each thread does full reduction serially
   - GROUP optimization only helps first reduction in multireduce
   - Second reduction runs serially with GROUP sync overhead but no benefit

**Verification Command**:
```bash
DEBUG=5 MUL_MAT_ID=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 3 > debug5.log 2>&1
grep -n "kernel void r_" debug5.log  # Find specific kernels
```

### 2. Scheduler Fusion Limits

**Current Issue**: [indexing.py:227-236](../tinygrad/schedule/indexing.py#L227-L236)

The PCONTIG guard prevents fusion when ranges don't match exactly:
```python
if all_all_same or (PCONTIG and all_same(local_rngs)):
    # Fusion succeeds
else:
    # Create new ranges, partial realization
```

**Impact**:
- RMSNorm: 2 kernels (could be 1)
- Softmax: 3 kernels (could be 2)
- Many small reduce ops remain unfused

**Recent Fix** (Feb 9 2026): Removed `not (PCONTIG > 1) or` guard on ending-range check
- Effect: RMSNorm 2→1 kernel, softmax 3→2 kernels
- 50 test_schedule expectations improved
- See [glm_context/scheduler_fusion.md](scheduler_fusion.md) for analysis

**Further Ideas**:
- Increase PCONTIG threshold beyond 1
- Add fusion cost model (avoid creating worse kernels)
- Profile fused vs unfused for common patterns

### 3. Metal Threadgroup Memory Overflow

**Problem**: Scheduler fusion can create >32KB local buffers, causing silent crashes.

**Current Guards** ([indexing.py:75-83](../tinygrad/schedule/indexing.py#L75-L83), [postrange.py:153-156](../tinygrad/codegen/opt/postrange.py#L153-L156)):

```python
# indexing.py - Cap individual LOCAL BUFFERIZE
if is_partial:
    local_bytes = prod([int(r.vmax)+1 for r in closed_ranges]) * s.dtype.itemsize
    if local_bytes > 32768:
        realized_ranges = list(range(len(ctx.range_map[s][1])))
        closed_ranges = tuple(ctx.range_map[s][1])
        is_partial = False

# postrange.py - Account for existing DEFINE_LOCAL bytes in GROUP check
upcast_local_sz = prod([self.full_shape[a] for a in self.axes_of(...)])
smem_sz = amt*upcast_local_sz*self.reduceop.dtype.itemsize
existing_local_bytes = sum(u.dtype.size * u.dtype.base.itemsize for u in self.ast.toposort() if u.op is Ops.DEFINE_LOCAL)
check(smem_sz + existing_local_bytes <= self.ren.shared_max, ...)
```

**Testing**: [test_reduce_broadcast_fusion.py](../test/test_reduce_broadcast_fusion.py) Tier 6 (3-matmul FFN no-crash tests)

### 4. Custom Kernel JIT Batching

**Problem**: Hand-written kernels in [nn/metal_mul_mat_id.py](../tinygrad/nn/metal_mul_mat_id.py) need JIT support.

**Requirements** (from MEMORY.md):
1. Extend `CompiledRunner`, not base `Runner`
2. Bake dimensions as `#define` in source
3. Create proper `ProgramSpec` with global_size, local_size, ins/outs
4. **ProgramSpec global_size must include z=n_sel** for ICB dispatch
5. Use `#define BYTE_OFF` for GGUF offsets (avoid buffer views)

**Current Status**:
- Q4_0, Q4_K, Q6_K mul_mat_id kernels implemented
- Byte offset approach (zero-copy, no buffer view bug)
- ds2-lite: 34 → 43 tok/s (+26%)
- GLM: 10 → 14.6 tok/s (+46%)

## Optimization Strategy

### Phase 1: MoE Kernel Optimization (Highest Impact)

**Target**: Get MoE expert kernels to 100+ GB/s (currently 2-15 GB/s)

1. **Fix GROUPTOP Threshold**
   - File: [heuristic.py:134](../tinygrad/codegen/opt/heuristic.py#L134)
   - Change: `2048 → 8192` or add MoE detection
   - Test: Verify GROUP optimization applies to MoE kernels

2. **Improve MV Heuristic for Dequant**
   - File: [heuristic.py:64-110](../tinygrad/codegen/opt/heuristic.py#L64-L110)
   - Current: `find_index_through_chain()` helper added
   - Verify: Check if MV optimization triggers on dequant+matmul
   - Test command:
     ```bash
     DEBUG=3 MUL_MAT_ID=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 3 2>&1 | grep MATVEC
     ```

3. **Q4_0 Packed-Dot GROUP Optimization**
   - File: [heuristic.py:112-131](../tinygrad/codegen/opt/heuristic.py#L112-L131)
   - Detects bitwise ops (AND, SHR) in reduce chain
   - Applies GROUP + LOCAL + UPCAST directly
   - Test: Verify this triggers for Q4_0 MoE kernels

### Phase 2: Scheduler Fusion (Medium Impact)

**Target**: Reduce kernel count, improve RMSNorm/softmax performance

1. **Monitor Fusion Changes**
   - Recent fix: indexing.py:236 removed PCONTIG guard
   - Effect: 50 test improvements, RMSNorm/softmax fusion
   - **Watch for**: Threadgroup memory overflows (32KB limit)

2. **Cost Model for Fusion Decisions**
   - File: [schedule/indexing.py:227-236](../tinygrad/schedule/indexing.py#L227-L236)
   - Current: Simple "all_same" check
   - Idea: Estimate kernel performance before/after fusion
   - Prevent: Creating kernels with poor memory access patterns

3. **Fusion Testing**
   - Run: [test/test_schedule.py](../test/test_schedule.py)
   - Verify: [test/test_fusion_op.py](../test/test_fusion_op.py)
   - Check: [test/test_softmax_fusion.py](../test/test_softmax_fusion.py)

### Phase 3: Advanced Optimizations (Lower Impact)

1. **Better Kernel Deduplication**
   - Schedule cache: [engine/schedule.py:133-209](../tinygrad/engine/schedule.py#L133-L209)
   - Current: Strips BIND values for cache key normalization
   - Idea: Better cache key computation for similar kernels

2. **Multi-Kernel Optimization**
   - File: [schedule/rangeify.py](../tinygrad/schedule/rangeify.py)
   - Current: Each kernel optimized independently
   - Idea: Consider kernel fusion opportunities during split

3. **Attention Kernel Fusion**
   - MLA attention in [apps/mla.py](../tinygrad/apps/mla.py)
   - Could benefit from flash attention patterns
   - See: [test/test_softmax_fusion.py](../test/test_softmax_fusion.py) for examples

## Profiling Tools

### Quick Profiling
```bash
# See glm_context/tools.md for full details
python profile_model.py glm-4.7:flash 20
python profile_model.py glm-4.7:flash 20 --with-source  # Includes Metal source
```

### Detailed Analysis
```bash
# Capture full trace
VIZ=-1 DEBUG=2 PROFILE=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 10

# View results
PYTHONPATH=. python extra/viz/cli.py --profile --device METAL
PYTHONPATH=. python extra/viz/cli.py --profile --device METAL --kernel "<kernel_name>"
```

### Metal Source Inspection
```bash
DEBUG=5 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 3 > debug5.log 2>&1
grep -n "kernel void r_9_32" debug5.log  # Find specific kernel
```

**Look for**:
- Grid dims (gidx0, gidx1): Need ~2000+ workgroups for bandwidth saturation
- Threadgroup memory: `threadgroup float` indicates GROUP active
- Reduction loops: Long `for (int Ridx...)` = missing GROUPTOP
- Scattered byte reads: `unsigned char val = *(data+...)` = poor coalescing

## Testing Commands

```bash
# Smoke test
MUL_MAT_ID=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --prompt "Hello" --count 10 > ./smoke.log 2>&1

# Benchmark
MUL_MAT_ID=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 20 > ./bench.log 2>&1

# With profiling
DEBUG=2 PROFILE=1 MUL_MAT_ID=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 10

# Check MV heuristic
DEBUG=3 MUL_MAT_ID=1 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_K_M" --benchmark 3 2>&1 | grep MATVEC
```

## Expected Performance Targets

### M3 Max (400 GB/s memory bandwidth)

- **Theoretical max**: ~154 tok/s (for 2.6B active params in bf16)
- **Current**: 14.6 tok/s (Q4_K_M with MUL_MAT_ID)
- **Target**: 80-100 tok/s (50-60% of theoretical)

**Breakdown**:
- MoE kernels (83% of time): Need 100+ GB/s (currently 2-15 GB/s) = **~7x speedup**
- Other kernels (17% of time): Already reasonable performance
- Overall: **5-7x total speedup possible**

### Validation

After each optimization:
1. Run benchmark: `MUL_MAT_ID=1 python tinygrad/apps/llm.py --benchmark 20`
2. Check kernel count: Should decrease or stay same
3. Profile top kernels: `python profile_model.py glm-4.7:flash 10`
4. Verify correctness: `--prompt "Hello" --count 10` should produce coherent text

## References

- [MEMORY.md](../MEMORY.md) - Historical optimization attempts and learnings
- [glm_context/scheduler_fusion.md](scheduler_fusion.md) - Fusion analysis
- [glm_context/architecture.md](architecture.md) - MLA, MoE details
- [glm_context/performance.md](performance.md) - Benchmark results
- [glm_context/bottlenecks.md](bottlenecks.md) - Detailed hotspot analysis
- [CLAUDE.md](../CLAUDE.md) - Kernel optimization guide

## Next Steps

1. **Immediate**: Test GROUPTOP threshold increase from 2048 to 8192
2. **Verify**: MV heuristic now detects dequant patterns (check with DEBUG=3)
3. **Monitor**: Threadgroup memory usage after scheduler fusion changes
4. **Measure**: Profile before/after each change with `profile_model.py`
5. **Document**: Update this file with results of each optimization attempt
