# Q4K Kernel Performance Analysis

## Benchmark Results (4096x4096, batch=1)

| Kernel | Per-kernel | Bandwidth | Notes |
|--------|------------|-----------|-------|
| q4k_linear_msl (warm) | **65us** | **145 GB/s** | Best after GPU warmup |
| q4k_linear_beam | 136us | 70 GB/s | Works with or without BEAM |
| q4k_linear_msl (cold) | 155us | 61 GB/s | Before GPU cache warms up |
| q4k_linear_uop | 1060us | 8.9 GB/s | UOp.special() - BEAM cannot help |
| q4k_linear_tensor | 5400us | 1.7 GB/s | Many unfused kernels |

## LLM Inference Results (youtu-llm:2b-Q4)

| Mode | tok/s | Notes |
|------|-------|-------|
| Q4K_FUSED=0 | 20 | Dequant weights once, then regular matmul |
| Q4K_FUSED=1 | 20 | q4k_linear_beam fused kernel |

**Conclusion**: Q4K kernel is NOT the bottleneck at 20 tok/s. Bottleneck is elsewhere (attention, Python overhead, kernel launch overhead).

## Key Findings

### 1. q4k_linear_beam achieves 70 GB/s

The UOp.range() custom kernel now works without BEAM by using specific opts:
```python
opts = (Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.GROUP, 0, 0))
```

- 136us per kernel = 70 GB/s
- No longer crashes without BEAM
- Default opts (UPCAST/UNROLL) break register accumulators → use GROUP opts instead

### 2. GPU warmup matters for hand-tuned kernels

The MSL kernel performance varies dramatically:
- Cold: 155us (61 GB/s)
- After ~30 iterations: 65us (145 GB/s)

This 2x speedup from warmup suggests GPU instruction/data cache effects.

### 3. UOp.special() vs UOp.range()

- `UOp.special()` (gidx0, lidx0) → fixed parallelism, BEAM cannot optimize
- `UOp.range()` → BEAM can tile and optimize, 70 GB/s with proper opts

### 4. Pure tensor ops don't fuse well for Q4K

The tensor-based approach generates ~45 separate kernels:
- `r_128_16_4_4_2_4_2_32` at ~305us each
- Bit manipulation ops prevent fusion

### 5. Custom kernel structure that works

```python
def _kernel(out: UOp, blk_u: UOp, inp: UOp) -> UOp:
    # Use UOp.range() NOT UOp.special() for BEAM optimization
    n = UOp.range(batch, 0)           # batch dimension
    m = UOp.range(out_features, 1)    # output dimension

    # Reduction axes with AxisType.REDUCE
    b = UOp.range(blocks_per_row, 2, AxisType.REDUCE)
    k = UOp.range(256, 3, AxisType.REDUCE)

    # Register accumulator
    acc = UOp.placeholder((1,), dtypes.float, slot=0, addrspace=AddrSpace.REG)
    acc = acc[0].set(0.0)

    # ... dequant logic ...

    # Accumulate with proper loop ending
    acc = acc[0].set(acc.after(k)[0] + w * x_val, end=k)
    acc = acc.after(acc.barrier().end(b))

    return out[n, m].store(acc[0]).end(n, m).sink(arg=KernelInfo(name="q4k_linear_beam"))
```

### 4. BEAM optimizes the schedule

With BEAM=4, the kernel gets tiled optimally. Without BEAM, it would be slower.

## Implications for LLM inference

1. **Replace q4k_linear_uop with q4k_linear_beam pattern** - The UOp.range() approach is faster
2. **Don't rely on tensor fusion for complex ops** - Custom kernels are necessary for Q4K
3. **BEAM search works on UOp.range() kernels** - Worth enabling BEAM for quantized models

## Files

- [q4k_beam.py](q4k_beam.py) - BEAM-optimizable Q4K implementations
- [q4k_uop_bench.py](q4k_uop_bench.py) - Benchmark comparing UOp vs MSL kernels
- [q4k_microbench.py](q4k_microbench.py) - Microbenchmark for MSL kernel

## Full LLM Profile (VIZ=-1, youtu-llm:2b-Q4)

After 10 benchmark iterations (steady-state ~29 tok/s @ 34ms/token):

| Kernel | Total Time | Count | Per-call | % of Total |
|--------|------------|-------|----------|------------|
| q4k_linear | 107.9ms | 2080 | **52us** | 25.5% |
| E_256512_2_4_2_16_2_2 | 36.2ms | 10 | 3.6ms | 8.5% |
| r_2048_16_384 | 27.4ms | 144 | 190us | 6.5% |
| r_1336_32_3_512_4 | 20.0ms | 9 | 2.2ms | 4.7% |
| E_16_32_4n66 | 16.9ms | 567 | 30us | 4.0% |
| r_16_128n1 | 16.5ms | 567 | 29us | 3.9% |
| E_64_32_3 | 13.3ms | 288 | 46us | 3.1% |
| ... | ... | ... | ... | ~40% |

**Key insight**: q4k_linear is only 25% of time. The other 75% is elementwise and reduction kernels!

## Cross-Model Comparison

| Model | Params | Steady-state | ms/tok | Scaling |
|-------|--------|--------------|--------|---------|
| youtu-llm:2b-Q4 | 0.69 GB | 29 tok/s | 34ms | 1x |
| deepseek-v2-lite | ~2 GB | 20 tok/s | 50ms | 1.5x |
| glm-4.7:flash | ~7 GB | 8.5 tok/s | 117ms | 3.4x |

glm4 is 10x more params but only 3.4x slower → **kernel count/overhead dominates, not memory bandwidth**.

## Root Cause Analysis

1. **Q4K kernel is fast enough** - 52us/call, 2080 calls = 108ms across 10 tokens = 10.8ms/token
2. **Reduction kernels are expensive** - r_2048_16_384 at 190us/call, 144 calls = 27.4ms for 10 tokens
3. **Many small kernels** - E_16_32_4n66: 567 calls at 30us each, dominated by launch overhead
4. **Kernel count is the issue** - ~400 kernels/token × ~20us launch overhead = 8ms/token overhead

## MLA_MSL Kernel Analysis

The existing `mla_attention` kernel in `extra/gemm/metal_mla_attention.py` is **21x slower** than tinygrad-generated kernels:

| Kernel | Per-call | Total | Notes |
|--------|----------|-------|-------|
| mla_attention (MLA_MSL=1) | **4ms** | 384ms (47%) | Serial O(n) loop over context |
| r_2048_16_384 (default) | 190us | 27ms (6.5%) | Parallel reduction |

**Root cause**: The MSL kernel loops over context positions serially (lines 51-67) with only TG=32 threads. As context grows, time grows linearly.

**Verdict**: MLA_MSL=1 is slower than default. Don't use it.

## Deepseek-v2-lite Profile (VIZ=-1)

**Config**: 64 experts, 4 selected per token, 27 MLA layers

**Steady state**: 20 tok/s @ 50ms/token, **1194 kernels per token**

| Kernel | Count/10tok | Per-token | Time | Notes |
|--------|-------------|-----------|------|-------|
| q4k_linear | 1490 | 149 | 56.5ms | Expert matmuls (3 × 4 × 27 = 324 expected) |
| E_32_2 | 1404 | 140 | 38.9ms | MoE gating tiny ops |
| E_16_2_2n1 | 1170 | 117 | 32.5ms | MoE routing |
| E_8_4_2 | 936 | 94 | 26.1ms | topk/gather fragments |
| r_88_32_3_8_4_2_32_8_4_2_32 | 234 | 23 | 87.9ms | Attention? |
| r_16_32_4_1408_6 | 234 | 23 | 143.4ms | Large reduction |

**Key insight**: The tiny E_* kernels from MoE gating (E_32_2, E_16_2_2n1, E_8_4_2) total ~500 calls/token. These are from `topk` using bitonic sort on 64 elements.

**llama.cpp difference**: They don't do tensor gathers. They compute expert indices then call kernels directly with pointer offsets. No intermediate gather kernels.

## SIMPLE_TOPK Optimization

Using iterative argmax instead of full bitonic sort for topk:

| Mode | Kernels/token | tok/s | Notes |
|------|---------------|-------|-------|
| Default (bitonic sort) | 1194 | 20 | Full sort for 64 elements |
| SIMPLE_TOPK=1 | **867** | 20 | Iterative argmax, k=4 iterations |

**27% kernel reduction** (1194 → 867) but same tok/s - kernel launch overhead isn't the main bottleneck at this scale. May help more on larger models.

Code in `tinygrad/apps/mla.py`:
```python
def _topk_simple(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
  """O(k*n) topk using iterative argmax. Fewer kernels than O(n log n) bitonic sort."""
  masked = scores.float()
  for _ in range(k):
    idx = masked.argmax(axis=-1, keepdim=True)
    val = masked.gather(-1, idx)
    # mask out selected element
    mask = idx._one_hot_along_dim(n).squeeze(-2)
    masked = mask.where(-inf, masked)
  return cat(vals), cat(indices)
```

## Kernel Analysis (with SIMPLE_TOPK=1)

| Kernel | Time (10 tok) | Per-token | What it does |
|--------|--------------|-----------|--------------|
| r_16_32_4_1408_6* | **148ms** | 14.8ms | MoE weighted output sum (getitem+matmul+mul+sum) |
| r_88_32_3_8_4_2_32 | 91ms | 9.1ms | Shared expert FFN (fused gate*up with silu) |
| q4k_linear | 56ms | 5.6ms | Expert matmuls (already optimized) |
| r_16_4* | 47ms | 4.7ms | Small reductions (argmax for SIMPLE_TOPK) |

**MoE output combination is the #1 bottleneck** (14.8ms/token = 30% of 50ms total):
```python
# Line 201 in mla.py
out = (self.ffn_down_exps(sel, gated).float() * probs.unsqueeze(-1)).sum(axis=2)
```

## Next Steps to Get WAY Faster

### Immediate: Fuse MoE output combination
The `expert_output * probs + sum` pattern should be a single kernel, not a reduction chain.

### Short-term (2-3x speedup)
1. **Fix MOE_FUSED kernel** - Currently 5x SLOWER than unfused (uses UOp.special, needs UOp.range)
2. **Fuse weighted sum** - Make `ffn_down_exps` return pre-weighted sums directly

### Medium-term (5-10x speedup)
3. **Mega-kernel per transformer block** - Fuse: RMSNorm → Q4K matmul → attention → RMSNorm → FFN
4. **Flash attention in UOp DSL** - Parallel tiled attention, not serial loop

### Speculative (10x+)
5. **Pipeline parallelism** - Overlap compute with next token's embedding lookup

## Files

- [q4k_beam.py](q4k_beam.py) - BEAM-optimizable Q4K implementations
- [q4k_uop_bench.py](q4k_uop_bench.py) - Benchmark comparing UOp vs MSL kernels
- [q4k_microbench.py](q4k_microbench.py) - Microbenchmark for MSL kernel
