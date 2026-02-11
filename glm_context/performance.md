# Performance Benchmarks & Analysis

## Q4K Kernel Performance Reference

### Benchmark Results (4096x4096, batch=1)

| Kernel | Per-kernel | Bandwidth | Notes |
|--------|------------|-----------|-------|
| q4k_linear_msl (warm) | **65us** | **145 GB/s** | Best after GPU warmup |
| q4k_linear_beam | 136us | 70 GB/s | Works with or without BEAM |
| q4k_linear_msl (cold) | 155us | 61 GB/s | Before GPU cache warms up |
| q4k_linear_uop | 1060us | 8.9 GB/s | UOp.special() - BEAM cannot help |
| q4k_linear_tensor | 5400us | 1.7 GB/s | Many unfused kernels |

### Key Insights
- **GPU warmup matters**: MSL kernel goes from 155us (cold) to 65us (warm) — 2x from GPU cache effects
- **UOp.special() vs UOp.range()**: `UOp.special()` fixes parallelism (BEAM can't optimize), `UOp.range()` lets BEAM tile and optimize (70 GB/s with proper opts)
- **Pure tensor ops don't fuse for Q4K**: Bit manipulation prevents fusion, generates ~45 separate kernels

---

## Cross-Model Performance Comparison

| Model | Params | Steady-state | ms/tok | Scaling |
|-------|--------|--------------|--------|---------|
| youtu-llm:2b-Q4 | 0.69 GB | 55 tok/s | 18ms | 1x |
| deepseek-v2-lite | ~2 GB | 20 tok/s | 50ms | 2.8x |
| glm-4.7:flash | ~7 GB | 18 tok/s | 56ms | 3.1x |

**Key finding**: glm4 is 10x more params but only ~3x slower — kernel count/dispatch overhead dominates, not memory bandwidth.

---

## Where Time Goes (youtu-llm:2b-Q4 profile)

| Category | % of Token Time | Notes |
|----------|----------------|-------|
| q4k_linear | 25% | Expert matmuls (already optimized) |
| Elementwise | 40% | Many small E_* kernels, launch-overhead dominated |
| Reductions | 20% | Attention, MoE weighted sums |
| Other | 15% | Gating, routing, misc |

**Root cause**: ~400 kernels/token x ~20us launch overhead = 8ms/token just in dispatch. The real bottleneck is kernel count, not individual kernel efficiency.

---

## MoE Performance Bottlenecks

For MoE models (deepseek-v2-lite, glm-4.7:flash):
1. **MoE output combination is #1 bottleneck** (30% of token time): `expert_output * probs + sum`
2. **Tiny gating kernels** (E_32_2, E_16_2_2n1, etc.) from topk/bitonic sort: ~500 calls/token
3. **SIMPLE_TOPK** (iterative argmax) reduces kernel count 27% but same tok/s at this scale

---

## Theoretical Bandwidth Limits (Apple M-series)

| Chip | Unified Memory BW | Expected tok/s (youtu-2b) | Expected tok/s (glm-4.7) |
|------|------------------|---------------------------|--------------------------|
| M1 | 68 GB/s | 37 tok/s | 12 tok/s |
| M2 | 100 GB/s | 55 tok/s | 18 tok/s |
| M3 | 100 GB/s | 55 tok/s | 18 tok/s |
| M4 | 120 GB/s | 66 tok/s | 21 tok/s |

Calculation: `tok/s = memory_bandwidth / model_size_gb`

Current performance matches M2/M3 theoretical limits, suggesting we're bandwidth-bound at the model level, not kernel-bound.
