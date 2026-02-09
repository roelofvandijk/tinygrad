# GLM Optimization Status (Feb 9, 2026)

## Current Performance
- **Baseline**: 14.96 tok/s (GLM-4.7-Flash Q4_0)
- **Target**: 40 tok/s  
- **Gap**: 2.67x speedup needed

## Real Per-Token Bottlenecks (Baseline Profile)

**Ignore elementwise kernels** - those are weight loading at startup, not per-token inference!

**Top 3 per-token bottlenecks:**
1. `r_96_16_4_4_128_2048` - **57ms @ 39 GB/s** (scattered memory access)
2. `r_2048_16_2_2_3_16` - **48ms @ 38 GB/s** (Q4_0 expert down)
3. `r_384_16_4_4_4_16` - **29ms @ 56 GB/s** (Q4_0 expert gate)

**Target**: 145 GB/s (hand-written Metal performance)
**Gap**: 2.6-3.8x slower than target

## Attempts That Failed

1. ❌ **Bucketing** (17 → 9.3 tok/s) - Masking overhead + cache misses
2. ❌ **Full cache + masking** (17 → 13.9 tok/s) - 100x more compute
3. ❌ **FLASH_ATTENTION=1** (17 → 15.3 tok/s) - Doesn't help MLA
4. ❌ **Aggressive Q4_0 heuristics** (17 → 15.7 tok/s) - Wrong opt combinations
5. ❌ **General GROUP heuristic** (17 → 12 tok/s) - Too aggressive, hurt performance

## What We Learned

**Attention bottleneck (r_288_... @ 5 GB/s) was OVERESTIMATED:**
- That 186ms kernel appears in warmup (includes compilation + weight loading)  
- Actual per-token attention is smaller portion

**Real issue: Q4_0 expert kernels at 38-56 GB/s vs 145 GB/s target**
- These are the consistent per-token bottlenecks
- Need 2.6-3.8x improvement
- Current heuristics (MATVEC) apply but don't achieve target bandwidth

## Next Steps

Focus on improving Q4_0 expert kernel bandwidth from 38-56 GB/s → 145 GB/s:
1. Understand why `r_96_16_4_4_128_2048` has scattered memory access (strides +8192, +16384, +24576)
2. Fix the memory layout or access pattern
3. Test with toy_q4_expert.py for fast iteration
