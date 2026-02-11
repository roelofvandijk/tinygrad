# GLM Optimization Analysis

## Current Performance (Q4_K_M baseline)
- **Baseline**: 19.43 tok/s (51.5ms/token)
- **Model**: GLM-4.7-Flash Q4_K_M, 47 blocks (1 dense + 46 MoE)
- **Kernels**: ~1358 kernels/token, 6 ICBs

## Top Bottlenecks (from debug2_baseline.log)

### MoE Operations (54% of per-token time)
1. **Shared expert gate+silu*up** (22%): 370us/call, 34 GB/s
   - Kernel: `r_96_16_4_4_128_2048`
   - Code: `self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))`
   - Problem: Poor parallelism, low bandwidth

2. **Expert gather gate_up** (21%): 356us/call, 160 GB/s
   - Kernel: `E_36864_32_4_3`
   - Code: `self._expert_blocks[sel]` fancy indexing
   - Problem: Pure data copy waste

3. **Expert Q4_0 gate_up matmul** (16%): 278us/call, 51 GB/s
   - Kernel: `r_768_16_4_4_4_16`
   - Code: Inline Q4_0 dequant+matmul for 4 selected experts
   - Problem: Low bandwidth

4. **Expert gather down_proj** (9%): 167us/call, 127 GB/s
   - Pure data copy for down_proj weights

5. **Expert Q4_0 down matmul** (8%): 153us/call, 47 GB/s

**Expert gathers alone = 30% of total time (pure waste)**

### Attention Operations
- **attn_output Q4_0** (9%): 151us/call, 93 GB/s - decent bandwidth
- **attn_q_b matmul** (4%): 67us/call
- **K absorb matmul** (3%): 52us/call
- **V absorb** (3%): 57us/call

### Small Operations (~14%)
- RMSNorm reduces, softmax, topk, cache ops

## Tested Optimizations (from benchmark logs)

### 1. both_fused: 23.95 tok/s (+23.3% vs baseline)
- **File**: bench_both_fused.log
- **Speedup**: 19.43 → 23.95 tok/s
- **What**: Unknown - likely attention path optimization
- **Hypothesis**: Fuse q_nope and q_pe operations in decode path (lines 98-106 in mla.py)

### 2. split_shexp: 20.95 tok/s (+7.8% vs baseline)
- **Files**: glm_split_shexp.log, profile_glm_split_shexp.log
- **Speedup**: 19.43 → 20.95 tok/s (or 17.40 → 19.52 in profile)
- **What**: Split shared expert FFN computation
- **Code**: Lines 145-149 in mla.py
- **Status**: ✅ IMPLEMENTED

### 3. merged: 21.21 tok/s (+9.2% vs baseline)
- **File**: bench_merged.log
- **Speedup**: 19.43 → 21.21 tok/s
- **What**: Unknown - likely scheduler-level kernel merging

### 4. kernel_opt: 20.75 tok/s (+6.8% vs baseline)
- **File**: bench_kernel_opt.log
- **Speedup**: 19.43 → 20.75 tok/s
- **What**: Unknown - likely kernel-level optimizations

### 5. FOLD_ON: 19.18 tok/s (-9.5% vs fold_off)
- **Files**: bench_q4_0_fold_on.log vs bench_q4_0_fold_off.log
- **Impact**: 21.20 → 19.18 tok/s with FOLD_ON enabled
- **What**: Unknown optimization that HURTS performance
- **Action**: Should be DISABLED for GLM

### 6. group5: 27.08 tok/s (+56% vs baseline, but risky)
- **File**: profile_group5.log
- **Speedup**: 17.40 → 27.08 tok/s, 1404 → 780 kernels/token
- **What**: Aggressive GROUP schedule optimization
- **Risk**: Metal threadgroup memory limits (32KB), needs careful gating

## Implementation Priority

### Tier 1: Quick Wins (1-2 days)
1. ✅ **split_shexp** - Done, testing needed
2. ❓ **both_fused** - Need to identify what it is (+23%)
3. ❓ **merged** - Need to identify (+9%)
4. ❓ **kernel_opt** - Need to identify (+7%)
5. ❓ **FOLD_ON=0** - Need to find and disable (-9.5% when enabled)

### Tier 2: High-Upside (1-2 weeks)
6. **group5 with guards** - Aggressive fusion with Metal memory checks (+56%)
7. **Prepacked expert layout** - Eliminate 30% expert gather overhead
8. **Fused gather/dequant/matmul** - Reduce MoE dispatch from 54% → ~25%

## Next Steps
1. Run benchmark with split_shexp to verify speedup
2. Analyze attention code (lines 88-125) to understand "both_fused"
3. Search scheduler code for potential "merged" and "kernel_opt" optimizations
4. Find what "FOLD_ON" controls and disable it
