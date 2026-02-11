# split_shexp Optimization Results

## Q4_0 Model (SUCCESSFUL)
- **Baseline** (profile_current.log): 17.40 tok/s
- **With split_shexp** (profile_glm_split_shexp.log): 19.52 tok/s  
- **Speedup**: +12.2% ✅

## Q4_K_M Model (NO EFFECT)
- **Baseline** (profile_fold_off.log): 9.57 tok/s
- **With split_shexp** (current_profile.log): 9.60 tok/s
- **Speedup**: +0.3% (within noise)

## Conclusion
- split_shexp optimization works for Q4_0 quantization
- No benefit for Q4_K_M (different bottlenecks)
- Q4_0 uses inline dequant kernels where fusion helps
- Q4_K_M uses custom mul_mat_id kernels (already optimized)

## Implementation Status
✅ Code change in mla.py lines 145-149 is CORRECT and BENEFICIAL for Q4_0
