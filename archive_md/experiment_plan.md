# Systematic Experiment Plan to Reach 24+ tok/s

Current: **20 tok/s** (with merge_gate_up optimization)
Target: **24+ tok/s** (matching bench_both_fused.log: 23.95 tok/s)

## Bottleneck Analysis (from glm_kernel_map.md)

### Top Time Consumers
1. Shared expert gate+silu*up: 22% (370us, **34 GB/s** - poor)
2. Expert gather: 21% (already optimized)
3. Expert matmul: 16% (278us, **51 GB/s** - poor)
4. Attention ops: ~15-20%

## Experiment 1: Remove Contiguous Calls (Test "both_fused")
**Hypothesis**: `.contiguous()` calls prevent scheduler fusion
**Expected**: +5-15% if scheduler can fuse more ops

### Changes to test:
```python
# mla.py line 141
# CURRENT:
gated = (gate.silu() * up).contiguous()

# TEST A: Remove contiguous
gated = gate.silu() * up

# mla.py line 142
# CURRENT:
expert_out = self.ffn_down_exps(sel, gated).contiguous()

# TEST B: Remove contiguous
expert_out = self.ffn_down_exps(sel, gated)

# mla.py line 149
# CURRENT:
out = out.contiguous() + shexp_out

# TEST C: Remove contiguous
out = out + shexp_out
```

**Test command:**
```bash
python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > exp1_no_contig.log 2>&1
```

## Experiment 2: Split Shared Expert (Test "split_shexp")
**Hypothesis**: Separating gate/up allows better fusion
**Expected**: +5-10%

### Changes:
```python
# mla.py line 145-146
# CURRENT (one line):
out = out.contiguous() + self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))

# TEST: Split into separate ops
shexp_gate = self.ffn_gate_shexp(h_norm).silu()
shexp_up = self.ffn_up_shexp(h_norm)
shexp_out = self.ffn_down_shexp(shexp_gate * shexp_up)
out = out + shexp_out
```

## Experiment 3: Attention Path Fusion (Test "both_fused")
**Hypothesis**: q_nope and q_pe split/cat creates overhead
**Expected**: +10-20%

### Changes to test:
```python
# mla.py lines 98-106
# CURRENT:
q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
# ... transforms ...
q = q_nope.cat(q_pe, dim=-1)

# TEST A: Avoid intermediate split
# (compute on full q, slice only when needed)

# TEST B: Decode-specific path
if T == 1:
    # Optimized decode path without split/cat
    pass
else:
    # Prefill path (keep current)
    pass
```

## Experiment 4: Remove Realize() Calls
**Hypothesis**: Explicit realize() prevents lazy fusion
**Expected**: +5-10%

### Changes:
```python
# mla.py line 113
# CURRENT:
self.cache_k[:, :, start_pos:start_pos+T, :].assign(k_new).realize()

# TEST: Remove realize
self.cache_k[:, :, start_pos:start_pos+T, :].assign(k_new)
```

## Experiment 5: Combine Multiple Optimizations
**Once individual gains are measured, combine the winners:**

1. Best of Exp 1 (contiguous removal)
2. Best of Exp 2 (split shared expert)
3. Best of Exp 3 (attention fusion)
4. Best of Exp 4 (realize removal)

**Expected combined**: 20 → 24+ tok/s

## Testing Protocol

For each experiment:
1. Make ONE change at a time
2. Run: `python3 profile_model.py glm-4.7:flash-unsloth-Q4_0 10 > expN.log 2>&1`
3. Extract steady-state tok/s: `grep "STEADY" expN.log`
4. Compare kernel count: `grep "Kernels:" expN.log`
5. If improvement > 2%, keep it; if regression, revert
6. Document result in results table

## Results Table (to fill in)

| Experiment | Change | Baseline | After | Speedup | Kernel Δ | Keep? |
|------------|--------|----------|-------|---------|----------|-------|
| Baseline   | -      | 20.0     | -     | -       | 1358     | -     |
| Exp1A      | No contig line 141 | 20.0 | ? | ? | ? | ? |
| Exp1B      | No contig line 142 | 20.0 | ? | ? | ? | ? |
| Exp1C      | No contig line 149 | 20.0 | ? | ? | ? | ? |
| Exp2       | Split shexp | 20.0 | ? | ? | ? | ? |
| Exp3A      | Attn no split | 20.0 | ? | ? | ? | ? |
| Exp3B      | Decode fast path | 20.0 | ? | ? | ? | ? |
| Exp4       | No realize | 20.0 | ? | ? | ? | ? |
| Combined   | Best of above | 20.0 | **24+?** | **+20%?** | <1200? | ✅ |

## Priority Order
1. **Exp1 (contiguous)** - Easiest to test, potentially high impact
2. **Exp2 (split shexp)** - Already implemented, quick verification
3. **Exp3 (attention)** - Likely the "both_fused" optimization
4. **Exp4 (realize)** - Minor impact expected
5. **Combined** - Once we know what works
