# MLA (Multi-head Latent Attention) Implementation Notes

## Three Formulations

### 1. Naive / "Expand at Cache Time" (HuggingFace Transformers)

Expands compressed latent to full multi-head K and V **before** caching:

```python
# HuggingFace DeepseekV2Attention
compressed_kv = kv_a_proj(hidden_states)                    # [B,T,kv_lora+rope]
k_nope, k_pe = split(compressed_kv)
kv_expanded = kv_b_proj(kv_a_norm(k_nope))                  # LARGE: [kv_lora] -> [n_heads*(nope+v)]
k_nope, v = split(kv_expanded)                              # full multi-head K and V
k = cat(k_nope, rope(k_pe))
cache_k, cache_v = update_cache(k, v)                       # TWO caches, full-size
attn = sdpa(q, cache_k, cache_v)
out = o_proj(attn)
```

**Cache per token**: `n_heads * (nope + rope + v_head)` = 5120 elements for DS2-Lite
**Extra cost**: kv_b_proj is a big matmul every token. Two separate cache writes.

### 2. Absorbed / "Compress at Cache Time" (llama.cpp, tinygrad, DeepSeek official)

Absorbs K/V projections into Q and output. Cache stores only compressed latent.

```python
# llama.cpp / tinygrad approach
compressed_kv = kv_a_proj(hidden_states)                    # [B,T,kv_lora+rope]
kv_normed, k_pe = split(compressed_kv)
kv_normed = kv_a_norm(kv_normed)

# ABSORB K into Q (project q into latent space)
q_absorbed = q_nope @ W_k_b.T                              # q in latent space
q = cat(q_absorbed, rope(q_pe))                             # [B,heads,T,kv_lora+rope]
k = cat(kv_normed, rope(k_pe))                              # [B,1,T,kv_lora+rope] (MQA!)

# SINGLE cache, no V buffer
cache_k = update_cache(k)                                   # only K, V is view of K
v = cache_k[:,:,:,:kv_lora_rank]                            # V = first kv_lora dims of K

# Standard attention
scores = q @ cache_k.T * scale
attn = softmax(scores) @ v                                  # [B,heads,T,kv_lora]

# ABSORB V projection
out = attn @ W_v_b.T                                       # decompress to v_head_dim
out = o_proj(out)
```

**Cache per token**: `kv_lora_rank + rope` = 576 elements for DS2-Lite (8.9x smaller!)
**No kv_b_proj expansion**. Two small matmul absorptions instead.

### 3. DeepSeek Reference Split-Scores (alternative absorbed)

```python
# DeepSeek-V3 inference/model.py
# Splits attention score into nope + rope parts
scores = einsum("bshc,btc->bsht", q_nope_absorbed, kv_cache) +  # nope
         einsum("bshr,btr->bsht", q_pe, pe_cache)               # rope
```

Two matmuls + add instead of one matmul on concatenated vectors.
**Worse for batch=1 decode** (more kernels). Only useful if caches need different precision.

## Comparison for Batch=1 Single-Token Decode

| Aspect | Naive (expand) | Absorbed (compress) |
|--------|---------------|---------------------|
| Cache memory/token | n_heads*(nope+rope+v) | kv_lora+rope |
| Cache writes/token | 2 (K+V) | 1 (K only) |
| kv_b_proj matmul | Every token (LARGE) | Never |
| Absorption matmuls | None | 2 small matmuls |
| Q@K pattern | MHA (n_heads K heads) | MQA (1 K head) |
| Total heavy matmuls | 6 | 7 (but 2 are small) |

**Winner**: Absorbed. Smaller cache, fewer bytes read, MQA attention is cheaper.

## llama.cpp Key Implementation Details

From `reference_material/llama.cpp/src/models/deepseek2.cpp`:

1. **No V buffer at all**: `has_v = !is_mla`. V is a view of K's first kv_lora_rank dims.
2. **K absorption**: `q_nope_absorbed = mul_mat(wk_b, q_nope)` — standard matmul, not einsum.
3. **V absorption**: Post-attention `kqv = mul_mat(v_mla, kqv)` in `build_attn_mha`.
4. **MQA pattern**: K has shape `[kv_lora+rope, 1, T]`, Q has `[kv_lora+rope, n_heads, T]`.

## Current tinygrad Implementation (mla.py)

Matches llama.cpp absorbed approach:
- Single K cache (`cache_k`), V is slice of K
- K absorbed into Q via `q_nope @ attn_k_b.weight.T` (matmul, not einsum)
- V absorbed post-attention via `attn @ attn_v_b.weight.T` (matmul, not einsum)
- fp16 RoPE (no float32 round-trip)
- Max-free softmax in float32 for T=1 decode

## Per-Block Kernel Anatomy (youtu-llm:2b-Q4_0, 586 kernels / 32 blocks = 18.3/block)

### Scheduling Structure

Each block has **2 scheduling barriers**:
1. `cache_k.assign(k_new).realize()` — mid-attention cache write
2. `.contiguous()` at end of `__call__` — block output barrier

This splits each block into 2 scheduling regions. The JIT calls `.schedule()` per region, producing ~18 kernels per block.

### The 18 Kernels Per Block

Annotated from DEBUG=2 kernel trace:

```
# Region 1: Before cache realize (Q/K computation + cache write)
 1. r_16_128n1         — residual_add + attn_norm RMSNorm (reduce dim=2048)        ~14us
 2. r_1536_16_4_16     — Q4_0 fused: attn_q_a + attn_kv_a dequant+matmul          ~100us  162 GB/s
 3. r_16_96            — q_a_norm RMSNorm (reduce dim=768)                          ~12us
 4. r_32_32_3_48_16    — Q4_0 dequant+matmul: attn_q_b (Q_LORA→heads*Q_HEAD)      ~192us  63 GB/s
 5. r_2_18_8_16_2_32_4 — K absorption: q_nope @ k_b^T + RoPE cat                   ~34us  64 GB/s
 6. r_576_16_4_16      — kv_a_norm + cache write (kv_normed.cat(k_pe) → cache)     ~47us  128 GB/s
 7. r_16_32            — kv_a_norm RMSNorm (reduce dim=512)                         ~11us

# SCHEDULING BARRIER: cache_k.assign(k_new).realize()

# Region 2: After cache realize (attention + FFN + output)
 8. r_16_start_pos_16_36  — QK matmul (attention scores, varies with context)       ~13us
 9. r_16_start_pos        — softmax exp().sum() (denominator)                        ~9us
10. r_8_4_16_4_4_sp       — softmax_div + attn@V + V_absorb (matmul chain)          ~11us
11. r_2048_16_32n1        — Q4_0 dequant+matmul: attn_output                        ~41us  52 GB/s
12. r_2048_16_4_16        — ??? (appears to be another Q4_0 matmul, investigating)   ~87us  152 GB/s
13. r_16_128              — ffn_norm RMSNorm (reduce dim=2048)                       ~16us
14. r_64_32_3_64_16_64_16 — Q4_0 fused: ffn_gate+silu * ffn_up (FUSED!)           ~545us  118 GB/s
15. r_2048_16_12_16       — Q4_0 dequant+matmul: ffn_down + residual               ~335us  104 GB/s
16. r_16_128n1            — next block norm OR output norm                           ~14us

# SCHEDULING BARRIER: .contiguous()

17. E_9_32_2n1            — contiguous elementwise copy (576 elements)               ~11us
18. E_9_16_4n1            — contiguous elementwise copy (576 elements)                ~9us
```

### Kernel Categories Per Token

| Category | Count/tok | % Time | Notes |
|----------|-----------|--------|-------|
| Q4_0 matmuls | 237 | 83% | Main computation, well-optimized (60-160 GB/s) |
| Elementwise | 137 | 12.5% | Contiguous copies, cache ops, dispatch-dominated |
| Small reduces | 213 | 4.6% | Norms, softmax — tiny kernels, launch-overhead dominated |

### Critical Insight: Cache `.realize()` HELPS Scheduling

**Without realize barriers** (full graph), the scheduler produces **420-548 kernels** for 32 blocks
because the graph is too complex for it to find optimal fusions.

**With realize barriers** (current), each block becomes an independent 18-kernel scheduling
problem. The JIT produces 586 total kernels via 32 × ~18.3 kernel scheduling calls.

**The raw graph (no barriers, no JIT) schedules to just 15 kernels!** But this is only achievable
in `.schedule()` mode, not through JIT (which must execute cache writes between blocks).

### Tested Approaches

| Approach | Kernels | tok/s | Notes |
|----------|---------|-------|-------|
| Current (barriers + contiguous) | 586 | 51.0 | Baseline |
| No `.contiguous()` | 3624 | 3.7 | Scheduler partitions cross-block graph poorly |
| Deferred cache + contiguous | 548 | — | `.cat()` prevents some fusions, only -6.5% |
| Deferred cache + no contiguous | 548 | — | 63s to schedule, no benefit |
| No cache (theoretical) | 15 | — | Not achievable with autoregressive decode |

### Why Cache Restructuring (cat approach) Doesn't Help

Replacing `cache.assign().realize() → cache.read()` with `cache.read_old().cat(k_new)`:
- Removes 1 scheduling barrier per block
- But `.cat()` introduces scheduling complexity
- Net result: 548 vs 586 kernels (-6.5%, ~38 fewer)
- The cat creates extra intermediate buffers that prevent fusions
- The 2 contiguous elementwise kernels (E_9_32_2n1, E_9_16_4n1) persist either way

## Performance Status

| Metric | Value |
|--------|-------|
| Model | youtu-llm:2b-Q4_0 (DeepSeek-V2 MLA, 32 blocks) |
| Params | 0.69 GB |
| Theoretical limit | 145 tok/s (100 GB/s bandwidth) |
| Current | **51.0 tok/s** (19.6ms/tok) |
| Kernels/token | 586 in 1 ICB |
| Avg overhead/kernel | 33.5us (execution + dispatch) |
| Bottleneck | Dispatch overhead, not bandwidth |

### To reach 80 tok/s (12.5ms/tok)

Need ~380 kernels/token (586 → 380 = -35%). That means cutting ~6.4 kernels per block (18 → 12).

Candidates:
- 2 contiguous elementwise (E_9_32_2n1, E_9_16_4n1): **64 kernels saved if removable**
- 4 RMSNorm reduces per block: **128 kernels if any can fuse with consumers**
- 1 softmax sum kernel (r_16_2): **32 kernels if fusable into attention**
- Kernel #12 (mystery extra Q4_0 matmul): investigate if eliminable

### Path Forward

1. **Understand the extra Q4_0 matmul (kernel #12)** — what creates `r_2048_16_4_16`? Is it a duplicate?
2. **Fuse RMSNorm into consumer** — if the norm reduce can fuse with the following matmul, saves 128 kernels
3. **Eliminate `.contiguous()` overhead** — the 2 elementwise kernels per block are pure waste if a different barrier mechanism works
4. **Teach scheduler about cross-block patterns** — the raw graph schedules to 15 kernels; if JIT could batch scheduling regions, massive savings possible
