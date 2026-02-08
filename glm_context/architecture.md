# Architecture Reference

## MLA (Multi-head Latent Attention)

### Three Formulations

#### 1. Naive / "Expand at Cache Time" (HuggingFace)
Expands compressed latent to full multi-head K and V **before** caching:
```
compressed_kv = kv_a_proj(hidden)           → [B,T,kv_lora+rope]
kv_expanded = kv_b_proj(kv_a_norm(k_nope))  → LARGE: [kv_lora] → [n_heads*(nope+v)]
k, v = split(kv_expanded)                    → full multi-head K and V
cache_k, cache_v = update(k, v)              → TWO caches, full-size
attn = sdpa(q, cache_k, cache_v)
```
Cache per token: `n_heads*(nope+rope+v)` = 5120 for DS2-Lite. Expensive kv_b_proj every token.

#### 2. Absorbed / "Compress at Cache Time" (llama.cpp, tinygrad, DeepSeek official)
Absorbs K/V projections into Q and output. Cache stores only compressed latent.
```
compressed_kv = kv_a_proj(hidden)            → [B,T,kv_lora+rope]
kv_normed, k_pe = split(compressed_kv)
q_absorbed = q_nope @ W_k_b.T               → q in latent space
q = cat(q_absorbed, rope(q_pe))              → [B,heads,T,kv_lora+rope] (MQA!)
k = cat(kv_normed, rope(k_pe))              → [B,1,T,kv_lora+rope]
cache_k = update(k)                          → SINGLE cache, no V buffer
v = cache_k[:,:,:,:kv_lora_rank]             → V = first kv_lora dims of K
scores = q @ cache_k.T * scale
attn = softmax(scores) @ v                   → [B,heads,T,kv_lora]
out = attn @ W_v_b.T                        → decompress to v_head_dim
```
Cache per token: `kv_lora_rank + rope` = 576 for DS2-Lite (**8.9x smaller**). No kv_b_proj.

#### 3. Split-Scores (DeepSeek reference alternative)
```
scores = einsum("bshc,btc->bsht", q_nope_absorbed, kv_cache)   # nope part
       + einsum("bshr,btr->bsht", q_pe, pe_cache)              # rope part
```
Two matmuls + add instead of one on concatenated vectors. **Worse for batch=1 decode** (more kernels).

### Comparison for Batch=1 Decode

| Aspect | Naive (expand) | Absorbed (compress) |
|--------|---------------|---------------------|
| Cache memory/token | n_heads*(nope+rope+v) | kv_lora+rope |
| Cache writes/token | 2 (K+V) | 1 (K only) |
| kv_b_proj matmul | Every token (LARGE) | Never |
| Q@K pattern | MHA (n_heads K heads) | MQA (1 K head) |

**Winner**: Absorbed. Smaller cache, fewer bytes read, MQA attention is cheaper.

### Current tinygrad Implementation (mla.py)
- Single K cache (`cache_k`), V is slice of K
- K absorbed into Q via `q_nope @ attn_k_b.weight.T`
- V absorbed post-attention via `attn @ attn_v_b.weight.T`
- fp16 RoPE (no float32 round-trip)
- Max-free softmax in float32 for T=1 decode

### MLA Weight Shapes (from GGUF)
```
attn_k_b.weight:     (H, kv_lora_rank, qk_nope_head_dim)   e.g. (20, 512, 192)
attn_v_b.weight:     (H, v_head_dim, kv_lora_rank)          e.g. (20, 256, 512)
attn_q_a.weight:     (q_lora_rank, dim)                      e.g. (768, 2048)
attn_q_b.weight:     (H*q_head_dim, q_lora_rank)             e.g. (5120, 768)
attn_kv_a_mqa.weight:(kv_lora_rank+rope, dim)                e.g. (576, 2048)
attn_output.weight:  (dim, H*v_head_dim)                      e.g. (2048, 5120)
```

### MLA Variant Differences
| | youtu-llm:2b | deepseek-v2-lite | GLM-4.7-Flash |
|--|-------------|------------------|---------------|
| q_lora_rank | 1536 | 0 (direct proj) | 768 |
| kv_lora_rank | 512 | 512 | 512 |
| qk_nope | 128 | 128 | 192 |
| qk_rope | 64 | 64 | 64 |
| v_head_dim | 128 | 128 | 256 |
| n_heads | 16 | 16 | 20 |
| attn_kv_b | separate k_b/v_b | combined kv_b (split at load) | separate k_b/v_b |

---

## MoE (Mixture of Experts) Routing

### Architecture
- **GLM-4.7**: 64 experts, 4 selected/token, 1 shared expert, sigmoid gating, 1.8x scale
- **deepseek-v2-lite**: 64 experts, 6 selected/token, 2 shared, sigmoid gating
- Layer 0 is always dense FFN (`n_layer_dense_lead=1`), remaining layers are MoE

### Gating Functions (from GGUF `expert_gating_func`)
| Value | Function | Used by |
|-------|----------|---------|
| 1 | Softmax | — |
| 2 | Sigmoid + normalize + scale | GLM-4.7, deepseek-v2-lite |
| 3 | Softmax-weight (softmax on selected) | — |

### Expert Weight Layout

All current MoE models use `expert_first_in_memory = True` (expert-contiguous):
```
Memory: [expert0_blocks][expert1_blocks][expert2_blocks]...
```

Detected automatically in `tinygrad/nn/state.py:543` from GGUF dimension order:
```python
expert_first_in_memory = (expert_dim_idx == 2)  # expert dim last in GGUF = first in memory
```

| Model | expert_first_in_memory | Expert Weights |
|-------|----------------------|----------------|
| deepseek-v2-lite | True | 52 instances |
| glm-4.7:flash | True | 138 instances |

### MoE Fusion Behavior in tinygrad

**Without `.contiguous()`**: dequant + matmul + weighted sum + shared expert + residual → ONE kernel → 17.2 tok/s. Terrible parallelism because compiler picks ONE threadgroup size.

**With `.contiguous()` after down proj**: dequant + matmul → fused kernel. Shared expert → separate. → 26.5 tok/s. The `.contiguous()` breaks fusion with shared expert but NOT the matmul↔weighted_sum fusion (scheduler's `found_contiguous` elides `CONTIGUOUS(RESHAPE(...))`).

**Fused MoE kernel shape**: `[2048, 6, 2816]` with REDUCE=[6, 2816]. GROUPTOP(1,16) gives 16 threads. The outer MUL is `weight_result * routing`, not `activation * weight`. The 2816 reduce dim is in the INNER computation (dequant+matmul).

### GLM MoE Weight Types

| Layer | GGML Type | Bytes/Block | Notes |
|-------|-----------|-------------|-------|
| ffn_gate_exps | Q4_K (12) | 144 | 260 tensors total |
| ffn_up_exps | Q4_K (12) | 144 | |
| ffn_down_exps | Q6_K (14) | 210 | 70 tensors total |
| ffn_gate_shexp | Q5_K (13) | 176 | 92 tensors, small (199MB) |
| ffn_up_shexp | Q5_K (13) | 176 | |

---

## Quantization Formats

### Q4_0 (Simple, fast)
- **18 bytes/block** → 32 elements (0.5625 bytes/element)
- Layout: `[d: fp16][packed: 16 × uint8]`
- Formula: `val = d * (nibble - 8)` where nibble = lo/hi 4 bits of packed byte
- Packed-dot: `(nib - 8) * x` per pair, scale applied per block
- **Faster than Q4_K for MoE** (+11-22%) because simpler dequant fuses better

### Q4_K (Hierarchical, accurate)
- **144 bytes/block** → 256 elements (0.5625 bytes/element, same compression)
- Layout: `[d: fp16][dmin: fp16][scales: 12 bytes][qs: 128 × uint8]`
- 8 sub-groups of 32 elements, each with 6-bit scale and min
- Formula: `val = d * sc[group] * q[i] - dmin * mn[group]`
- Scale unpacking is complex (3 bytes encode 2 packed 6-bit values)

### Q6_K (High quality)
- **210 bytes/block** → 256 elements (0.82 bytes/element)
- Layout: `[ql: 128][qh: 64][scales: 16 × int8][d: fp16]`
- 6 bits per element: ql gives low 4 bits, qh gives high 2 bits (interleaved)
- Formula: `val = d * ((ql | (qh << 4)) - 32) * scale[k/16]`

### Q5_K
- **176 bytes/block** → 256 elements (0.6875 bytes/element)
- Similar structure to Q4_K but with extra high-bit byte per element

### Compression Ratio Impact on Fusion Strategy

| Format | Compression | Fused BW | Split BW | Winner |
|--------|------------|----------|----------|--------|
| Q4_K | 3.56x | ~14 GB/s | ~60 GB/s | **Fused** (3.56 × 14 ≈ 50 > bandwidth save) |
| Q6_K | 2.44x | ~2 GB/s | ~60 GB/s | **Split** (2.44 × 2 ≈ 5 << 60) |
| Q5_K | 2.91x | ~3 GB/s | ~60 GB/s | **Split** |

---

## Per-Block Kernel Anatomy

### youtu-llm:2b-Q4_0 (18 kernels × 32 blocks + 10 overhead = 586)

Each block has ONE scheduling barrier: `cache_k.assign(k_new).realize()` mid-attention.

**Region 1 (7 kernels): Q/K computation → cache write**
```
r_16_128n1         — attn_norm RMSNorm reduce (→1 scalar)                    14us
r_1536_16_4_16     — FUSED: attn_q_a + attn_kv_a Q4_0 dequant+matmul       100us 162GB/s
r_16_96            — q_a_norm RMSNorm reduce (→1 scalar)                      12us
r_32_32_3_48_16    — attn_q_b Q4_0 dequant+matmul                           192us  63GB/s
r_2_18_8_16_2_32_4 — K absorption: q_nope @ k_b^T + RoPE cat                 34us  64GB/s
E_9_32_2n1         — kv_a_norm ewise + RoPE + cat → k_new                    11us
E_9_16_4n1         — cache assign: copy k_new into cache[start_pos]            9us
```

**Region 2 (11 kernels): attention + FFN**
```
r_16_32            — kv_a_norm RMSNorm reduce (→1 scalar)                    11us
r_16_sp_16_36      — QK matmul (attention scores)                             13us
r_16_sp            — softmax exp().sum()                                       9us
r_8_4_16_4_4_sp    — softmax_div + attn@V + V_absorb fused                   11us
r_2048_16_32n1     — attn_output Q4_0 dequant+matmul                         41us  52GB/s
r_2048_16_4_16     — residual + Q4_0 matmul (fused add)                      87us 152GB/s
r_16_128           — ffn_norm RMSNorm reduce (→1 scalar)                      16us
r_64_32_3_...      — FUSED: ffn_gate+silu * ffn_up Q4_0                     545us 118GB/s
r_2048_16_12_16    — ffn_down Q4_0 + residual                                335us 104GB/s
r_16_128n1         — next block's attn_norm reduce                            14us
```

**What fuses well**: RMSNorm elementwise INTO Q4_0 matmuls, attn_q_a + attn_kv_a into one kernel, ffn_gate + silu + ffn_up into one kernel, residual adds into Q4_0 matmuls.

**What can't fuse**: RMSNorm reduces (→1 scalar, different output shape), softmax sum (reduce dependency), cache assign (necessary for KV cache).

### GLM-4.7-Flash (37 kernels × 46 MoE blocks + dense block = ~1703)

Each MoE block: 28-kernel main schedule + 4-kernel tail + custom MSL runners.

**Attention phase (19 kernels)**: q_a norm, q_b projection, kv_a projection + norm, RoPE, QK + softmax, V matmul, output projection

**MoE phase (18 kernels)**: FFN norm, router matmul + sigmoid, shared expert fused, pairwise topk (3 kernels), expert gather (3), expert gate·silu·up fused, routing norm, expert down, MoE output + residual

**Three killer kernels** (per block, warmup times):
| Kernel | Time | BW | What |
|--------|------|----|------|
| Expert gate·up+silu (Q4_0 fused) | 1115us | 13 GB/s | THE bottleneck |
| Shared expert gate·up+silu | 800us | 16 GB/s | Fused dequant |
| Attn KV_A projection | 720us | 4 GB/s | Fused dequant |

---

## Scheduling Structure

### Cache `.realize()` HELPS Scheduling
- **Without barriers** (full graph): scheduler produces 420-548 kernels for 32 blocks (graph too complex)
- **With barriers** (current): each block is independent 18-kernel problem → 586 total
- **Raw graph (no barriers, no JIT)**: schedules to just **15 kernels** — but not achievable with autoregressive decode

### Cross-Block Scheduling
The scheduler prefetches next block's initial ops (norm, kv_a, rope, cache assign) within the current block's schedule call. The 18 kernels/layer includes 5 from the next block.

### YaRN (Yet another RoPE extensioN)
Extended context via RoPE frequency scaling. Key indicators:
- `rope.scaling.type == "yarn"` or `rope.scaling.factor > 1`
- Uses `precompute_freqs_cis_yarn()` for frequency computation
- `mscale` adjusts attention scaling: `scale = mscale² / sqrt(cache_dim)`
- For GLM-4.7: cache_dim = kv_lora_rank + qk_rope_head_dim = 576, mscale = 1.4046
