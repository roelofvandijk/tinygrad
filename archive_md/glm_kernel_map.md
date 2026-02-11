# GLM-4.7-Flash Q4_0 Kernel Map

Source: `debug2_baseline.log` tok 3 (unjitted, individual kernel times)
Model: 47 blocks (1 dense + 46 MoE), 1358 kernels/token, ~49ms/tok steady-state (20.3 tok/s)

## Model Params
dim=2048, n_heads=20, q_lora_rank=768, kv_lora_rank=512, qk_nope=192, qk_rope=64, v_head_dim=256
64 experts, 4 sel/tok, 1 shared expert, moe_hidden=10240
All weights Q4_0 in this GGUF. Q4_0 = 18 bytes/block, 32 el/block.

## One MoE Block (28 kernels, ~1740us)

Kernel sequence from block 2 (kernels 28-55):

### Attention Phase (15 kernels, ~585us)

| # | Kernel | us | GB/s | What | Code |
|---|--------|---:|-----:|------|------|
| 28 | `r_16_128n1` | 14 | 0 | attn_norm RMSNorm reduce | `self.attn_norm(x)` reduce |
| 29 | `r_48_16_4_4_4_16n1` | 43 | 78 | attn_q_a Q4_0 matmul (768,2048) | `self.attn_q_a(x_norm)` |
| 30 | `r_16_48` | 11 | 0 | q_a_norm RMSNorm reduce | `self.attn_q_a_norm(...)` reduce |
| 31 | `r_320_8_4_4_3_16n1` | 63 | 154 | attn_q_b Q4_0 matmul (5120,768) | `self.attn_q_b(...)` |
| 32 | `r_20_18_2_16_4_4_12` | 54 | 84 | K absorb: q_nope@k_b^T + RoPE | `q_nope @ attn_k_b.weight` |
| 33 | `r_20_(sp+1)_16_36` | 12 | 24 | QK matmul (attention scores) | `q @ cache_k.T` |
| 34 | `r_5_4_(sp+1)` | 10 | 0 | softmax exp().sum() | softmax reduce |
| 35 | `r_5_8_16_4_4_(sp+1)` | 15 | 6 | softmax_div + attn@V | `softmax @ cache_v` |
| 36 | `r_320_16_4_4_32n1` | 59 | 89 | V absorb: attn@v_b^T + attn_output Q4_0 | `attn @ attn_v_b.weight` |
| 37 | `r_128_16_4_4_10_16n1` | 145 | 93 | attn_output Q4_0 (2048,5120) + residual | `self.attn_output(...)` + residual |
| 38 | `r_16_128` | 16 | 0 | ffn_norm RMSNorm reduce | `self.ffn_norm(h)` reduce |
| 39 | `r_4_16_4_4_128` | 50 | 11 | router matmul + sigmoid | `h_norm @ ffn_gate_inp.T` + sigmoid |
| 40 | `r_96_16_4_4_128_2048` | 373 | 34 | **shared expert gate+silu*up fused Q4_0** | `ffn_gate_shexp(h_norm).silu() * ffn_up_shexp(h_norm)` |
| 51 | `r_16_128n1` | 14 | 0 | (from shared expert) ffn_norm reduce | shared expert path |
| 52 | `r_36_16_4_4_128n1` | 43 | 55 | (from shared expert) | shared expert path |

### MoE Phase (13 kernels, ~1155us)

| # | Kernel | us | GB/s | What | Code |
|---|--------|---:|-----:|------|------|
| 41 | `r_64_16_4` | 10 | 0 | topk: pairwise comparison ranks | `_topk_pairwise` compare+sum |
| 42 | `r_4_16_4` | 10 | 0 | topk: match*arange sum → indices | `_topk_pairwise` indices |
| 43 | `E_36864_32_4_3` | 313 | 160 | **expert gather gate_up** (4 experts) | `self._expert_blocks[sel]` for gate_up |
| 44 | `E_4` | 11 | 0 | gather routing probs | `gate_scores.gather(-1, sel)` |
| 45 | `r_768_16_4_4_4_16` | 271 | 51 | **expert Q4_0 gate_up matmul** | `ffn_gate_up_exps(sel, h_norm)` dequant+matmul |
| 46 | `r_4` | 7 | 0 | routing prob normalize | `probs / probs.sum().maximum(eps)` |
| 47 | `E_24_4_16_4` | 7 | 4 | silu * up elementwise | `gate.silu() * up` |
| 48 | `r_4_128_16_4_4_96` | 234 | 170 | **expert down_proj matmul** (not Q4_0, fp16 cache) | `ffn_down_exps(sel, gated)` |
| 49 | `r_16_32_4_4` | 8 | 3 | weighted sum over experts | `(expert_out * probs).sum(axis=2)` |
| 50 | `r_128_16_4_4_96` | 69 | 88 | shared expert down Q4_0 + MoE output + residual | `ffn_down_shexp(...)` + residual |
| 51 | `r_16_128n1` | 14 | 0 | next block attn_norm reduce (prefetched) | next block's `self.attn_norm` |
| 52 | `r_36_16_4_4_128n1` | 43 | 55 | next block q_a matmul (prefetched) | next block's `self.attn_q_a` |
| 53 | `r_16_32` | 11 | 0 | next block q_a_norm reduce (prefetched) | next block's `self.attn_q_a_norm` |
| 54 | `E_9_32_2n1` | 11 | 1 | next block RoPE + kv_a_norm + cat | next block's kv preprocessing |
| 55 | `E_9_16_4n1` | 11 | 0 | next block cache assign | next block's cache write |

## Aggregate Per-Token Budget (46 MoE blocks)

| Kernel | Calls/tok | us/call | us/tok | % | What |
|--------|----------:|--------:|-------:|--:|------|
| `r_96_16_4_4_128_2048` | 46 | 370 | **17012** | **22%** | shared expert gate+silu*up Q4_0 |
| `E_36864_32_4_3` | 46 | 356 | **16396** | **21%** | expert gather gate_up |
| `r_768_16_4_4_4_16` | 46 | 278 | **12794** | **16%** | expert Q4_0 gate_up matmul |
| `r_128_16_4_4_10_16` | 47 | 151 | **7098** | **9%** | attn_output Q4_0 (2048,5120) |
| `E_18432_32_4_3` | 42 | 167 | **7021** | **9%** | expert gather down_proj |
| `r_2048_16_2_2_3_16` | 42 | 153 | **6443** | **8%** | expert Q4_0 down matmul |
| `r_9680_16_4_4_128` | 1 | 4663 | 4663 | 6% | dense block 0 FFN (big matmul) |
| `r_640_16_4_4_4_16_64_16` | 1 | 3498 | 3498 | 4% | dense block 0 gate+silu*up Q4_0 |
| `r_128_16_4_4_96` | 46 | 72 | 3306 | 4% | shared expert down + residual |
| `r_320_8_4_4_3_16` | 47 | 67 | 3142 | - | attn_q_b Q4_0 matmul |
| `r_320_16_4_4_32` | 47 | 57 | 2696 | - | V absorb + attn_output |
| `r_20_18_2_16_4_4_12` | 47 | 52 | 2465 | - | K absorb matmul |
| `r_4_16_4_4_128` | 46 | 51 | 2364 | - | router matmul + sigmoid |
| everything else | ~900 | ~15 | ~13500 | - | norms, softmax, topk, cache, etc |
| **TOTAL** | ~1358 | | **~78000** | | (sum of individual kernel times) |

## Top Bottlenecks

1. **Shared expert gate+silu*up** (22%): `r_96_16_4_4_128_2048` — 370us, only 34 GB/s effective. Weight (10240,2048)+(10240,2048) = 2x 10240x2048 Q4_0 fused with silu*up. Low bandwidth suggests poor parallelism.
2. **Expert gather gate_up** (21%): `E_36864_32_4_3` — 356us, 160 GB/s. Pure data copy: `self._expert_blocks[sel]` fancy indexing. 36864 = 4 experts x 9216 blocks? (merged gate+up: 20480 out / 32 el_per_block * 18 bytes = 11520 bytes/expert-row... 4*9216=36864 elements copied).
3. **Expert Q4_0 gate_up matmul** (16%): `r_768_16_4_4_4_16` — 278us, 51 GB/s. Inline Q4_0 dequant+matmul for 4 selected experts. Low bandwidth.
4. **attn_output Q4_0** (9%): `r_128_16_4_4_10_16` — 151us, 93 GB/s. Weight (2048,5120). Decent bandwidth.
5. **Expert gather down_proj** (9%): `E_18432_32_4_3` — 167us, 127 GB/s. Data copy for down_proj weights.
6. **Expert Q4_0 down matmul** (8%): `r_2048_16_2_2_3_16` — 153us, 47 GB/s. Inline Q4_0 dequant+matmul for down_proj.

**MoE total = 21% + 16% + 9% + 8% = 54% of per-token time is MoE expert dispatch.**
**Expert gathers alone = 21% + 9% = 30% is pure data copying.**
