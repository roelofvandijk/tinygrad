# GLM-4.7-Flash Optimization — What To Do Next

## Current State (Feb 9 2026)

| Model | tok/s | ms/tok | Kernels/tok | Theoretical | Efficiency |
|-------|-------|--------|-------------|-------------|------------|
| youtu-llm:2b-Q4_0 | **56** | 17.8 | 586 | 145 tok/s | 39% |
| deepseek-v2-lite-Q4_0 | **27** | 37 | 780 | — | — |
| GLM-4.7-Flash-Q4_0 | **20.9** | 48 | ~1450 | 80 tok/s | 26% |

GLM: 47 blocks (1 dense + 46 MoE), 64 experts, 4 selected/token, 1 shared expert.
Expert matmuls = **74% of GLM token time** (41.2ms of 55.8ms). Everything else is noise.

---

## Actionable Ideas (ranked by expected impact)

### 1. Custom `mul_mat_id` MSL for Q4_0 Expert Matmuls ★★★

**Status**: TODO — highest-ceiling approach
**Expected**: GLM 21→30-40 tok/s

The Q4_0 packed-dot kernels (gate/up at 45 GB/s, down at 30 GB/s) hit a **~50 GB/s ceiling
from scattered byte reads** in tinygrad-generated code. Each thread reads individual
`unsigned char` values. This is a fundamental codegen limitation — no heuristic change fixes it.

llama.cpp's MSL kernel achieves 145 GB/s via `uint16_t` reads, SIMD reduction, multi-row
amortization, and stride-4 block interleaving. A `mul_mat_id` variant would:
- Read Q4_0 blocks with coalesced `uint16_t` loads
- SIMD-reduce within threadgroup
- Handle expert selection (gather) inside the kernel — eliminate separate gather kernels
- Extend `CompiledRunner` for JIT batching (same pattern as existing Q4K/Q6K runners)

**Files**: Create `tinygrad/nn/metal_mul_mat_id_q4_0.py` modeled on the byte-offset
approach from MEMORY.md. Integrate in [quantized.py](../tinygrad/apps/quantized.py) `QuantizedExpertWeights.__call__`.

**Key requirements**:
- Extend `CompiledRunner`, not `Runner` (JIT batching)
- `ProgramSpec.global_size` must include `z=n_sel` for ICB dispatch
- Use `#define BYTE_OFF` for GGUF offsets (buffer views break JIT)
- Per-layer kernel cached via `lru_cache` (each layer has unique byte offset)

**Estimated savings**: gate+up 14.2ms → ~4.5ms, down 9.8ms → ~2ms = **-17.5ms/tok**

### 2. Fuse Gate+Up Into One Expert Matmul ★★

**Status**: TODO — reduces kernel count and expert weight gathers
**Expected**: ~15-25% MoE speedup

Currently two separate matmuls read the same input and index the same experts:
```python
gate = self.ffn_gate_exps(sel, h_norm)   # (B,T,K,H)
up   = self.ffn_up_exps(sel, h_norm)     # (B,T,K,H)
```

Concatenate weights into `(num_experts, 2*moe_hidden_dim, dim)`, one matmul, split:
```python
gate_up = self.ffn_gate_up_exps(sel, h_norm)  # (B,T,K,2*H)
gate, up = gate_up.chunk(2, dim=-1)
```

Halves expert weight gathers and kernel dispatches. llama.cpp does this.

**Files**: [glm.py](../tinygrad/apps/glm.py) `GLMTransformerBlock._feed_forward`,
[quantized.py](../tinygrad/apps/quantized.py) `QuantizedExpertWeights` (concat at load time).

### 3. Benchmark `Tensor.topk` vs `topk_pairwise` ★

**Status**: TODO — may eliminate 16 lines of custom code
**Expected**: Same or fewer kernels, simpler code

[glm.py:7-18](../tinygrad/apps/glm.py#L7-L18) `topk_pairwise` uses an O(n^2) pairwise
comparison matrix to avoid bitonic sort overhead. It was written because `Tensor.topk` used
to generate 29 kernels via `sort()`. But `Tensor.topk` may now be better optimized.

Benchmark both for n=64, k=4 (GLM's config) and n=64, k=6 (ds2-lite):
```python
# Current: pairwise comparison matrix, O(n^2), fixed ~3 kernels
values, indices = topk_pairwise(scores, k)
# Alternative: may be competitive now
values, indices = scores.topk(k, dim=-1)
```

If `Tensor.topk` is competitive in kernel count, delete `topk_pairwise`.

### 4. Absorb K Projection Into Q Weights at Load Time ★

**Status**: TODO — eliminates one per-head matmul per layer per token
**Expected**: ~5-10% on attention layers

In [glm.py:98](../tinygrad/apps/glm.py#L98):
```python
q_nope = q_nope @ self.attn_k_b.weight.transpose(-1, -2)  # every token!
```

This is `(B, n_heads, T, qk_nope_head_dim) @ (n_heads, qk_nope_head_dim, kv_lora_rank)`.
For T=1 decode, k_b doesn't change. Pre-multiply `attn_q_b[:, :qk_nope] @ k_b^T` once
during model loading.

**Files**: [glm.py](../tinygrad/apps/glm.py) `GLMTransformerBlock.__init__` and
`_attention`.

### 5. Audit `.contiguous()` Calls ★

**Status**: TODO — each unnecessary one adds a kernel dispatch
**Expected**: ~5% fewer kernels per layer

Current `.contiguous()` calls in [glm.py:135-140](../tinygrad/apps/glm.py#L135-L140):
```python
if self.split_moe_boundaries: gate, up = gate.contiguous(), up.contiguous()
...
if self.split_moe_boundaries: expert_out = expert_out.contiguous()
...
if self.split_moe_boundaries: out = out.contiguous()
```

These were added to break problematic fusion. Profile with and without each one.
If idea #1 (custom MSL) handles expert matmuls, some splits become unnecessary.

### 6. Kernel Count Reduction via Scheduler Improvements

**Status**: PARTIALLY DONE — scheduler fusion change landed, more possible
**Expected**: 1450 → ~1200 kernels (reduce dispatch overhead ~17%)

**Done**: indexing.py:236 change (RMSNorm 2→1, softmax 3→2), threadgroup memory guards.

**Remaining targets** (from [bottlenecks.md](bottlenecks.md)):
- Fuse RMSNorm reduce+ewise: -230 kernels (5 norms × 46 blocks)
- Fuse QK + softmax: -46 kernels
- Expert gather consolidation: 4→1 per block = -138 kernels

These require deeper scheduler changes and are medium-difficulty.

---

## Dead Ends (don't retry)

| Idea | Result | Why It Failed |
|------|--------|---------------|
| MV heuristic walk through dequant chains | -1% regression | Fused dequant has MULTIPLE reduce ranges; `ranges_subset` fails in both INDEX orderings |
| GROUPTOP threshold 2048→16384 | **-33%** (12.8 tok/s) | Applied to ALL kernels including well-optimized attention — no selective targeting |
| Metal ICB barrier removal | **0%** | Nearly all kernels data-dependent; would need kernel reordering to help |
| Ushort loads (codegen rewrite) | **-4.8%** | ALU overhead of unpacking ushort → 4 nibbles > bandwidth saving |
| Ushort loads (tensor bitcast) | NaN/inf | Wrong nibble access patterns, shape mismatches |
| MOE_FUSED custom UOp kernels | **6x slower** | Dispatch overhead + scattered byte reads |
| Per-expert dispatch | **-18%** | 6x more kernel dispatches |
| Remove all `.contiguous()` | 3600+ kernels | Scheduler can't partition large graphs |
| BEAM on fused dequant | No help | Can't fix structural problem (serial reduction) |
| Combined q_a+kv_a projection | Same | Split consumption → scheduler duplicates |
| Combined gate+up projection | +1 kernel | Breaks parallel reduce fusion |
| Pre-dequant experts to fp16 | Math says worse | 3.56x bandwidth increase > any fusion benefit |
| Q8 repack for experts | OOM | 64 experts blows RAM |
| FP16 softmax with `.max()` | +32 kernels | Max adds kernels, no speedup |

### Why the MV Heuristic Fix Is Blocked

The MV heuristic ([heuristic.py:67-89](../tinygrad/codegen/opt/heuristic.py#L67-L89))
requires `MUL(INDEX, INDEX)`. Fused dequant creates `MUL(dequant_chain(INDEX), INDEX)`.
Multiple approaches tried to walk backward through the chain:

1. Walk `backward_slice` to find INDEX → finds INDEXes but range checks fail
2. Use `.ranges` instead of `split_uop(Ops.ADD)` → helps but `ranges_subset` still fails
3. Try both orderings of idx0/idx1 → neither passes all checks

**Root cause**: Fused dequant creates MULTIPLE reduce ranges (block index + element-within-block).
Weight INDEX has extra sub-block ranges not in activation INDEX.
`all(r in idx_b.ranges for r in idx_a.ranges)` fails in both orderings.

The current workaround ([heuristic.py:76-78](../tinygrad/codegen/opt/heuristic.py#L76-L78))
skips the range check when weight side isn't INDEX. This lets MV fire on some patterns but
doesn't solve the core mismatch. **Custom MSL (idea #1) is the real fix.**

### Why the Q4_0 GROUP Heuristic Has a ~50 GB/s Ceiling

The heuristic ([heuristic.py:91-110](../tinygrad/codegen/opt/heuristic.py#L91-L110)) detects
dequant kernels via AND/SHR ops and applies GROUP+LOCAL+UPCAST. This gets gate/up kernels to
45-50 GB/s and down to 30-42 GB/s. But tinygrad's codegen generates scattered byte reads
(`unsigned char val = *(data+...)`) — each thread reads individual bytes from different memory
locations. No amount of threadgroup/local optimization fixes this memory access pattern.
Only hand-written MSL with coalesced `uint16_t` reads can break past 50 GB/s.

---

## Code Simplification Ideas

See [ideas.md](ideas.md) and [opus_ideas.md](opus_ideas.md) for full details. Summary:

| # | Idea | Saves | Priority |
|---|------|-------|----------|
| 1 | Merge TransformerBlock + MLATransformerBlock + GLMTransformerBlock | ~60 lines | High |
| 2 | Tokenizer preset logic → data dict | ~20 lines | Medium |
| 3 | Factor shared Q4_0 dequant-matmul between QuantizedLinear/ExpertWeights | ~20 lines | Medium |
| 4 | Move chat HTML/server to separate file | ~72 lines from llm.py | Medium |
| 5 | Remove QL_CACHE_* debug knobs | ~10 lines | Low |
| 6 | Simplify replace_quantized_modules tree walk | ~8 lines | Low |

### GLM-Specific: Three Block Classes → One

Currently three transformer blocks exist:
- `TransformerBlock` ([llm.py:155-231](../tinygrad/apps/llm.py#L155-L231)) — standard attention
- `MLATransformerBlock` ([mla.py](../tinygrad/apps/mla.py)) — MLA attention, MoE FFN
- `GLMTransformerBlock` ([glm.py:35-149](../tinygrad/apps/glm.py#L35-L149)) — MLA attention (GLM-specific), MoE FFN

GLM and MLA blocks share ~80% of their code (MoE FFN, norms, `__call__`). The only difference
is attention details (GLM has `apply_rope_interleaved`, different cache structure, per-head
weights as separate class). A single class with attention as a strategy would eliminate ~100
lines total.

---

## Performance Numbers to Remember

| Metric | Value | Source |
|--------|-------|--------|
| M2/M3 memory bandwidth | ~100 GB/s | Apple spec |
| M3 Max memory bandwidth | ~400 GB/s | Apple spec |
| Q4_0 packed-dot ceiling (tinygrad codegen) | ~50 GB/s | Scattered byte reads |
| Q4K MSL dense matvec | 145 GB/s | Hand-written Metal |
| Q4_0 packed-dot (llama.cpp MSL) | 229 GB/s | Hand-written Metal, SIMD |
| GLM active params per token | 1.24 GB | 3B active of 30B total |
| Theoretical GLM tok/s (M2/M3) | ~80 | 1.24 GB / 100 GB/s × overhead |
| Expert kernels per token | ~550 | gate+up+down × 46 blocks |
| Non-expert kernels per token | ~900 | norms, attention, routing, etc. |

---

## Key Insight: The Codegen vs MSL Boundary

tinygrad's compiler generates *reasonable* kernels (50-90 GB/s for simple patterns) but cannot
match hand-written MSL for quantized matmuls (145-229 GB/s). The gap comes from:

1. **Memory access pattern**: Codegen reads bytes one at a time. MSL reads `uint16_t` coalesced.
2. **SIMD utilization**: Codegen uses threadgroup barriers. MSL uses `simd_sum` hardware shuffles.
3. **Multi-row amortization**: Codegen processes one output row per thread. MSL amortizes
   activation reads across multiple rows.

For dense attention matmuls, tinygrad's MV heuristic + codegen achieves 60-90 GB/s —
acceptable. For MoE expert matmuls with 4-64 experts, the gap matters enormously because
these dominate 74% of GLM's token time.

**Strategy**: Use custom MSL only for the critical path (MoE expert matmuls), let tinygrad
handle everything else. This is already the pattern for Q4K/Q6K dense linears.

---

## Quick Reference

```bash
# Smoke test
python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --prompt "Hello" --count 10 > ./smoke.log 2>&1

# Benchmark (20 tokens)
python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --benchmark 20 > ./bench.log 2>&1

# Profile (kernel analysis)
python profile_model.py glm-4.7:flash 10
python profile_model.py glm-4.7:flash 10 --with-source  # top 3 kernels' Metal source

# Debug kernel opts
DEBUG=3 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --benchmark 1 2>&1 | grep MATVEC

# Full Metal source dump
DEBUG=5 python tinygrad/apps/llm.py --model "glm-4.7:flash-Q4_0" --benchmark 3 > debug5.log 2>&1
```

## File Map

| File | Role |
|------|------|
| [apps/llm.py](../tinygrad/apps/llm.py) | Transformer, tokenizer, CLI, server |
| [apps/glm.py](../tinygrad/apps/glm.py) | GLMTransformerBlock, topk_pairwise, ExpertWeights |
| [apps/mla.py](../tinygrad/apps/mla.py) | MLATransformerBlock (ds2-lite/youtu) |
| [apps/quantized.py](../tinygrad/apps/quantized.py) | QuantizedLinear, QuantizedExpertWeights |
| [apps/rope.py](../tinygrad/apps/rope.py) | RoPE + YaRN precomputation |
| [codegen/opt/heuristic.py](../tinygrad/codegen/opt/heuristic.py) | MV, GROUPTOP, Q4_0 GROUP heuristics |
| [schedule/indexing.py](../tinygrad/schedule/indexing.py) | Scheduler fusion decisions |
| [codegen/opt/postrange.py](../tinygrad/codegen/opt/postrange.py) | GROUP shared memory check |

## Related Docs

| Doc | What's in it |
|-----|-------------|
| [START.md](START.md) | Quick start, key findings, what worked/failed |
| [bottlenecks.md](bottlenecks.md) | 3 performance gaps, per-model budgets, full experiment log |
| [architecture.md](architecture.md) | MLA formulations, MoE routing, quant formats, kernel anatomy |
| [glm_hot_kernels.md](glm_hot_kernels.md) | Metal source analysis of top 6 GLM kernels |
| [llama_cpp.md](llama_cpp.md) | llama.cpp's mul_mat_id, ICB barriers, kernel techniques |
| [scheduler_fusion.md](scheduler_fusion.md) | indexing.py:236 change, threadgroup memory fix |
| [ideas.md](ideas.md) | Synthesized code simplification and perf ideas (30 items) |
| [opus_ideas.md](opus_ideas.md) | Detailed Opus analysis of 20 improvement ideas |
| [tools.md](tools.md) | Profiling commands, benchmarking rules, chat templates |
