# Synthesized Ideas for GLM/LLM Code Simplification & Performance

*Consolidated from: opus_ideas.md, gemini_ideas.md, gpt51_mini_ideas.md, gpt5mini_ideas.md*

## Context

master `llm.py` is 368 lines. The glm6 branch adds +1,014 lines across 7 files:

| File | Lines | Role |
|------|-------|------|
| `apps/llm.py` | 592 (+224) | Tokenizer, TransformerBlock, model dict, server, CLI |
| `apps/mla.py` | 178 (new) | MLATransformerBlock, _topk_pairwise, MLA GGUF helpers |
| `apps/quantized.py` | 179 (new) | QuantizedLinear, QuantizedExpertWeights, replace_quantized_modules |
| `apps/rope.py` | 99 (new) | RoPE + YaRN precomputation and application |
| `apps/smoke_test.py` | 113 (new) | Multi-model regression test |
| `codegen/opt/heuristic.py` | 220 (+63) | MV dequant-chain fix, Q4_0 GROUP heuristic |
| `nn/state.py` | 467 (+123) | dequantize_q4k/q5k/q6k/q4_0, GGML_QUANT_INFO, quantized gguf_load |

---

## A. Cut Lines in Half

### 1. Merge TransformerBlock and MLATransformerBlock
*Sources: opus, gemini, gpt51-mini*

`TransformerBlock` (llm.py:154-230) and `MLATransformerBlock` (mla.py:28-153) duplicate
`_feed_forward`, `__call__`, norm layers, and MoE routing. The only real difference is
`_attention`. Merge into one class with attention as a strategy — either a constructor flag
that selects the attention method, or a small factory keyed by config. The MoE feed-forward
code is nearly identical across both (mla.py:127-150 vs llm.py:218-227) except for gating
function variants and shared experts. A single `_feed_forward` handling the superset works.

**Saves ~60 lines.** Biggest single line-count win.

### 2. Tokenizer Preset Logic → Data Dict
*Sources: opus, gemini, gpt51-mini*

`role()` (llm.py:114-120), `end_turn()` (121-127), `build_chat_ids()` (128-144) have
cascading if/elif chains for each preset. Replace with a `PRESET_CONFIG` dict:

```python
PRESET_CONFIG = {
  "glm4": {"role_fmt": "<|{role}|>\n", "end_turn": [], "prefix": "<sop>", "think": True},
  "qwen2": {"role_fmt": "<|im_start|>{role}\n", "end_turn_has_eos": True, ...},
}
```

Then `role()`, `end_turn()`, `build_chat_ids()` become 3 lines each. Adding new presets
is just data — no new branches.

**Saves ~20 lines.** Also reduces bugs when adding models.

### 3. Delete `_rope_interleaved` from mla.py — Use rope.py
*Sources: opus, gemini*

`mla.py:78-84` defines `_rope_interleaved` that's nearly identical to
`rope.py:55-63` `apply_rope_interleaved`. The mla.py version skips the float32 round-trip,
but `apply_rope_interleaved` casts back to `input_dtype` anyway. Either use it directly or
add a `native_dtype=True` flag.

**Saves ~8 lines** and eliminates a subtle divergence risk.

### 4. Factor Shared Q4_0 Dequant-Matmul
*Sources: opus, gpt5mini*

`QuantizedExpertWeights.__call__` (quantized.py:126-137) duplicates `QuantizedLinear`'s
Q4_0 inline-dequant logic (lines 75-87). Extract a shared `_q4_0_matmul(x_flat, blocks,
out_features, in_features)` function.

**Saves ~20 lines** of copy-pasted bit manipulation.

### 5. Move Chat HTML / Server to Separate File
*Sources: opus, gpt5mini*

The OpenAI-compatible server (llm.py:440-511, 72 lines) including `CHAT_HTML` blob and
`Handler` class is a distinct feature. Move to `apps/serve.py`. This keeps `llm.py` focused
on model definition and generation.

**Saves ~72 lines** from llm.py (moves them, doesn't delete).

### 6. Remove QL_CACHE_* Debug Knobs
*Source: opus*

`quantized.py:56-65` has 10 lines of per-layer-size `getenv` switches that are debugging
aids for one model. Replace with `QL_CACHE_ALL` or remove — the cache fallback at line 88
already handles this.

**Saves ~10 lines.**

### 7. Simplify replace_quantized_modules Tree Walk
*Sources: opus, gemini*

`replace_quantized_modules` (quantized.py:144-179) manually walks the model tree with string
splitting and `getattr` chains. Use `nn.state.get_state_dict(model)` which returns a flat
dict, or use a 3-line `_resolve_path(model, dotted_name)` helper.

**Saves ~8 lines** and is more robust to model structure changes.

### 8. Inline or Remove Static Method Wrappers
*Sources: opus, gpt5mini*

`_prepare_state_dict`, `_permute_llama_qk_weights`, `_split_mla_kv_weights` (llm.py:283-328,
46 lines) are only called from `from_gguf`. They could be inlined, or the Q/K permutation
could happen in the dequant pipeline, and kv_b splitting could happen during module init.
The `exp_probs_b` rename hack (llm.py:352-355) could be a proper keymap entry.

**Saves ~15 lines** by removing wrapper indirection.

### 9. Model Dict → Metadata Table
*Sources: gpt51-mini, gpt5mini*

Turn the `models` dict (llm.py:417-435) into a metadata table carrying URL, architecture,
default quantization flag, and max_context hint. This lets the CLI, smoke test, and any
future tooling share one entry. The `--quantized` default can read from metadata instead of
checking string prefixes.

**Saves ~5 lines** and prevents prefix-matching bugs.

### 10. Centralize Download/Cache Logic
*Sources: gpt51-mini, gpt5mini*

The `pathlib` logic in llm.py:532-537 (local model detection, URL resolution) is duplicated
by smoke_test.py. Factor into a `resolve_model(name_or_path)` helper in a shared module.

**Saves ~10 lines** across files.

---

## B. Increase Performance (tok/s)

### 11. Teach MV Heuristic to See Through Dequant Chains ★★★
*Sources: opus, gemini (implied by heuristic discussion)*

**The #1 performance opportunity.** The MV heuristic (heuristic.py:67-89) requires
`MUL(INDEX, INDEX)`. Fused dequant+matmul creates `MUL(dequant_chain(INDEX), INDEX)` which
fails pattern match. The current fix (line 76-78) skips the range check when the weight side
isn't INDEX, but it should walk backward through the dequant chain to find the underlying
INDEX and extract its ranges.

Currently fused Q4K/Q6K MoE kernels run at ~2 GB/s with serial reduction (no GROUP). With
MV matching, they'd get GROUP(16)+LOCAL(4)+UPCAST(4) and should hit 40-80 GB/s.

**Impact: 2-4x MoE layer throughput.** This alone could take GLM from ~21 to ~40+ tok/s.

### 12. Fuse Gate+Up Expert Matmul ★★
*Source: opus*

GLM does two separate expert matmuls reading the same input:
```python
gate = self.ffn_gate_exps(sel, h_norm).silu().contiguous()
up = self.ffn_up_exps(sel, h_norm).contiguous()
```

Concatenate gate and up weights into `(num_experts, 2*moe_hidden_dim, dim)`, do one matmul,
split. This halves expert weight gathers and kernel dispatches. llama.cpp does this.

**Impact: ~15-25% MoE speedup.**

### 13. Cache the Absorbed K Projection ★
*Source: opus*

`q_nope = q_nope @ self.attn_k_b.weight.T` (mla.py:105) runs every token. Pre-multiply
`attn_q_b_nope = attn_q_b[:, :qk_nope] @ k_b^T` once at load time. Eliminates one per-head
matmul per layer per token.

**Impact: ~5-10% on attention layers.**

### 14. Audit .contiguous() Calls ★
*Sources: opus, gemini*

Several `.contiguous()` exist "to break fusion" but some may no longer be needed:
- `TransformerBlock.__call__` (llm.py:230) and `MLATransformerBlock.__call__` (mla.py:153)
- `mla.py:139-140` gate/up contiguous — needed to break the dequant+matmul+weighted_sum
  mega-kernel, but if MV heuristic is fixed (idea 11), fused kernels might be fine

Each unnecessary `.contiguous()` adds a kernel dispatch. Profile with and without.

**Impact: ~5% fewer kernels per layer.**

### 15. Pre-allocate KV Caches in __init__
*Sources: opus, gemini, gpt51-mini*

Both attention implementations check `if not hasattr(self, "cache_kv")` inside forward
(llm.py:205, mla.py:111). Pre-allocate in `__init__` to make the JIT graph fully static
from the first token.

**Impact: Faster first-token latency, cleaner JIT graphs.**

### 16. Remove KV Cache .realize() Calls
*Sources: gemini, gpt5mini*

`_attention` calls `.realize()` on the KV cache assign (llm.py:207, mla.py:113). This
forces synchronization. Check if `assign` works lazily without it — the JIT should handle
the dependency.

**Impact: Reduced sync overhead, potential for better pipelining.**

### 17. Generalize custom_q4_0_linear to All Quant Types
*Sources: gpt51-mini, gpt5mini*

`custom_q4_0_linear` (quantized.py:7-30) only handles Q4_0. Extend to Q4_K/Q6_K so all
attention and FFN matmuls can skip the fp16 dequant cache (3.56x data amplification).
This needs the UOp DSL to express Q4K/Q6K block structure.

**Impact: 3.56x less memory traffic for quantized matmuls.** But note: MEMORY.md warns
that UOp DSL custom kernels were 6x SLOWER than tinygrad-fused path for MoE, so this needs
careful benchmarking.

### 18. Fuse FFN gate→silu→up→down
*Sources: gpt5mini*

For the dense FFN path, use the scheduler/codegen to fuse `ffn_gate(x).silu() * ffn_up(x)`
into fewer kernels. The `contiguous()` between gate and down (llm.py:226) may prevent this.

**Impact: Fewer kernel dispatches on dense blocks.**

### 19. Static Shapes in JIT
*Source: gemini*

Ensure `start_pos` is strictly symbolic and `T=1` is constant during generation. Any shape
variation disables JIT. The current code does this (llm.py:280) but verify edge cases —
the `start_pos != 0 and t.shape[-1] == 1` guard means the first token bypasses JIT.

**Impact: Ensures JIT is always active during generation.**

### 20. Batch-Oriented Generation
*Sources: gpt5mini*

Replace the per-token generator loop with a batched step API that processes multiple
positions, reducing Python-level overhead and enabling better GPU utilization.

**Impact: Moderate — Python overhead is small vs kernel time, but adds up at high tok/s.**

---

## C. Reuse Existing tinygrad Concepts

### 21. Use extra/models/llama.py Attention as Base
*Source: opus*

`extra/models/llama.py:36-109` has a battle-tested `Attention` class with KV cache, RoPE,
GQA, and `scaled_dot_product_attention`. The standard `TransformerBlock._attention`
(llm.py:189-216) reimplements this. The llama.py Attention has a `linear` parameter in its
constructor — pass `QuantizedLinear` for the quantized path.

**Saves ~25 lines** and reuses proven code.

### 22. Use Tensor.scaled_dot_product_attention More
*Sources: opus, gemini*

MLA attention (mla.py:116-124) manually computes `qk = q @ k.T * scale` then softmax then
`attn @ v`. For the QK+softmax part, `scaled_dot_product_attention` could work with the
kv_normed_cache as the "value" matrix. More importantly, this enables future flash attention
backends — manual QK+softmax+AV prevents this.

### 23. Reuse extra/models/llama.py FeedForward
*Source: opus*

`extra/models/llama.py:111-120` defines `FeedForward` (gate+up+down+silu). Both
TransformerBlock and MLATransformerBlock (dense path) do the same thing. Import and reuse.

**Saves ~5 lines** per block type.

### 24. Benchmark Tensor.topk vs _topk_pairwise
*Sources: opus, gemini*

`_topk_pairwise` (mla.py:6-22, 16 lines) exists because `Tensor.topk` generated many
kernels. But `Tensor.topk` (tensor.py:2774) uses `sort()` + shrink now. Benchmark both for
n=8, k=2 (GLM config). If `Tensor.topk` is competitive in kernel count, delete the custom
implementation.

### 25. Use PatternMatcher for Dequant Detection in Heuristic
*Source: opus*

The Q4_0 GROUP heuristic (heuristic.py:91-110) checks
`any(u.op in {Ops.AND, Ops.SHR} for u in toposort())`. This is a manual graph traversal.
tinygrad's `PatternMatcher` + `UPat` is designed for exactly this — more idiomatic and
extensible (e.g., detect Q4K vs Q6K patterns separately).

### 26. Consolidate Quantization Into nn.quant Module
*Sources: gpt5mini, gemini*

Move `QuantizedLinear`, `QuantizedExpertWeights`, dequant functions, and
`replace_quantized_modules` into `tinygrad/nn/quant.py`. Keep `apps/quantized.py` as a thin
wrapper or delete it. The quantization logic belongs in `nn/` next to `state.py` where the
GGML info already lives.

### 27. Use nn.Linear Interface for QuantizedLinear
*Source: gemini*

Make `QuantizedLinear` and `ExpertWeights` mimic `nn.Linear`'s interface (including `weight`
attribute) so `nn.state.get_parameters`, `nn.state.get_state_dict`, and other standard
introspection tools work seamlessly.

### 28. Memory-Mapped Disk Tensors for Faster Load
*Source: gpt5mini*

Use `Tensor(device="disk:...")` for large GGUF files in `from_gguf` to avoid full RAM copies.
Allows partial dequantization and faster startup, especially for 7B+ models.

### 29. Leverage GlobalCounters in Server for Metrics
*Source: gpt51-mini*

Reuse the `Timing`/`GlobalCounters` instrumentation from the benchmark loop
(llm.py:541-548) inside the HTTP handler so each request emits tok/s metrics.

### 30. TransformerConfig Dataclass
*Source: gpt51-mini*

Introduce a `TransformerConfig` dataclass mirroring the metadata parsed in `from_gguf`.
Pass it into the constructor instead of 20+ keyword arguments. Reduces parameter extraction
code and makes model configs serializable/comparable.

---

## Priority Matrix

| Tier | Ideas | Category | Expected Impact |
|------|-------|----------|-----------------|
| **S** | 11 (MV through dequant) | Perf | 2-4x MoE throughput |
| **A** | 12 (gate+up fusion), 1 (merge blocks), 5 (server split) | Perf/Lines | 15-25% MoE + -130 lines |
| **B** | 2 (preset dict), 4 (factor Q4_0), 13 (cache K absorb), 15 (pre-alloc KV) | Mixed | -40 lines + 5-10% perf |
| **C** | 3 (rope dedup), 6 (QL_CACHE), 7 (tree walk), 8 (static methods), 14 (contiguous audit) | Lines/Perf | -40 lines + ~5% |
| **D** | 21-30 (reuse patterns) | Reuse | Code quality, future-proofing |

**The single most impactful change is idea 11**: teaching the MV heuristic to walk through
dequant chains. This is a ~20-line change in heuristic.py that could double GLM throughput
by enabling GROUP+LOCAL+UPCAST on the MoE expert kernels that currently run with serial
reduction at 2 GB/s.
