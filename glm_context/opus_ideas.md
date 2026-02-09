# Opus Ideas for GLM/LLM Code Simplification & Performance

## Scope of the Diff

master `llm.py` is 368 lines. The branch adds 4 new files and modifies 3, totaling **+1,014 lines**:

| File | Lines | Purpose |
|------|-------|---------|
| `apps/llm.py` | 592 (+224) | Tokenizer, TransformerBlock, MLA routing, model dict, server, CLI |
| `apps/mla.py` | 178 (new) | MLATransformerBlock, _topk_pairwise, load_mla_params_from_gguf |
| `apps/quantized.py` | 179 (new) | QuantizedLinear, QuantizedExpertWeights, replace_quantized_modules |
| `apps/rope.py` | 99 (new) | RoPE + YaRN precomputation, apply_rope, apply_rope_interleaved |
| `apps/smoke_test.py` | 113 (new) | Multi-model smoke test |
| `codegen/opt/heuristic.py` | 220 (+63) | MV dequant-chain fix, Q4_0 GROUP heuristic |
| `nn/state.py` | 467 (+123) | dequantize_q4k/q5k/q6k/q4_0, GGML_QUANT_INFO, quantized gguf_load |

## The 20 Ideas

---

### A. Cut Lines in Half (Ideas 1-8)

#### 1. Merge TransformerBlock and MLATransformerBlock

`TransformerBlock` (llm.py:154-230, 77 lines) and `MLATransformerBlock` (mla.py:28-153, 126 lines)
duplicate `_feed_forward`, `__call__`, norm layers, and MoE routing. The only real difference is
`_attention`. Merge into one class with the attention method as a strategy:

```python
class TransformerBlock:
  def __init__(self, ..., attention_type="standard"):
    # shared: norms, FFN (dense or MoE), __call__
    # differs: attention projections only
```

The MoE feed-forward code in `MLATransformerBlock._feed_forward` (mla.py:127-150) is almost
identical to `TransformerBlock._feed_forward` (llm.py:218-227) except for the gating function
and shared experts. A single `_feed_forward` with the superset of features would work. The
`ExpertWeights` class and MoE routing could be a `MoEFFN` helper used by both.

**Saves ~60 lines** (eliminate duplicated _feed_forward, __call__, norms).

#### 2. Delete _rope_interleaved from mla.py — Use rope.py

`mla.py:78-84` defines `_rope_interleaved` as a staticmethod that's nearly identical to
`rope.py:55-63` `apply_rope_interleaved`. The only difference: mla.py skips the float32
round-trip ("safe for fp16"). But `apply_rope_interleaved` already casts back to `input_dtype`,
so it handles fp16 fine — the float() intermediate is for numerical safety with large values.

Either use `apply_rope_interleaved` directly, or add a `native_dtype=True` flag. The mla.py
version is 6 lines; removing it and the import cleans up the code.

**Saves ~8 lines** and eliminates a subtle divergence risk.

#### 3. Inline the QL_CACHE_* Debug Knobs or Remove Them

`quantized.py:56-65` has 10 lines of `getenv("QL_CACHE_ATTN_Q_A", 0)` etc. matching specific
`(out_features, in_features)` tuples. These are debugging aids for one model (youtu-llm:2b)
that shouldn't be in production code. Replace with a single `QL_CACHE_ALL` env var or remove
entirely — the cache path is already the fallback at line 88.

**Saves ~10 lines** of per-layer-size debug switches.

#### 4. Collapse QuantizedLinear and QuantizedExpertWeights

`QuantizedExpertWeights` (quantized.py:91-142, 52 lines) duplicates most of `QuantizedLinear`'s
Q4_0 inline-dequant logic (lines 126-137 mirror lines 75-87). Factor the shared dequant-matmul
into a function:

```python
def _q4_0_matmul(x_flat, blocks, out_features, in_features):
  # shared Q4_0 packed-dot logic (~12 lines)
  ...

class QuantizedLinear:
  def __call__(self, x): return _q4_0_matmul(x, self._q4_0_blocks, ...)

class QuantizedExpertWeights:
  def __call__(self, sel, x): return _q4_0_matmul(x_flat, sel_blocks, ...)
```

**Saves ~20 lines** of duplicated Q4_0 inline dequant.

#### 5. Move Tokenizer Preset Logic to a Data Dict

`SimpleTokenizer.role()` (llm.py:114-120), `end_turn()` (121-127), and `build_chat_ids()`
(128-144) have cascading if/elif chains for each preset. Define a `PRESET_CONFIG` dict:

```python
PRESET_CONFIG = {
  "glm4": {"role_fmt": "<|{role}|>\n", "end_turn": [], "prefix": "<sop>", "think": True},
  "qwen2": {"role_fmt": "<|im_start|>{role}\n", "end_turn_has_eos": True, ...},
  ...
}
```

Then `role()`, `end_turn()`, and `build_chat_ids()` become 3 lines each looking up the config.

**Saves ~20 lines** and makes adding new presets trivial.

#### 6. Simplify replace_quantized_modules Tree Walk

`replace_quantized_modules` (quantized.py:144-179) manually walks the model tree with string
splitting and `getattr` chains (12 lines of tree traversal). Use `nn.state.get_state_dict(model)`
which already returns a flat `{name: param}` dict, then use the same path to `setattr`:

```python
def replace_quantized_modules(model, quantized_tensors, state_dict):
  sd = nn.state.get_state_dict(model)
  for name, (blocks, shape, ggml_type, *rest) in list(quantized_tensors.items()):
    module_path = name[:-7]  # strip .weight
    parent, attr = _resolve_path(model, module_path)  # 3-line helper
    if isinstance(getattr(parent, attr), nn.Linear): ...
```

**Saves ~8 lines** and is more robust.

#### 7. Eliminate _prepare_state_dict, _permute_llama_qk_weights, _split_mla_kv_weights

These static methods on Transformer (llm.py:283-328, 46 lines) are only called from `from_gguf`.
They could be inlined into `from_gguf` or moved to standalone functions in a `gguf_helpers.py`.
But more importantly: the llama Q/K permutation (lines 297-311) dequantizes quantized Q/K
weights just to permute them, then loads them as fp16. This should happen in the dequant
pipeline, not as a special-case in the model loader.

For MLA kv_b splitting (lines 314-327): this could be a transform in `load_state_dict` or
handled by `split_kv_b` being called during module init.

**Saves ~15 lines** by removing wrapper methods.

#### 8. Move the Chat HTML / Server to a Separate File

The OpenAI-compatible server (llm.py:440-511, 72 lines) including `CHAT_HTML` blob and
`Handler` class is a distinct feature from the model itself. Move to `apps/serve.py` and
import. This keeps `llm.py` focused on model definition and generation.

**Saves ~72 lines** from llm.py (moves them, doesn't delete).

---

### B. Increase Performance (Ideas 9-14)

#### 9. Teach the MV Heuristic to See Through Dequant Chains

The current MV fix (heuristic.py:76-78) skips the range check when the weight side isn't an
INDEX (i.e., it's a dequant chain). But it only tries the first matching `act_i` then breaks.
The real fix: when the weight side is a dequant chain, walk backward through it to find the
underlying INDEX(es) and extract their ranges for the subset check. This would let MV apply
GROUP+LOCAL+UPCAST to fused dequant matmuls — the single biggest optimization lever.

**Current state**: Fused Q4K/Q6K MoE kernels run at ~2 GB/s because MV can't match them, so
they get no GROUP (serial reduction). With MV matching, they'd get GROUP(16)+LOCAL(4)+UPCAST(4)
and should hit 40-80 GB/s.

**Impact**: Could 2-4x MoE layer throughput. This is the #1 performance opportunity.

#### 10. Fuse Expert Gate+Up into a Single Matmul

Currently GLM does two separate expert matmuls per token:
```python
gate = self.ffn_gate_exps(sel, h_norm).silu().contiguous()
up = self.ffn_up_exps(sel, h_norm).contiguous()
```

Both read the same input `h_norm` and index the same experts `sel`. Concatenate gate and up
weights into one `(num_experts, 2*moe_hidden_dim, dim)` tensor, do one matmul, then split:
```python
gate_up = self.ffn_gate_up_exps(sel, h_norm)  # (B, T, K, 2*H)
gate, up = gate_up.chunk(2, dim=-1)
gated = gate.silu() * up
```

This halves the number of expert matmuls (and expert weight gathers). llama.cpp does this.

**Impact**: ~15-25% speedup on MoE layers (fewer kernel dispatches, better memory coalescing).

#### 11. Cache the Absorbed K Projection

In MLA attention (mla.py:105): `q_nope = q_nope @ self.attn_k_b.weight.transpose(-1, -2)`
This is a `(B, n_heads, T, qk_nope_head_dim) @ (n_heads, qk_nope_head_dim, kv_lora_rank)` matmul
that happens every token. For T=1 generation, the k_b weight doesn't change. The absorption
should be done once during model loading:

```python
# In __init__ or after weight loading:
# attn_q_b already projects to (n_heads, q_head_dim). Split into nope+pe parts.
# Pre-multiply: attn_q_b_nope = attn_q_b[:, :qk_nope] @ k_b^T
# Then at runtime: q_absorbed = attn_q_b_absorbed(q_lora) — one matmul instead of two
```

This eliminates one per-head matmul per layer per token.

**Impact**: ~5-10% on attention layers.

#### 12. Remove Unnecessary .contiguous() Calls

Several `.contiguous()` calls exist "to break fusion" but may no longer be necessary:

- `TransformerBlock.__call__` (llm.py:230): `.contiguous()` after feed-forward — needed for
  JIT boundary but check if removing it changes kernel count
- `MLATransformerBlock.__call__` (mla.py:153): same
- `mla.py:139-140`: gate and up contiguous — these were added to break the dequant+matmul+weighted_sum
  mega-kernel. But if MV heuristic is fixed (idea 9), the fused kernel might be fine

Each unnecessary `.contiguous()` adds a kernel dispatch. Profile with and without each one.

**Impact**: 1-3 fewer kernels per layer = ~5% at current kernel counts.

#### 13. Pre-allocate All KV Caches in __init__

Both attention implementations use `if not hasattr(self, "cache_kv")` / `if not hasattr(self, "cache_k")` checks inside the forward pass (llm.py:205, mla.py:111). This means the first
token allocates caches dynamically. Pre-allocate in `__init__`:

```python
def __init__(self, ...):
  self.cache_k = Tensor.empty((1, 1, max_context, cache_dim), dtype=dtypes.float16).contiguous().realize()
```

This makes the JIT graph fully static from the first token and avoids the hasattr branch.

**Impact**: Faster first-token latency, cleaner JIT graphs.

#### 14. Use float32 Softmax for Decode (Already Done in MLA, Not in Standard)

MLA attention (mla.py:121-122) does `e = qk.float().exp(); attn_weights = (e / e.sum(...)).half()`
for T=1 decode. Standard attention (llm.py:213) uses `scaled_dot_product_attention` which
may or may not do float32 internally. Ensure both paths use float32 for softmax to avoid
numerical issues with long contexts — and more importantly, ensure this compiles to a single
fused kernel, not separate float→exp→sum→div→half kernels.

**Impact**: Numerical correctness + potential kernel fusion improvement.

---

### C. Reuse Existing tinygrad Concepts (Ideas 15-20)

#### 15. Use extra/models/llama.py's Attention Class as Base

`extra/models/llama.py:36-109` has a battle-tested `Attention` class with KV cache, RoPE,
GQA support, and `scaled_dot_product_attention`. The standard `TransformerBlock._attention`
in `apps/llm.py:189-216` reimplements all of this. Options:

a) Import and subclass `extra.models.llama.Attention` for the standard (non-MLA) path
b) Factor out a shared `KVCache` class that both use

The llama.py Attention even has a `linear` parameter for the constructor, meaning you can
pass `QuantizedLinear` in place of `nn.Linear`. This is exactly what the quantized path needs.

**Saves ~25 lines** and reuses proven code.

#### 16. Use Tensor.scaled_dot_product_attention Everywhere

MLA attention (mla.py:116-124) manually computes `qk = q @ k.T * scale` then softmax then
`attn @ v`. For the non-absorbed path (standard attention in llm.py:213), it already uses
`scaled_dot_product_attention`. The MLA path can't use it directly because of the absorbed-V
trick (attn @ kv_normed_cache @ v_b^T). But the QK scoring + softmax part could still use it
with a custom value matrix = `kv_normed_cache[:, :, :, :kv_lora_rank]`.

More practically: `Tensor.scaled_dot_product_attention` may enable flash attention backends
in the future. Any manual QK+softmax+AV prevents this.

#### 17. Reuse extra/models/llama.py's FeedForward Class

`extra/models/llama.py:111-120` defines a `FeedForward` class (gate+up+down with silu). Both
`TransformerBlock._feed_forward` and `MLATransformerBlock._feed_forward` (dense path) do the
same thing. Import and reuse:

```python
from extra.models.llama import FeedForward
# Dense FFN: self.ffn = FeedForward(dim, hidden_dim)
# In _feed_forward: return h + self.ffn(self.ffn_norm(h))
```

The MoE path is different and needs custom code, but the dense fallback is identical.

**Saves ~5 lines** per block type.

#### 18. Use nn.state.gguf_load's Existing convert_from_gguf Pattern

`extra/models/llama.py:259-272` defines `convert_from_gguf()` with a keymap dict that renames
GGUF tensor names to model attribute names. Currently `apps/llm.py` relies on GGUF names matching
model attribute names exactly (which they do for the `blk.N.attn_q` convention). But the llama.py
approach is more explicit and handles edge cases like `output.weight` tying.

Instead of the current `_prepare_state_dict` + `exp_probs_b` rename hack (llm.py:352-355), use
a proper keymap that handles all renamings in one place.

#### 19. Leverage Tensor.topk for Expert Selection Where Possible

`_topk_pairwise` (mla.py:6-22) exists because `Tensor.topk` generates many kernels via bitonic
sort. But `Tensor.topk` (tensor.py:2774) uses `sort()` + shrink, which might now be better
optimized. Benchmark both for n=8, k=2 (GLM's config):

```python
# Current: 3 kernels, O(n^2) — 16 lines
values, indices = _topk_pairwise(scores, k)

# Alternative: may be fewer kernels now
values, indices = scores.topk(k, dim=-1)
```

If `Tensor.topk` is now competitive in kernel count, delete the 16-line custom implementation.

#### 20. Use PatternMatcher for the Q4_0 Heuristic Detection

The Q4_0 GROUP heuristic (heuristic.py:91-110) detects dequant kernels by checking
`any(u.op in {Ops.AND, Ops.SHR} for u in k.reduceop.src[0].toposort())`. This is a manual
graph traversal. tinygrad's `PatternMatcher` + `UPat` is designed exactly for this:

```python
_is_dequant = PatternMatcher([
  (UPat(Ops.AND), lambda: True),
  (UPat(Ops.SHR), lambda: True),
])
# Then: if graph_rewrite(k.reduceop.src[0], _is_dequant, ...) found matches...
```

This is more idiomatic tinygrad and would integrate better with future pattern-based
optimizations. It also makes it easy to extend the detection (e.g., detect Q4K vs Q6K
patterns separately and apply different opts).

---

## Priority Ranking

| Priority | Idea | Type | Expected Impact |
|----------|------|------|-----------------|
| 1 | 9. MV heuristic through dequant chains | Perf | 2-4x MoE throughput |
| 2 | 10. Fuse gate+up expert matmul | Perf | 15-25% MoE speedup |
| 3 | 1. Merge TransformerBlock + MLA | Lines | -60 lines |
| 4 | 4. Factor Q4_0 dequant-matmul | Lines | -20 lines |
| 5 | 5. Preset config dict | Lines | -20 lines |
| 6 | 8. Separate server file | Lines | -72 lines from llm.py |
| 7 | 11. Cache absorbed K projection | Perf | 5-10% attention |
| 8 | 15. Reuse llama.py Attention | Reuse | -25 lines + proven code |
| 9 | 12. Audit .contiguous() calls | Perf | ~5% fewer kernels |
| 10 | 3. Remove QL_CACHE debug knobs | Lines | -10 lines |
