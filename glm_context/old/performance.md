# Performance Analysis: tinygrad LLM vs llama.cpp

This document compares tinygrad's LLM implementation with llama.cpp, focusing on performance optimizations and how tinygrad's architectural strengths can close the gap.

## Summary

| Aspect | llama.cpp | tinygrad | tinygrad Opportunity |
|--------|-----------|----------|---------------------|
| Quantized inference | Handcrafted SIMD kernels | Generic dequantize + matmul | Lazy graph fusion + BEAM |
| Flash attention | Custom fused kernels (FA-2) | Available in fa.py | Integrate into llm.py |
| KV cache | Paged, ring buffer, speculative | Simple contiguous buffer | Graph-level cache ops |
| MLA absorption | Full weight absorption | Implemented, matches llama.cpp | ✅ Done |
| MoE routing | Fused expert selection | Separate selection + gather | PatternMatcher fusion |
| Memory layout | Column-major (optimal for loads) | Row-major | Automatic layout optimization |

## tinygrad's Core Strengths

### 1. Laziness → Automatic Fusion

tinygrad builds computation graphs lazily. The scheduler sees the **entire graph** before execution:

```python
# These are lazy - no computation yet
w = dequantize_q4k(blocks)
y = x @ w.T
# NOW the scheduler sees both ops and can fuse them
y.realize()
```

**Why this matters**: llama.cpp needs hand-written fused kernels for each `(quant_type, hardware)` combination. tinygrad should discover fusion automatically.

**Current status**: Dequant and matmul are fused in `QuantizedLinear` via block-wise iteration, but not as a single kernel. Investigation needed on why full fusion doesn't happen.

### 2. BEAM Search → Hardware Adaptation

llama.cpp maintains **separate codepaths** for AVX2, AVX-512, NEON, etc. tinygrad's BEAM search finds optimal kernels automatically:

```bash
BEAM=4 python tinygrad/apps/llm.py --model deepseek-v2-lite --count 10
```

BEAM searches over:
- Tile sizes and shapes
- Vectorization strategies
- Memory access patterns
- Local/shared memory usage

**Opportunity**: Extend BEAM's search space to include dequantization patterns for quantized matmul.

### 3. PatternMatcher → Graph Rewriting

tinygrad's `graph_rewrite` with `PatternMatcher` can transform computation graphs:

```python
# Example: fuse MoE routing pattern
pm = PatternMatcher([
  (UPat(Ops.GATHER, src=(expert_weights, topk_indices)), fuse_moe_gather),
])
```

**Opportunity**: Add patterns for:
- MoE expert batching (router → gather → matmul → scatter)
- Flash attention pattern detection
- Quantized matmul fusion

## llama.cpp Techniques (Reference)

### 1. Quantized Matmul Kernels

llama.cpp's core advantage: **fused quantized matmul**:

```cpp
// Single kernel: dequant interleaved with accumulation
ggml_vec_dot_q4_K_q8_K(n, &dst, src0_quantized, src1_quantized);
```

- Dequantization happens in SIMD registers
- Zero intermediate memory allocation
- Per-quant-type specialization

**tinygrad path forward**:
1. Short-term: Improve `q4k_linear_fused` to match llama.cpp
2. Long-term: Let BEAM discover this pattern automatically

### 2. Activation Quantization

llama.cpp quantizes activations on-the-fly:
```cpp
quantize_row_q8_K(activations, quantized_activations, n);
// Then: q4_K weights × q8_K activations
```

Reduces memory bandwidth 4x for matmul.

**tinygrad path forward**: Add `Tensor.quantize_q8()` method, let scheduler fuse with matmul.

### 3. Flash Attention

llama.cpp uses FA-2 style kernels:
- Tiled computation fitting in L2/SRAM
- Online softmax (no full attention matrix)
- Causal mask fusion

**tinygrad status**: `extra/thunder/tiny/fa.py` exists but isn't integrated into `llm.py`. Needs:
- fp16 support (currently bfloat16 only)
- GQA/MQA handling
- KV cache integration

### 4. MoE Expert Fusion

llama.cpp's `build_moe_ffn()`:
1. Fused gating: router → topk → weights
2. Expert batching: group tokens by expert
3. Combined output without full materialization

**tinygrad path forward**: PatternMatcher rule to detect and fuse MoE pattern.

## Recommended Improvements

### High Impact (Leverage tinygrad Strengths)

| Improvement | Approach | Effort |
|-------------|----------|--------|
| Quant matmul fusion | Investigate why scheduler doesn't fuse dequant+matmul | Medium |
| BEAM for quant ops | Extend search space to include dequant patterns | Medium |
| Flash attention | Integrate fa.py, add fp16/GQA support | Low |
| MoE fusion pattern | Add PatternMatcher rule | Low |

### Medium Impact

| Improvement | Approach | Effort |
|-------------|----------|--------|
| Activation quantization | Add `quantize_q8` op | Medium |
| MLA flash attention | Custom kernel for absorbed MLA pattern | High |
| Paged KV cache | Non-contiguous cache blocks | Medium |

### Lower Priority

| Improvement | Approach | Effort |
|-------------|----------|--------|
| Quantized KV cache | Cache in Q8/Q4 | Low |
| Continuous batching | Multi-sequence support | High |
| Speculative decoding | Draft model integration | High |

## Investigation Questions

1. **Why doesn't dequant+matmul fuse?**
   - Is it the `.realize()` calls in block iteration?
   - UOp boundaries preventing fusion?
   - Memory layout incompatibility?

2. **What does BEAM find for Q4_K matmul?**
   ```bash
   BEAM=4 DEBUG=2 python -c "
   from tinygrad import Tensor
   # Profile Q4_K matmul kernel selection
   "
   ```

3. **Can PatternMatcher detect MoE patterns?**
   - Router logits → topk → gather → matmul → scatter
   - This is exactly what graph_rewrite is designed for

## Benchmarking

```bash
# llama.cpp baseline
./llama-cli -m model.gguf -p "Hello" -n 100 --no-mmap

# tinygrad with BEAM optimization
BEAM=4 python tinygrad/apps/llm.py --model model --prompt "Hello" --count 100

# Profile kernel breakdown
DEBUG=2 python tinygrad/apps/llm.py --model model --prompt "Hello" --count 10 2>&1 | grep -E "kernel|time"
```

Key metrics:
- **Tokens/second**: Primary throughput
- **Time to first token**: Interactive latency
- **Memory usage**: Especially for long contexts
- **Kernel fusion rate**: Via `DEBUG=2` output

## Investigation: Youtu-LLM 2B on M3 (Feb 2026)

### Baseline Performance

| Mode | Kernels | Batches | tok/s | Notes |
|------|---------|---------|-------|-------|
| Quantized (Q4K) | 1165 | 6 | ~3 | Heavy dequant overhead |
| Dequantized (fp16) | 784 | 5 | ~4 | Better but still slow |
| Expected (M3 32GB) | ~100-200 | 1-2 | 30-50 | Memory bandwidth limited |

**Key finding**: 4 tok/s on M3 is 10x slower than expected. The bottleneck is **too many kernels** (784) preventing efficient execution.

### Identified Fusion Blockers

1. **`.realize()` on KV cache assigns** - Forces sync after every layer
   ```python
   # OLD: Breaks fusion between layers
   self.cache_k[:, :, start_pos:start_pos+T, :].assign(k).realize()

   # NEW: Let scheduler handle write-before-read dependency
   self.cache_k[:, :, start_pos:start_pos+T, :].assign(k)
   ```

2. **`.contiguous()` at block boundaries** - Prevents cross-block fusion
   ```python
   # OLD: Forces memory layout, breaks fusion
   def __call__(self, x, start_pos):
     return self._feed_forward(self._attention(x, start_pos), start_pos).contiguous()

   # NEW: Allow lazy evaluation across blocks
   def __call__(self, x, start_pos):
     return self._feed_forward(self._attention(x, start_pos), start_pos)
   ```

3. **`.contiguous()` in FFN** - Breaks fusion between silu and multiply
   ```python
   # OLD
   gated = self.ffn_gate(h_norm).float().silu().contiguous() * self.ffn_up(h_norm).float()

   # NEW
   gated = self.ffn_gate(h_norm).float().silu() * self.ffn_up(h_norm).float()
   ```

### Kernel Analysis

**Per-layer breakdown** (~24 kernels per layer for 32 layers = 768 kernels):
- Q projection (LoRA path): 3-4 kernels (down, norm, up, reshape)
- KV projection: 2-3 kernels (linear, split, norm)
- RoPE: 1-2 kernels
- KV cache write: 1 kernel (ASSIGN)
- Q absorption matmul: 1 kernel
- QK matmul + softmax + V matmul: 3-4 kernels
- V expansion matmul: 1 kernel
- Output projection: 1 kernel
- RMSNorm (FFN): 1 kernel
- FFN gate/up/down: 3-4 kernels
- Residual adds: 1-2 kernels

**Slow kernels identified**:
| Kernel | Time | Notes |
|--------|------|-------|
| Output projection (2048→128256) | 6-16ms | Dominates total time |
| FFN down (5632→2048) | 0.3ms | Q4K dequant overhead |
| q4k_linear_fused | 1-4ms | Hand-tuned, still slow |

### Quantized vs Dequantized

**Quantized (Q4K) slower because**:
1. `q4k_linear_fused` kernel doesn't utilize full memory bandwidth
2. Extra dequant kernels add latency
3. Separate dequant+matmul doesn't fuse to single kernel

**Dequantized (fp16) faster because**:
1. Contiguous fp16 weights = better memory access
2. Metal's matmul kernels highly optimized for fp16
3. No dequant overhead per kernel

**Recommendation**: For small MLA models like Youtu (2B), **dequantized is faster** until quantized kernels are properly fused.

### QUANT_LAZY Optimization

Added lazy dequantization path that dequantizes full weight at once (vs block-by-block):

```python
# QUANT_LAZY=1 (default): Full dequant, single matmul
if getenv("QUANT_LAZY", 1):
  w = self._dequant_fn(self.blocks).reshape(self.out_features, self.in_features)
  return x.linear(w.T, None)

# QUANT_LAZY=0: Block-by-block with realizes (slower)
for bi in range(blocks_per_row):
  w = dequantize(block[bi]).realize()
  out += x_slice @ w.T
  out = out.realize()  # Sync every block!
```

### BEAM_SKIP_MS Optimization

Added early exit from BEAM search when kernel is fast enough:

```python
if skip_threshold > 0 and min(tms) < skip_threshold:
  beam = [(candidates[i], min(tms))]
  exiting = True
  break
```

Usage: `BEAM_SKIP_MS=1` exits BEAM as soon as any kernel runs < 1ms.

### Remaining Work

1. **Test fusion changes** - Verify kernel count reduced after removing realizes/contiguous
2. **Profile with VIZ=1** - Visualize which ops fuse and which don't
3. **BEAM on slow kernels** - Target the 16ms output projection
4. **Flash attention for MLA** - Custom kernel for absorbed MLA pattern
5. **Fused Q4K matmul** - Single kernel: dequant in registers → accumulate

### Commands for Testing

```bash
# Baseline (before changes)
DEBUG=2 python tinygrad/apps/llm.py --model youtu-llm:2b-Q4 --no-quantized --benchmark 3

# After removing fusion blockers
DEBUG=2 python tinygrad/apps/llm.py --model youtu-llm:2b-Q4 --no-quantized --benchmark 3

# With BEAM optimization
BEAM=4 BEAM_SKIP_MS=1 DEBUG=2 python tinygrad/apps/llm.py --model youtu-llm:2b-Q4 --no-quantized --benchmark 3

# Visualize kernel fusion
VIZ=1 python tinygrad/apps/llm.py --model youtu-llm:2b-Q4 --no-quantized --prompt "Hi" --count 2
```

## MLA Efficiency Improvements (llm_mla.py)

Comparing `MLATransformerBlock` with standard `TransformerBlock` reveals several efficiency opportunities:

### 1. Use `scaled_dot_product_attention` (biggest impact)

**Current** (manual matmul):
```python
qk = q.matmul(k.transpose(-2, -1)) * scale
attn_weights = qk.softmax(-1).cast(qk.dtype)
attn = attn_weights.matmul(v)
```

**Standard block** (fused attention):
```python
attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
```

The fused attention is significantly faster, especially for large contexts. **Solution**: Pre-scale Q to compensate for MLA's custom attention scale (`mscale²/sqrt(q_head_dim)`). For the absorbed path, also compensate for different Q dimension (`kv_lora+rope` vs `q_head_dim`).

### 2. Combine KV cache into single tensor

**MLA** (two tensors):
```python
self.cache_k = Tensor.zeros(B, 1, self.max_context, cache_dim, ...)
self.cache_v = Tensor.zeros(B, 1, self.max_context, self.kv_lora_rank, ...)
```

**Standard** (single tensor):
```python
self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim, ...)
```

Single cache tensor allows single assign/realize instead of two.

### 3. Move imports to module level

Imports inside `_attention` add overhead on every call:
```python
def _attention(self, x, start_pos):
    from tinygrad.apps.llm import precompute_freqs_cis  # Move to top of file
    from tinygrad.apps.yarn import precompute_freqs_cis_yarn  # Move to top
```

### 4. Remove unnecessary type conversions

MLA had many `.float()` casts that were removed:
```python
# OLD (removed)
router_logits = h_norm.float() @ self.ffn_gate_inp.weight.float().T
gated = gate_out.float().silu() * up_out.float()

# NEW
router_logits = h_norm @ self.ffn_gate_inp.weight.T
gated = gate_out.silu() * up_out
```

### 5. Simplify freqs_cis caching

The fallback logic is complex but rarely triggers since `freqs_cis_cache` is set in Transformer constructor. Consider removing or making it an assertion.

### 6. Consider removing `.contiguous()` calls

- Line 20 (`precompute_yarn_freqs_cis`): May not be needed
- Line 307 (dense FFN): Prevents fusion between gate/up

### Summary of Changes Made

| Change | Status | Impact |
|--------|--------|--------|
| Remove `.float()` in apply_rope_interleaved | ✅ Done | Minor |
| Remove `.float()` in MoE routing | ✅ Done | Medium |
| Remove `.float()` in expert outputs | ✅ Done | Medium |
| Remove `.float()` in shared experts | ✅ Done | Minor |
| Remove `.float()` in dense FFN | ✅ Done | Minor |
| Use `scaled_dot_product_attention` | ✅ Done | High |
| Combine KV cache | ❌ Needs investigation | Medium |
| Move imports to module level | ❌ TODO | Low |

## Conclusion

llama.cpp's advantage is **hand-optimized kernels**. tinygrad's advantage is **automatic optimization**.

The path forward isn't to replicate llama.cpp's hand-written kernels, but to:
1. Fix barriers preventing automatic fusion (dequant+matmul)
2. Extend BEAM's search space for quantized ops
3. Add PatternMatcher rules for common patterns (MoE, flash attention)

tinygrad's architecture should make these optimizations **discoverable** rather than requiring manual implementation for each hardware target.

**Key insight from Youtu investigation**: Most performance loss comes from **sync points** (`.realize()`, `.contiguous()`) that break fusion. Removing these allows the scheduler to see larger graphs and fuse more aggressively.
