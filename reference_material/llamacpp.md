# llama.cpp Implementation Reference

Summary of key patterns from llama.cpp that differ from tinygrad's approach.

## MoE (Mixture of Experts)

### Key File: `src/llama-graph.cpp` - `build_moe_ffn()`

### Topk Routing
llama.cpp has a fused CUDA kernel (`ggml/src/ggml-cuda/topk-moe.cu`) that does:
1. Optional softmax/sigmoid over logits
2. Iterative argmax to find top-k experts (not full sort)
3. Weight normalization
4. All in one kernel launch

```cpp
// topk-moe.cu - iterative argmax, k iterations
for (int k = 0; k < n_expert_used; k++) {
    // Find max across experts (warp reduction)
    // Mark selected expert as -inf for next iteration
    // Store weight and index
}
```

### Expert Computation - `ggml_mul_mat_id`
The key difference: llama.cpp uses `build_lora_mm_id()` which wraps `ggml_mul_mat_id`.
This does matmul with built-in expert selection - no intermediate gather tensor.

```cpp
// Line 1251: Direct matmul with expert selection
ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts);
// Result: [n_ff, n_expert_used, n_tokens]
```

### Weighted Sum - Explicit Adds
After computing expert outputs, llama.cpp does:
1. Multiply by weights: `experts = ggml_mul(experts, weights)`
2. Sum via k-1 explicit adds (NOT a reduction over axis):

```cpp
// Lines 1326-1350
experts = ggml_mul(ctx0, experts, weights);  // [n_embd, n_expert_used, n_tokens]

// Create views for each expert
for (int i = 0; i < n_expert_used; ++i) {
    cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, ...);
}

// Sum via explicit adds (k-1 adds for k experts)
moe_out = cur_experts[0];
for (int i = 1; i < n_expert_used; ++i) {
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
}
```

### tinygrad Comparison

**tinygrad (current):**
```python
# Creates [B, T, k, hidden] intermediate, then reduces
out = (self.ffn_down_exps(sel, gated).float() * probs.unsqueeze(-1)).sum(axis=2)
```

**llama.cpp approach (what we should do):**
```python
# No intermediate tensor, k-1 explicit adds
outs = [self.ffn_down_exps.forward_single(sel[..., i], gated[..., i, :]) * probs[..., i:i+1]
        for i in range(k)]
out = outs[0] + outs[1] + outs[2] + outs[3]  # for k=4
```

Or better: fuse weighted sum into the matmul kernel itself.

---

## MLA (Multi-head Latent Attention)

### Key File: `src/models/deepseek2.cpp`

### Absorption Optimization
llama.cpp implements MLA with "absorption" - the K projection weights are absorbed into the query:

```cpp
// Line 119: Absorb k_b into q_nope
// q_nope: [n_embd_head_qk_nope, n_tokens, n_head]
// wk_b: [n_embd_head_qk_nope, kv_lora_rank, n_head]
ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
// Result: [kv_lora_rank, n_tokens, n_head]
```

This converts MLA into MQA (single KV head), reducing the KV cache to just `kv_lora_rank` dimensions.

### KV Cache Structure
- Without absorption: Full MHA, KV cache is `[n_embd_head_k, n_head, n_tokens]`
- With absorption: MQA-like, KV cache is `[kv_lora_rank + n_rope, 1, n_tokens]`

### Query Construction
```cpp
// Concat absorbed q_nope with rope'd q_pe
// q_nope_absorbed: [kv_lora_rank, n_head, n_tokens]
// q_pe: [n_rope, n_head, n_tokens]
ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
// Result: [kv_lora_rank + n_rope, n_head, n_tokens]
```

### Key Insight
The attention is computed in the compressed space (`kv_lora_rank`), then the V projection (`wv_b`) is applied after attention to decompress back to `n_embd`.

---

## Q4K Dequantization

### Block Structure (144 bytes per 256 values)
```
Offset  Size  Content
0-1     2     d (scale) as float16
2-3     2     dmin (min scale) as float16
4-7     4     scales low 6 bits (4 groups)
8-11    4     mins low 6 bits (4 groups)
12-15   4     scales/mins high bits packed
16-143  128   quantized values (4 bits each, 256 values)
```

### Dequantization Formula
```cpp
// For each of 8 groups of 32 values:
scale = d * sc[group]
min = dmin * mn[group]
// For each value in group:
weight = scale * q - min
```

Where `q` is the 4-bit quantized value (0-15).

### llama.cpp Metal Kernel Pattern
They use SIMD groups with 32 threads processing 8 values each:
- Thread reads 1 byte (2 nibbles)
- Extracts high/low nibbles
- Broadcasts scale/min via SIMD shuffle
- Accumulates dot product in registers

---

## Key Takeaways for Optimization

1. **Avoid gather tensors** - Use matmul-with-selection ops instead of gather→matmul
2. **Explicit adds vs reduction** - For small k (4-8), k-1 adds beats reduction kernel
3. **Fuse routing** - Topk + softmax + normalization should be one kernel
4. **MLA absorption** - Compute in compressed space, decompress after attention
5. **Q4K locality** - Process blocks in SIMD-width chunks, keep scales in registers
