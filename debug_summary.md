# DeepSeek-V2-Lite MLA Debug Summary

## Problem
Model produces wrong output - grammatically plausible but semantically incorrect.

### Test 1: "the capital of france is "
- **Expected output** (llama.cpp): "Paris."
- **Actual output** (tinygrad): "1", "10-10", garbage
- **Top predicted tokens**: [16, 18, 17, 790, 40] = ['1', '3', '2', 'âĢľ', 'I']

### Test 2: "what is 1+1" (log2.txt)
- **Expected output**: "2" or explanation
- **Actual output**: ". is the question.\n\n\n\n\n"
- **Top 5 tokens**: [13, 10, 16, 317, 11] = ['.', '+', '1', ' is', ':']

## What Works (Verified)

| Component | Test | Result |
|-----------|------|--------|
| Q4_K dequantization | numpy vs tinygrad | PASS (max_diff < 1e-3) |
| Q6_K dequantization | numpy vs tinygrad | PASS (max_diff < 1e-3) |
| Token embeddings | shape, values | PASS (mean=-0.0006, std=0.13) |
| Output layer weights | Q6_K load | PASS (mean=0.0003, std=0.15) |
| Tokenization | encode/decode | PASS (token 8913 = " Paris") |
| RoPE frequencies (YaRN) | numpy vs tinygrad | PASS (max_diff < 1e-6) |
| RoPE application | interleaved format | PASS (max_diff < 1e-6) |
| Attention scale formula | mscale²/sqrt(q_head_dim) | PASS (static check) |
| Absorbed vs unabsorbed QK | equivalence test | PASS |
| MoE routing | expert selection | PASS |
| All 26 parity tests | llama_cli_parity_checks.py | PASS |

## Key Parameters (from GGUF/llama.cpp)

```
arch = deepseek2
n_heads = 16
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
q_head_dim = 192 (128 + 64)

YaRN:
  scaling_factor = 40.0
  freq_scale = 0.025 (1/40)
  yarn_log_multiplier = 0.707
  attn_factor = 1.0
  mscale = 1.725899
  scale = 0.214971 (mscale²/sqrt(192))
```

## Fixes Already Applied

1. **mscale computation** - Using `rope_attn_factor` instead of `yarn_attn_factor`
2. **expert_weights_norm** - Default changed to `False` for DeepSeek2
3. **yarn_attn_factor cancellation** - Fixed for deepseek2 architecture

## What Hasn't Been Tested

1. **Full forward pass comparison** - numpy vs tinygrad through all 27 layers
2. **Layer-by-layer hidden state comparison** - comparing intermediate activations
3. **Fused Q4K kernel with large tensors** - test passes with small (2x256), model uses large (3072x2048)
4. **Position index handling during generation** - start_pos increments

## Hypotheses

1. **Fused kernel bug at scale** - Q4K_FUSED test uses 2x256, model uses much larger tensors
2. **Weight layout issue** - Something in how quantized blocks are reshaped for large matrices
3. **Accumulated numerical error** - Small errors compound over 27 layers
4. **Cache/position bug** - KV cache or position indexing issue during generation

## Files Created

- `debug_rope.py` - YaRN frequency comparison
- `debug_rope_apply.py` - RoPE application comparison
- `debug_output_layer.py` - Q6_K output layer verification
- `debug_single_layer.py` - Single layer forward pass
- `debug_mla_numpy_vs_tinygrad.py` - MLA attention comparison
- `test/llama_cli_parity_checks.py` - 26 parity tests (all pass)
- `test/test_gguf_weight_parity.py` - Weight loading verification

## Next Steps

1. Compare first layer output between tinygrad and a known-good reference
2. Add logging to compare hidden states after each layer
3. Test Q4K_FUSED with model-sized tensors (not small test tensors)
4. Binary search: find which layer's output diverges first
