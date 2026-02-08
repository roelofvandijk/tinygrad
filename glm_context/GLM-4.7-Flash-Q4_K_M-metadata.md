# GLM-4.7-Flash-Q4_K_M.gguf Metadata Analysis

## Key-Value Metadata

### deepseek2
| Key | Value |
|-----|-------|
| `deepseek2.attention.head_count` | 20 |
| `deepseek2.attention.head_count_kv` | 1 |
| `deepseek2.attention.key_length` | 576 |
| `deepseek2.attention.key_length_mla` | 256 |
| `deepseek2.attention.kv_lora_rank` | 512 |
| `deepseek2.attention.layer_norm_rms_epsilon` | 9.999999747378752e-06 |
| `deepseek2.attention.q_lora_rank` | 768 |
| `deepseek2.attention.value_length` | 512 |
| `deepseek2.attention.value_length_mla` | 256 |
| `deepseek2.block_count` | 47 |
| `deepseek2.context_length` | 202752 |
| `deepseek2.embedding_length` | 2048 |
| `deepseek2.expert_count` | 64 |
| `deepseek2.expert_feed_forward_length` | 1536 |
| `deepseek2.expert_gating_func` | 2 |
| `deepseek2.expert_group_count` | 1 |
| `deepseek2.expert_group_used_count` | 1 |
| `deepseek2.expert_shared_count` | 1 |
| `deepseek2.expert_used_count` | 4 |
| `deepseek2.expert_weights_norm` | True |
| `deepseek2.expert_weights_scale` | 1.7999999523162842 |
| `deepseek2.feed_forward_length` | 10240 |
| `deepseek2.leading_dense_block_count` | 1 |
| `deepseek2.rope.dimension_count` | 64 |
| `deepseek2.rope.freq_base` | 1000000.0 |
| `deepseek2.vocab_size` | 154880 |

### general.architecture
| Key | Value |
|-----|-------|
| `general.architecture` | 'deepseek2' |

### general.base_model
| Key | Value |
|-----|-------|
| `general.base_model.0.name` | 'GLM 4.7 Flash' |
| `general.base_model.0.organization` | 'Zai Org' |
| `general.base_model.0.repo_url` | 'https://huggingface.co/zai-org/GLM-4.7-Flash' |
| `general.base_model.count` | 1 |

### general.basename
| Key | Value |
|-----|-------|
| `general.basename` | 'Glm-4.7-Flash' |

### general.file_type
| Key | Value |
|-----|-------|
| `general.file_type` | 15 |

### general.languages
| Key | Value |
|-----|-------|
| `general.languages` | ['en', 'zh'] |

### general.license
| Key | Value |
|-----|-------|
| `general.license` | 'mit' |

### general.name
| Key | Value |
|-----|-------|
| `general.name` | 'Glm-4.7-Flash' |

### general.quantization_version
| Key | Value |
|-----|-------|
| `general.quantization_version` | 2 |

### general.quantized_by
| Key | Value |
|-----|-------|
| `general.quantized_by` | 'Unsloth' |

### general.repo_url
| Key | Value |
|-----|-------|
| `general.repo_url` | 'https://huggingface.co/unsloth' |

### general.sampling
| Key | Value |
|-----|-------|
| `general.sampling.temp` | 1.0 |

### general.size_label
| Key | Value |
|-----|-------|
| `general.size_label` | '64x2.6B' |

### general.tags
| Key | Value |
|-----|-------|
| `general.tags` | ['unsloth', 'text-generation'] |

### general.type
| Key | Value |
|-----|-------|
| `general.type` | 'model' |

### quantize
| Key | Value |
|-----|-------|
| `quantize.imatrix.chunks_count` | 85 |
| `quantize.imatrix.dataset` | 'unsloth_calibration_GLM-4.7-Flash.txt' |
| `quantize.imatrix.entries_count` | 607 |
| `quantize.imatrix.file` | 'GLM-4.7-Flash-GGUF/imatrix_unsloth.gguf' |

### tokenizer.chat_template
| Key | Value |
|-----|-------|
| `tokenizer.chat_template` | '[gMASK]<sop>\n{%- if tools -%}\n<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{% for tool in tools %}\n{{ tool | tojson(ensure_ascii=False) }}\n{% endfor %}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg...' |

### tokenizer.ggml
| Key | Value |
|-----|-------|
| `tokenizer.ggml.bos_token_id` | 154822 |
| `tokenizer.ggml.eom_token_id` | 154829 |
| `tokenizer.ggml.eos_token_id` | 154820 |
| `tokenizer.ggml.eot_token_id` | 154827 |
| `tokenizer.ggml.model` | 'gpt2' |
| `tokenizer.ggml.padding_token_id` | 154821 |
| `tokenizer.ggml.pre` | 'glm4' |
| `tokenizer.ggml.unknown_token_id` | 154820 |

### Large Arrays (skipped)
| Key | Length |
|-----|--------|
| `tokenizer.ggml.merges` | 321649 |
| `tokenizer.ggml.token_type` | 154880 |
| `tokenizer.ggml.tokens` | 154880 |

## Tensor Summary

### Non-Quantized Tensors
Total: 422

| Name | Shape | Dtype |
|------|-------|-------|
| `blk.0.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.0.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.0.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.0.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.0.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.0.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.0.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.1.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.1.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.1.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.1.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.1.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.1.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.1.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.1.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.1.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.10.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.10.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.10.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.10.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.10.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.10.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.10.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.10.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.10.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.11.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.11.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.11.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.11.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.11.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.11.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.11.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.11.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.11.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.12.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.12.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.12.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.12.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.12.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.12.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.12.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.12.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.12.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.13.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.13.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.13.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.13.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.13.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.13.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.13.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.13.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.13.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.14.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.14.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.14.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.14.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.14.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.14.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.14.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.14.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.14.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.15.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.15.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.15.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.15.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.15.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.15.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.15.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.15.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.15.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.16.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.16.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.16.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.16.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.16.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.16.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.16.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.16.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.16.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.17.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.17.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.17.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.17.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.17.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.17.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.17.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.17.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.17.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.18.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.18.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.18.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.18.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.18.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.18.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.18.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.18.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.18.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.19.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.19.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.19.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.19.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.19.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.19.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.19.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.19.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.19.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.2.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.2.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.2.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.2.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.2.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.2.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.2.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.2.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.2.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.20.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.20.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.20.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.20.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.20.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.20.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.20.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.20.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.20.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.21.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.21.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.21.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.21.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.21.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.21.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.21.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.21.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.21.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.22.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.22.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.22.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.22.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.22.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.22.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.22.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.22.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.22.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.23.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.23.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.23.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.23.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.23.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.23.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.23.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.23.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.23.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.24.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.24.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.24.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.24.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.24.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.24.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.24.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.24.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.24.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.25.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.25.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.25.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.25.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.25.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.25.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.25.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.25.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.25.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.26.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.26.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.26.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.26.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.26.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.26.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.26.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.26.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.26.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.27.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.27.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.27.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.27.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.27.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.27.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.27.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.27.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.27.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.28.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.28.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.28.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.28.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.28.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.28.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.28.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.28.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.28.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.29.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.29.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.29.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.29.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.29.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.29.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.29.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.29.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.29.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.3.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.3.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.3.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.3.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.3.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.3.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.3.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.3.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.3.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.30.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.30.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.30.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.30.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.30.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.30.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.30.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.30.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.30.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.31.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.31.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.31.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.31.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.31.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.31.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.31.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.31.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.31.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.32.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.32.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.32.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.32.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.32.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.32.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.32.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.32.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.32.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.33.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.33.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.33.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.33.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.33.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.33.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.33.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.33.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.33.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.34.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.34.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.34.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.34.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.34.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.34.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.34.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.34.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.34.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.35.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.35.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.35.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.35.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.35.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.35.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.35.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.35.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.35.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.36.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.36.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.36.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.36.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.36.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.36.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.36.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.36.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.36.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.37.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.37.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.37.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.37.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.37.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.37.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.37.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.37.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.37.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.38.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.38.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.38.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.38.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.38.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.38.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.38.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.38.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.38.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.39.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.39.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.39.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.39.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.39.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.39.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.39.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.39.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.39.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.4.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.4.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.4.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.4.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.4.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.4.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.4.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.4.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.4.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.40.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.40.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.40.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.40.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.40.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.40.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.40.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.40.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.40.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.41.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.41.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.41.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.41.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.41.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.41.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.41.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.41.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.41.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.42.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.42.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.42.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.42.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.42.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.42.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.42.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.42.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.42.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.43.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.43.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.43.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.43.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.43.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.43.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.43.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.43.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.43.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.44.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.44.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.44.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.44.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.44.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.44.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.44.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.44.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.44.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.45.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.45.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.45.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.45.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.45.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.45.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.45.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.45.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.45.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.46.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.46.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.46.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.46.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.46.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.46.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.46.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.46.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.46.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.5.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.5.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.5.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.5.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.5.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.5.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.5.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.5.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.5.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.6.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.6.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.6.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.6.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.6.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.6.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.6.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.6.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.6.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.7.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.7.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.7.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.7.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.7.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.7.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.7.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.7.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.7.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.8.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.8.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.8.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.8.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.8.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.8.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.8.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.8.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.8.ffn_norm.weight` | (2048,) | dtypes.float |
| `blk.9.attn_k_b.weight` | (20, 512, 192) | dtypes.float |
| `blk.9.attn_kv_a_mqa.weight` | (576, 2048) | dtypes.float |
| `blk.9.attn_kv_a_norm.weight` | (512,) | dtypes.float |
| `blk.9.attn_norm.weight` | (2048,) | dtypes.float |
| `blk.9.attn_q_a_norm.weight` | (768,) | dtypes.float |
| `blk.9.attn_v_b.weight` | (20, 256, 512) | dtypes.float |
| `blk.9.exp_probs_b.bias` | (64,) | dtypes.float |
| `blk.9.ffn_gate_inp.weight` | (64, 2048) | dtypes.float |
| `blk.9.ffn_norm.weight` | (2048,) | dtypes.float |
| `output_norm.weight` | (2048,) | dtypes.float |

### Quantized Tensors
Total: 422

#### Layer 0 (example)
| Name | Shape | Quant Type |
|------|-------|------------|
| `blk.0.attn_output.weight` | (2048, 5120) | 12 |
| `blk.0.attn_q_a.weight` | (768, 2048) | 12 |
| `blk.0.attn_q_b.weight` | (5120, 768) | 12 |
| `blk.0.ffn_down.weight` | (2048, 10240) | 14 |
| `blk.0.ffn_gate.weight` | (10240, 2048) | 12 |
| `blk.0.ffn_up.weight` | (10240, 2048) | 12 |

#### Non-Layer Tensors
| Name | Shape | Quant Type |
|------|-------|------------|
| `output.weight` | (154880, 2048) | 14 |
| `token_embd.weight` | (154880, 2048) | 12 |

#### All Layers Summary
Number of transformer blocks: 47

#### Tensors per layer
```
blk.{i}.attn_output.weight: (2048, 5120) (type 12)
blk.{i}.attn_q_a.weight: (768, 2048) (type 12)
blk.{i}.attn_q_b.weight: (5120, 768) (type 12)
blk.{i}.ffn_down.weight: (2048, 10240) (type 14)
blk.{i}.ffn_gate.weight: (10240, 2048) (type 12)
blk.{i}.ffn_up.weight: (10240, 2048) (type 12)
```
