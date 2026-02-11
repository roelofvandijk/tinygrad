#!/usr/bin/env python3
import hashlib
import os
import pathlib

# Model URLs from llm.py
models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "glm-4.7:flash-unsloth-Q4_0": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_0.gguf",
  "glm-4.7:flash-unsloth-Q6_K": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q6_K.gguf",
  "glm-4.7:flash-unsloth-Q4_K_M": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
  "glm-4.7:flash-bartowski-Q4_0": "https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF/resolve/main/zai-org_GLM-4.7-Flash-Q4_0.gguf",
  "deepseek-v2-lite-Q4_K_M": "https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/main/DeepSeek-V2-Lite.Q4_K_M.gguf",
  "deepseek-v2-lite-Q4_0": "https://huggingface.co/zhentaoyu/DeepSeek-V2-Lite-Chat-Q4_0-GGUF/resolve/main/deepseek-v2-lite-chat-q4_0.gguf",
  "youtu-llm:2b-Q4_K_M": "https://huggingface.co/AaryanK/Youtu-LLM-2B-GGUF/resolve/main/Youtu-LLM-2B.q4_k_m.gguf",
  "youtu-llm:2b-Q4_0": "https://huggingface.co/AaryanK/Youtu-LLM-2B-GGUF/resolve/main/Youtu-LLM-2B.q4_0.gguf",
}

# Get cache directory
cache_dir = pathlib.Path(os.path.expanduser("~/Library/Caches/tinygrad/downloads"))
models_dir = pathlib.Path("./models")
models_dir.mkdir(exist_ok=True)

# Create hash -> name mapping
hash_to_names = {}
for name, url in models.items():
  url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
  if url_hash not in hash_to_names:
    hash_to_names[url_hash] = []
  hash_to_names[url_hash].append(name)

# Create symlinks for existing files
found = 0
for hash_name, model_names in hash_to_names.items():
  cache_file = cache_dir / hash_name
  if cache_file.exists():
    found += 1
    size_gb = cache_file.stat().st_size / (1024**3)
    print(f"✓ Found {hash_name} ({size_gb:.1f} GB)")
    for model_name in model_names:
      # Create symlink with .gguf extension
      link_path = models_dir / f"{model_name}.gguf"
      if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
      link_path.symlink_to(cache_file)
      print(f"  → {link_path}")
  else:
    print(f"✗ Missing {model_names[0]} (hash: {hash_name})")

print(f"\nCreated symlinks for {found}/{len(hash_to_names)} unique models in ./models/")
