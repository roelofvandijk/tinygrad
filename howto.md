# LLM Apps How-To Guide

## Model Loading

### From URL (Recommended)
```python
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, SimpleTokenizer, models

# Load from predefined models dict
model, kv = Transformer.from_gguf(Tensor.from_url(models["glm-4.7:flash"]), max_context=4096, quantized=True)
tok = SimpleTokenizer.from_gguf_kv(kv)
```

### Cache Location
Downloaded models are cached at:
- macOS: `~/Library/Caches/tinygrad/downloads/`
- Linux: `~/.cache/tinygrad/downloads/`

Cache filename is MD5 hash of the URL:
```python
import hashlib
url = "https://huggingface.co/..."
cache_name = hashlib.md5(url.encode('utf-8')).hexdigest()
```

### Pre-populating Cache
To avoid re-downloading, move existing GGUF files to cache with correct hash name:
```bash
# Get the hash for a URL
python -c "import hashlib; print(hashlib.md5('https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf'.encode()).hexdigest())"
# 946c72affc2a08d1db03146bcb52c03a

# Move file to cache
mv GLM-4.7-Flash-Q4_K_M.gguf ~/Library/Caches/tinygrad/downloads/946c72affc2a08d1db03146bcb52c03a
```

### Reading GGUF Metadata from Cache
```python
from tinygrad.nn.state import gguf_load
# Load directly from cache path (hash of URL)
kv, weights, quantized = gguf_load('/Users/you/Library/Caches/tinygrad/downloads/<hash>')

# View expert config
for k, v in sorted(kv.items()):
  if 'expert' in k.lower():
    print(f"{k}: {v}")
```

Example for deepseek-v2-lite:
```
deepseek2.expert_count: 64
deepseek2.expert_used_count: 4      # only 4 of 64 experts active per token
deepseek2.expert_feed_forward_length: 1536
deepseek2.expert_gating_func: 2     # sigmoid gating
```

## Chat Templates & Presets

### Tokenizer Presets
Each model uses a preset that defines chat formatting:
- `llama3`, `llama-v3`, `llama-bpe`: Llama-style `<|start_header_id|>role<|end_header_id|>`
- `qwen2`: Qwen-style `<|im_start|>role`
- `glm4`: GLM-style `<|user|>`, `<|assistant|>` with `<sop>` prefix
- `deepseek-llm`: Simple `User: ` / `Assistant: ` format
- `youtu`: Youtu-style `<|User|>`, `<|Assistant|>`
- `olmo`: OLMo-style `<|user|>`, `<|assistant|>`

### Thinking/Reasoning Models
Some models support thinking mode with `<think>` tags:

| Model | Thinking Mode | How to Enable |
|-------|--------------|---------------|
| GLM-4.7 | Yes | Add `<think>\n` after `<\|assistant\|>` |
| Youtu-LLM | Yes | Add `<think>\n` after `<\|Assistant\|>` (or model generates naturally) |
| DeepSeek-V2-Lite | No | Simple chat format, no thinking |

The `SimpleTokenizer.build_chat_ids()` method automatically adds `<think>\n` for glm4 and youtu presets.

### Extracting Chat Template from GGUF
```python
from tinygrad import Tensor, nn
kv, _, _ = nn.state.gguf_load(Tensor.from_url(url).to(None))
template = kv.get('tokenizer.chat_template', None)
print(template)
```

## Troubleshooting

### Wrong Output / Model Not Following Instructions
1. **Check chat formatting**: Raw prompts vs chat-formatted prompts produce different results
   ```bash
   # With chat formatting (recommended)
   python llm.py --model "glm-4.7:flash" --prompt "Hello"

   # Raw prompt (no chat template)
   python llm.py --model "glm-4.7:flash" --prompt "Hello" --raw-prompt
   ```

2. **Check thinking mode**: GLM4 without `<think>` prefix answers differently
   - With `<think>`: Produces reasoning steps like "1. Identify the question..."
   - Without `<think>`: May just list answers directly

### Model Produces Garbage
1. **Check tokenizer preset**: Verify `tokenizer.ggml.pre` in GGUF matches expected preset
2. **Check BOS token**: Some models need BOS, others don't
   ```python
   add_bos = kv.get('tokenizer.ggml.add_bos_token', True)
   bos_id = kv.get('tokenizer.ggml.bos_token_id') if add_bos else None
   ```

### Memory Issues
1. Use `quantized=True` for large MoE models (GLM, DeepSeek)
2. Reduce `max_context` to lower memory usage
3. Models are auto-quantized if they start with "glm-" or "deepseek-"

### Slow First Token
First token is slow due to:
1. Model loading and weight initialization
2. JIT compilation (subsequent tokens use cached kernels)
3. RoPE frequency precomputation

## Architecture Notes

### MLA (Multi-head Latent Attention)
Used by DeepSeek-V2 and GLM-4.7 models. Key indicators:
- `kv_lora_rank > 0` in GGUF metadata
- Uses compressed KV cache (smaller memory footprint)
- Has `attn_kv_a_mqa`, `attn_kv_b` weight naming

### MoE (Mixture of Experts)
Used by OLMoE, GLM-4.7, DeepSeek-V2-Lite. Key indicators:
- `expert_count > 0` in GGUF metadata
- Weight names contain `_exps` (e.g., `ffn_gate_exps`)
- `expert_used_count` defines how many experts per token

### YaRN (Yet another RoPE extensioN)
Extended context via RoPE frequency scaling. Key indicators:
- `rope.scaling.type == "yarn"` or `rope.scaling.factor > 1`
- Uses `precompute_freqs_cis_yarn()` for frequency computation
- `mscale` adjusts attention scaling for long contexts

## Running the Smoke Test
```bash
python tinygrad/apps/smoke_test.py
```
Tests youtu-llm, deepseek-v2-lite, and glm-4.7:flash with expected token outputs.

## Quick Profile (Recommended)

Use `profile_model.py` for a single-command overview of performance, kernel count, scheduling, and hotspots:

```bash
# Full profile report
.venv2/bin/python3 profile_model.py deepseek-v2-lite          # default 10 tokens
.venv2/bin/python3 profile_model.py glm-4.7:flash 10
.venv2/bin/python3 profile_model.py youtu-llm:2b-Q4 20

# With extra env vars (e.g. toggle features)
.venv2/bin/python3 profile_model.py deepseek-v2-lite 10 MOE_ADDS=0
```

Runs `DEBUG=2 PROFILE=1` under the hood, parses both the debug log and the profile pickle, and produces:
1. **Performance**: per-token ms/tok/s, steady-state average
2. **Kernel count**: total per token, ICB breakdown, avg overhead per kernel
3. **Scheduling**: cache hit/miss stats, per-token schedule patterns
4. **Warm-up kernel analysis**: top kernels by total time and call count (real pre-JIT times)
5. **Categories**: elementwise vs reduction vs q4k etc., with per-token estimates
6. **Steady-state GPU time**: per-ICB-batch timing from PROFILE pickle
7. **File paths**: raw log, profile pickle, rewrites pickle, and copy-paste commands to dig deeper

The report footer shows paths and commands for deeper investigation:
- `extra/viz/cli.py --profile` for aggregated kernel timing
- `extra/viz/cli.py --kernel` for rewrite traces (requires VIZ=-1 capture)
- `grep` commands to find Metal source in the debug log
- `DEBUG=5` command to capture full Metal source of all kernels

### NOTE on timing
- **Warm-up times** (DEBUG=2, pre-JIT): REAL per-kernel times. Best proxy for individual kernel efficiency.
- **Steady-state times** (PROFILE pickle): Per-ICB-batch GPU times are real. Individual kernel times WITHIN an ICB are evenly divided across all kernels (artifact of `collect_timestamps()` in `metal.py`).

## Profiling Slow Kernels (VIZ=-1)

Use VIZ=-1 to capture rewrite traces (and it implicitly enables profiling). The profile is written to:
- `/tmp/profile.pkl.<user>` (actual path varies by OS, shown in `profile_model.py` output)
- `/tmp/rewrites.pkl.<user>`

### Capture a profile
```bash
VIZ=-1 .venv2/bin/python3 tinygrad/apps/llm.py --model "deepseek-v2-lite" --benchmark 10
```

### View the slowest kernels
```bash
PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --profile --device METAL
```

### Inspect a specific kernel (top 10 occurrences)
```bash
PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --profile --device METAL --kernel "<kernel_name>"
```

## Kernel Analysis with DEBUG=5

DEBUG=5 prints the **Metal source code** for every kernel. Essential for understanding why a kernel is slow.

### Capture kernel sources
```bash
DEBUG=5 .venv2/bin/python tinygrad/apps/llm.py --model "deepseek-v2-lite" --benchmark 3 > ./debug5.log 2>&1
```

### Find a specific kernel's Metal source
```bash
# Search by kernel name (from VIZ=-1 profile)
grep -n "kernel void r_9_32" debug5.log
# Then read the source at that line number
```

### What to look for in kernel source
- **Grid dims**: `gidx0`, `gidx1` → workgroups. Need ~2000+ for bandwidth saturation on Apple Silicon.
- **Threadgroup dims**: `lidx0`, `lidx1` → threads per workgroup. 64-256 typical.
- **Accumulators**: `acc0[N]` → N outputs per thread (UPCAST=N).
- **Reduction loops**: `for (int Ridx...)` → serial work per thread. Long loops = missing GROUPTOP.
- **Byte reads**: `unsigned char val = *(data+...)` → quantized dequant, inherently scattered.
- **Threadgroup memory**: `threadgroup float` → GROUPTOP/GROUP is active (good).

### Red flags
- **< 20 GB/s bandwidth**: Too few workgroups OR scattered memory access
- **No threadgroup reduction**: Each thread does full reduction alone. Missing GROUPTOP/GROUP.
- **UPCAST but no GROUP**: Thread writes N elements but reads the full reduction serially.

## Understanding Kernel Optimization (heuristics.py)

tinygrad's optimizer (`tinygrad/codegen/opt/heuristic.py`) applies opts in priority order:

1. **Tensor Cores** (`USE_TC > 0`) — skipped for most quantized kernels
2. **Matvec (MV)** — detects `MUL(INDEX, INDEX)` pattern, applies GROUP+LOCAL+UPCAST
   - Controlled by `MV_BLOCKSIZE=4`, `MV_THREADS_PER_ROW=8`, `MV_ROWS_PER_THREAD=4`
   - **Fails for fused dequant+matmul** because pattern is `MUL(dequant_chain, INDEX)` not `MUL(INDEX, INDEX)`
3. **GROUPTOP** — threadgroup-cooperative reduction, only if output upcast dims ≤ 2048, only size 16
4. **Masked UPCAST** — for WHERE-gated axes ≤ 7
5. **More UPCAST** — aggressive, while upcast_size < 32
6. **UNROLL** — last reduce dim if ≤ 32
7. **LOCAL** — assign threads to output dims, up to 128 total

### Why fused Q5K MoE kernels are slow
The MV heuristic requires `mulop.src[0].op is Ops.INDEX and mulop.src[1].op is Ops.INDEX`. Q5K dequant inlines
bitwise ops between the INDEX and the MUL, breaking pattern match. Falls through to generic opts → no GROUP.
Result: serial 1536-element reduction per thread at 2 GB/s.

### Debugging opt decisions
```bash
# See which opts were applied
DEBUG=3 .venv2/bin/python tinygrad/apps/llm.py --model "deepseek-v2-lite" --benchmark 1 2>&1 | grep "MATVEC\|GROUPTOP"
```

## Benchmarking Rules

**NEVER run multiple GPU benchmarks in parallel.** They contest GPU memory and will lock up the machine.
Always run benchmarks sequentially, one at a time:
```bash
# CORRECT: sequential
.venv2/bin/python tinygrad/apps/llm.py --model deepseek-v2-lite --benchmark 10 > ./bench1.log 2>&1
# wait for it to finish, then:
.venv2/bin/python tinygrad/apps/llm.py --model deepseek-v2-lite --benchmark 10 > ./bench2.log 2>&1
```

```bash
# WRONG: parallel GPU benchmarks — WILL CRASH THE MACHINE
command1 & command2 &  # DO NOT DO THIS
```

## Available Models
See `models` dict in llm.py for full list:
- Llama 3.2 (1B, 3B)
- Llama 3.1 (8B)
- Qwen3 (0.6B, 1.7B, 8B, 30B-A3B MoE)
- OLMoE (1B-7B)
- GLM-4.7-Flash
- DeepSeek-V2-Lite
- Youtu-LLM (2B)
