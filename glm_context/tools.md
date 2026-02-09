# Commands, Profiling & Workflows

## Model Loading

### From URL
```python
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, SimpleTokenizer, models

model, kv = Transformer.from_gguf(Tensor.from_url(models["glm-4.7:flash"]), max_context=4096, quantized=True)
tok = SimpleTokenizer.from_gguf_kv(kv)
```

### Cache Location
- macOS: `~/Library/Caches/tinygrad/downloads/`
- Linux: `~/.cache/tinygrad/downloads/`

Cache filename = MD5 hash of URL:
```python
import hashlib
hashlib.md5('https://huggingface.co/...'.encode('utf-8')).hexdigest()
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

### Reading GGUF Metadata
```python
from tinygrad.nn.state import gguf_load
kv, weights, quantized = gguf_load('/Users/you/Library/Caches/tinygrad/downloads/<hash>')
for k, v in sorted(kv.items()):
  if 'expert' in k.lower():
    print(f"{k}: {v}")
```

---

## Chat Templates & Presets

| Preset | Format | Models |
|--------|--------|--------|
| `llama3`, `llama-v3`, `llama-bpe` | `<\|start_header_id\|>role<\|end_header_id\|>` | Llama family |
| `qwen2` | `<\|im_start\|>role` | Qwen family |
| `glm4` | `<\|user\|>`, `<\|assistant\|>` with `<sop>` | GLM-4.7 |
| `deepseek-llm` | `User: ` / `Assistant: ` | DeepSeek-V2-Lite |
| `youtu` | `<\|User\|>`, `<\|Assistant\|>` | Youtu-LLM |
| `olmo` | `<\|user\|>`, `<\|assistant\|>` | OLMoE |

### Thinking/Reasoning Models

| Model | Thinking | How to Enable |
|-------|----------|---------------|
| GLM-4.7 | Yes | `<think>\n` after `<\|assistant\|>` |
| Youtu-LLM | Yes | `<think>\n` after `<\|Assistant\|>` |
| DeepSeek-V2-Lite | No | Simple chat format |

`SimpleTokenizer.build_chat_ids()` automatically adds `<think>\n` for glm4 and youtu presets.

### Extract Chat Template from GGUF
```python
from tinygrad import Tensor, nn
kv, _, _ = nn.state.gguf_load(Tensor.from_url(url).to(None))
print(kv.get('tokenizer.chat_template', None))
```

---

## Benchmarking Rules

**NEVER run multiple GPU benchmarks in parallel.** They contest GPU memory and will lock up the machine. Always run sequentially:

### Quick smoke test
```bash
.venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4" --prompt "Hello" --count 5 > ./smoke_test.log 2>&1
```

### Benchmark
```bash
.venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4" --benchmark 20 > ./bench.log 2>&1
```

### Full smoke test (all models)
```bash
python tinygrad/apps/smoke_test.py
```
Tests youtu-llm, deepseek-v2-lite, and glm-4.7:flash with expected token outputs.

---

## Quick Profile (profile_model.py)

Single-command overview of performance, kernel count, scheduling, and hotspots:

```bash
.venv2/bin/python3 profile_model.py deepseek-v2-lite          # default 10 tokens
.venv2/bin/python3 profile_model.py glm-4.7:flash 10
.venv2/bin/python3 profile_model.py youtu-llm:2b-Q4 20

# With extra env vars
.venv2/bin/python3 profile_model.py deepseek-v2-lite 10 MOE_ADDS=0
```

Runs `DEBUG=2 PROFILE=1` under the hood. Produces:
1. **Performance**: per-token ms/tok/s, steady-state average
2. **Kernel count**: total per token, ICB breakdown, avg overhead per kernel
3. **Scheduling**: cache hit/miss stats
4. **Warm-up kernel analysis**: top kernels by total time and call count (real pre-JIT times)
5. **Categories**: elementwise vs reduction vs q4k etc.
6. **Steady-state GPU time**: per-ICB-batch timing from PROFILE pickle
7. **File paths**: raw log, profile pickle, copy-paste commands for deeper investigation

---

## Timing Caveats

- **Warm-up times** (DEBUG=2, pre-JIT): REAL per-kernel times. Best proxy for individual kernel efficiency.
- **Steady-state times** (PROFILE pickle): Per-ICB-batch GPU times are real. Individual kernel times WITHIN an ICB are **evenly divided** across all kernels (artifact of `collect_timestamps()` in `metal.py`). A trivial 6-element kernel shows the same ~37us as a massive Q4K matmul.

---

## Profiling Slow Kernels (VIZ=-1)

VIZ=-1 captures rewrite traces and implicitly enables profiling:

```bash
# Capture
VIZ=-1 .venv2/bin/python3 tinygrad/apps/llm.py --model "deepseek-v2-lite-Q4_0" --benchmark 10

# View slowest kernels
PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --profile --device METAL

# Inspect a specific kernel
PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --profile --device METAL --kernel "<kernel_name>"
```

Output written to `/tmp/profile.pkl.<user>` and `/tmp/rewrites.pkl.<user>`.
Also available via profile_model.py

---

## Kernel Analysis with DEBUG=5

DEBUG=5 prints **Metal source code** for every kernel:

```bash
DEBUG=5 .venv2/bin/python tinygrad/apps/llm.py --model "deepseek-v2-lite-Q4_0" --benchmark 3 > ./debug5.log 2>&1
```

Find a specific kernel:
```bash
grep -n "kernel void r_9_32" debug5.log
```

### What to look for
- **Grid dims**: `gidx0`, `gidx1` → workgroups. Need ~2000+ for bandwidth saturation.
- **Threadgroup dims**: `lidx0`, `lidx1` → threads per workgroup. 64-256 typical.
- **Accumulators**: `acc0[N]` → N outputs per thread (UPCAST=N).
- **Reduction loops**: `for (int Ridx...)` → serial work per thread. Long loops = missing GROUPTOP.
- **Byte reads**: `unsigned char val = *(data+...)` → quantized dequant, scattered.
- **Threadgroup memory**: `threadgroup float` → GROUPTOP/GROUP is active (good).

### Red flags
- **< 20 GB/s bandwidth**: Too few workgroups OR scattered memory access
- **No threadgroup reduction**: Each thread does full reduction alone
- **UPCAST but no GROUP**: Thread writes N elements but reads full reduction serially

---

## JIT Batching Analysis

```bash
DEBUG=2 .venv2/bin/python tinygrad/apps/llm.py --model "youtu-llm:2b-Q4" --benchmark 5 > ./debug2.log 2>&1
# Then check:
grep "jit execs" debug2.log
grep "JIT GRAPHing" debug2.log
```

---

## Understanding Kernel Optimization (heuristic.py)

tinygrad's optimizer applies opts in priority order:

1. **Tensor Cores** (`USE_TC > 0`) — skipped for most quantized kernels
2. **Matvec (MV)** — detects `MUL(INDEX, INDEX)` pattern, applies GROUP+LOCAL+UPCAST
   - Controlled by `MV_BLOCKSIZE=4`, `MV_THREADS_PER_ROW=8`, `MV_ROWS_PER_THREAD=4`
   - **Fails for fused dequant+matmul** because pattern is `MUL(dequant_chain, INDEX)`
3. **GROUPTOP** — threadgroup cooperative reduction, only if output upcast dims ≤ 2048, only size 16
4. **Masked UPCAST** — for WHERE-gated axes ≤ 7
5. **More UPCAST** — aggressive, while upcast_size < 32
6. **UNROLL** — last reduce dim if ≤ 32
7. **LOCAL** — assign threads to output dims, up to 128 total

### Debugging opt decisions
```bash
DEBUG=3 .venv2/bin/python tinygrad/apps/llm.py --model "deepseek-v2-lite-Q4_0" --benchmark 1 2>&1 | grep "MATVEC\|GROUPTOP"
```

### Pattern matching profiling
```bash
TRACK_MATCH_STATS=2 PYTHONPATH="." python3 test/external/external_benchmark_schedule.py
```

---

## Troubleshooting

### Wrong Output / Model Not Following Instructions
```bash
# With chat formatting (recommended)
python llm.py --model "glm-4.7:flash" --prompt "Hello"
# Raw prompt (no chat template)
python llm.py --model "glm-4.7:flash" --prompt "Hello" --raw-prompt
```

### Model Produces Garbage
1. Check tokenizer preset: `tokenizer.ggml.pre` in GGUF
2. Check BOS token: `kv.get('tokenizer.ggml.add_bos_token', True)`
3. Check GGUF source: bartowski Q4_0 was broken, unsloth works

### Memory Issues
1. Use `quantized=True` for large MoE models
2. Reduce `max_context`
3. Models auto-quantized if they start with "glm-" or "deepseek-"

### Slow First Token
Due to model loading, JIT compilation, and RoPE precomputation. Subsequent tokens use cached kernels.

---

## Available Models

See `models` dict in `tinygrad/apps/llm.py`:
- Llama 3.2 (1B, 3B), Llama 3.1 (8B)
- Qwen3 (0.6B, 1.7B, 8B, 30B-A3B MoE)
- OLMoE (1B-7B)
- GLM-4.7-Flash
- DeepSeek-V2-Lite
- Youtu-LLM (2B)
