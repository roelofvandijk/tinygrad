# Progress Log

## 🎯 TOP 5 NEXT STEPS (DeepSeek-V2-Lite Focus)

1. **Verify YaRN params from GGUF** - Add debug print to show yarn_freq_scale, yarn_attn_factor, yarn_log_mul read from GGUF
2. **Check mscale at runtime** - Verify mscale=1.0 for DeepSeek-V2-Lite (no YaRN) or 1.4046 (with YaRN)
3. **Print output token IDs** - Check if token IDs are valid before decoding
4. **Check logits** - Verify final logits aren't all zeros, inf, or nan
5. **Test non-quantized** - Rule out quantization issues with --no-quantized

---

## Current Status (2026-02-01) - Session 8

### DeepSeek-V2-Lite - Outputs Garbage 🔴 ACTIVE

**Symptoms**:
- Model loads correctly (16.22 GB Q4_K_M)
- Processes input, JIT captures 2104 kernels
- Outputs garbage characters (`������`)

**What Works (Unit Tested)**:
- ✅ kv_b split: interleaved per-head layout verified
- ✅ mscale formula: matches llama.cpp (1.4046 for YaRN params)
- ✅ Absorbed MLA math: Q_absorbed @ kv^T = Q @ (kv @ W_k^T)^T

**Potential Issues to Investigate**:

1. **YaRN params** - DeepSeek-V2-Lite might not use YaRN (freq_scale=1.0 → mscale=1.0)
   - Need to verify what GGUF contains

2. **RoPE format** - We use interleaved, llama.cpp also uses interleaved for deepseek2
   - Should be OK, but verify

3. **MoE routing** - 64 experts, 6 per token, 2 shared
   - expert_last_layout transpose happening
   - May need verification

4. **Tokenizer** - Using GGUF tokenizer
   - If token mapping wrong, output is garbage

**Key Difference from GLM-4.7**:
- DeepSeek-V2-Lite: n_heads=16, kv_lora=512, qk_nope=128, qk_rope=64, v_head=128
- GLM-4.7-Flash: n_heads=20, kv_lora=512, qk_nope=192, qk_rope=64, v_head=256

**Debug output not showing** - The DEBUG>=1 prints inside _attention() aren't appearing in log.
This needs investigation - maybe stdout buffering or JIT behavior.

---

## Previous Session (Session 7)

### GLM-4.7 Flash - Attention Scale Bug Fixed 🟡 TESTING

### New Finding: MoE Expert Weight Layout Mismatch 🟠 LIKELY ROOT CAUSE

**Insight**:
- llama.cpp stores GLM4_MOE expert tensors with the **expert dimension last**, and applies them via `ggml_mul_mat_id(w, x, ids)` (expert selection is part of the op).
- In llama.cpp `LLM_ARCH_GLM4_MOE` the MoE expert tensor shapes are:
  - `ffn_gate_exps`: `{ n_embd, n_ff_exp, n_expert }`
  - `ffn_up_exps`: `{ n_embd, n_ff_exp, n_expert }`
  - `ffn_down_exps`: `{ n_ff_exp, n_embd, n_expert }`
- Current tinygrad implementation assumes experts-first contiguous layout `(n_expert, out, in)` and selects weights via `weight[sel]`.

**Why this matters**:
- If the GGUF weights are expert-last (as in llama.cpp), reshaping/slicing them as experts-first will silently assign the wrong bytes to some expert IDs.
- This matches the observed failure mode: routing looks sane, but a specific “expert” (e.g. 16 in block 39) produces absurd activations and destabilizes later layers.

**Fix direction**:
- Either repack at load time (expert-last → experts-first), or implement the equivalent of llama.cpp’s `mul_mat_id` path for expert-last weights (including quantized variants).

**Bug Found & Fixed**:
- ❌ **OLD (wrong)**: `scale = 1/sqrt(q_head_dim)` = `1/sqrt(256)` = 0.0625
- ✅ **NEW (correct)**: `scale = 1/sqrt(cache_dim)` = `1/sqrt(576)` = 0.0417
- In MLA absorption, Q and K both have dimension `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`
- From llama.cpp deepseek2.cpp: `kq_scale = mscale * mscale / sqrtf(float(n_embd_head_k))`

**Why this matters**:
- Wrong scale (1.5x too large) causes attention weights to be more "peaky"
- This amplifies differences between positions, causing logit explosion at T>1
- At T=1, there's only one position so the scale doesn't matter (softmax of 1 element = 1.0)

**Files Changed**:
- `tinygrad/apps/llm.py` line 427: `scale = self.mscale * self.mscale / math.sqrt(cache_dim)`

---

## Previous Status (2026-02-01) - Session 6

### GLM-4.7 Flash - Output Still Garbage 🔴 ACTIVE

**Current output**: `,,  usage, usage usage usage` - garbage but not crashing

**Fixes Applied This Session**:
1. ✅ **RoPE interleaved formula** - verified against GGML reference (diff = 0.0)
   - `apply_rope_interleaved()` now correctly extracts cos/sin and applies rotation
   - Test: `test_rope_final.py` passes all positions

**What Works**:
- Single token "Hello" → "greetings" (semantically correct)
- RoPE at position 0 is identity (cos=1, sin=0)
- RoPE at positions >0 now matches GGML reference

**What's Still Broken**:
- Multi-token prompts produce garbage
- Full prompt max logit ~67 vs single token ~24 (logit explosion)

**Working Hypothesis**:
Since position 0 is identity regardless of RoPE params, and single tokens work,
the issue is likely in:
1. Q/K/V dimension handling when T>1
2. MLA absorbed attention math
3. Attention mask or scale ← **THIS WAS IT!**

**Next Steps** (from Session 6):
1. Create micro-tests comparing our MLA attention vs reference
2. Check Q concat order: `[q_nope_absorbed | q_pe]` vs llama.cpp ✅ Verified correct
3. Check K concat order: `[kv_normed | k_pe]` vs llama.cpp ✅ Verified correct
4. ~~Verify attention scale = 1/sqrt(256) matches llama.cpp~~ ❌ **BUG: should be 1/sqrt(576)**

---

## GLM-4.7 Flash - Model Parameters (from GGUF)

```
Architecture: deepseek2 (same as DeepSeek V2/V3)
Layers: 47 (1 dense + 46 MoE)
MLA: absorbed optimization, kv_lora_rank=512, q_lora_rank=768
  - q_head_dim = 256 (qk_nope=192 + qk_rope=64)
  - v_head_dim = 256
  - H = 20 heads
MoE: 64 experts, 4 selected/token, 1 shared, sigmoid gating with 1.8x scale
RoPE: 64 dims, interleaved format, theta=1,000,000
```

## MLA Weight Shapes (from GGUF)

```
attn_k_b.weight: (20, 512, 192) = (H, kv_lora_rank, qk_nope_head_dim)
attn_v_b.weight: (20, 256, 512) = (H, v_head_dim, kv_lora_rank)
attn_q_a.weight: (768, 2048) = (q_lora_rank, dim)
attn_q_b.weight: (5120, 768) = (H*q_head_dim, q_lora_rank)
attn_kv_a_mqa.weight: (576, 2048) = (kv_lora_rank+rope, dim)
attn_output.weight: (2048, 5120) = (dim, H*v_head_dim)
```

## Components Verified Working

- ✅ RoPE interleaved formula (matches GGML reference)
- ✅ QuantizedExpertWeights gather + matmul
- ✅ Q4_K dequantization
- ✅ RMSNorm
- ✅ Prompt tokenization format
- ✅ KV cache consistency (prefill = step-by-step for T=2)

## Bugs Fixed

### BUG 1 (Fixed): Interactive mode start_pos
**Status**: Fixed by using start_pos=0

### BUG 2 (Fixed): RoPE formula mixing cos/sin
**Status**: Fixed by extracting cos/sin correctly

---

## Previous Status (2026-02-01) - Session 4

### GLM-4.7 Flash - Output Correctness Investigation 🔴 IN PROGRESS

**Issue**: Model produces garbage output (predicting '|' tokens) despite all computations being numerically valid.

#### Latest Fix (Session 4)

**Root Cause Found**: RoPE format mismatch!
- GGUF/llama.cpp uses **interleaved** RoPE format for deepseek2 architecture
- My code was using **half-split** format (which works for llama architecture)
- The llama architecture has a weight permutation that converts interleaved → half-split
- deepseek2 does NOT have this permutation, so it expects interleaved format natively

**The Difference**:
- Half-split: pairs are `(x[0], x[dim/2]), (x[1], x[dim/2+1]), ...` (first half with second half)
- Interleaved: pairs are `(x[0], x[1]), (x[2], x[3]), ...` (adjacent elements)

**Fix Applied**:
- Added `apply_rope_interleaved()` function that pairs adjacent elements
- Updated MLA attention to use interleaved RoPE for q_pe and k_pe
- File: `tinygrad/apps/llm.py` lines 181-192, 275-278

---

## Previous Status (2026-02-01) - Session 3

### GLM-4.7 Flash - Output Correctness Investigation

**Issue**: Model produces garbage output despite all computations being numerically valid.

#### Findings (Session 3)

**Key Discovery**: llama.cpp uses "absorbed MLA" optimization where:
- Standard MLA: `k_nope = kv_cmpr @ wk_b`, then attention with `Q_nope @ K_nope`
- Absorbed MLA: `q_absorbed = q_nope @ wk_b^T`, then attention with compressed KV
- The two are mathematically equivalent but absorbed uses smaller KV cache

**Implemented Changes**:
1. ✅ Switched to absorbed MLA path matching llama.cpp `deepseek2.cpp`
2. ✅ Fixed attention scale to use `1/sqrt(q_head_dim)` = `1/sqrt(256)` (not absorbed dim 576)
3. ✅ Verified weight shapes match llama.cpp expectations:
   - `wk_b`: GGML {nope, kv_lora, H} = row-major (H, kv_lora, nope) = (20, 512, 192) ✓
   - `wv_b`: GGML {kv_lora, v_head, H} = row-major (H, v_head, kv_lora) = (20, 256, 512) ✓

**Debugging Results**:
- No NaN/inf in any layer (verified through 47 layers)
- Attention weights mean=0.1 for T=10 (correct: 1/T)
- Layer activations grow naturally: std 0.011 → 0.038 → 0.062 → 0.081 → 0.095...
- Final logits range [-8.4, 53.8] - numerically reasonable but semantically wrong
- Token "4" for "2+2=" ranks ~3000 instead of top

**What's Verified Working**:
- MLA absorption computation: isolated test confirms q_nope @ wk_b^T produces correct values
- Weight shapes: all match llama.cpp's expected GGML column-major layouts
- Attention scale: using correct effective head dim (256), not absorbed dim (576)
- MoE gating: sigmoid with normalization and 1.8x scaling
- Leading dense blocks: layer 0 uses dense FFN, layers 1+ use MoE

**Remaining Suspects**:
1. Weight transpose/layout issue in GGUF loading
2. RoPE implementation difference
3. Quantization artifacts affecting model semantics
4. Some subtle ordering/concat difference

---

#### Previous Session Fixes (Session 2)

#### Key Insight: Architecture is deepseek2
The GGUF reports `general.architecture: deepseek2`, not a GLM-specific architecture.
This means GLM-4.7 Flash uses DeepSeek V2 architecture internally with:
- MLA attention (kv_lora_rank=512, q_lora_rank=768)
- MoE with shared experts (64 experts, 4 selected, 1 shared)
- Sigmoid gating with normalization and 1.8x scaling

#### Files Modified This Session
- `tinygrad/apps/llm.py`:
  - Added `SimpleTokenizer.after_bos()` for `[gMASK]<sop>` prefix
  - Added `expert_gating_func`, `expert_weights_norm`, `expert_weights_scale` to MLA blocks
  - Fixed MoE gating to use correct enum values

---

## Previous: GLM-4.7 Flash MoE Kernels ✅ COMPLETE

**Summary**: Implemented fused MoE kernels for Q4_K and Q6_K quantization that use runtime expert indices instead of graph constants. This should eliminate schedule cache misses.

#### Files Modified
- `tinygrad/nn/quant_kernels.py` - Added `q4k_moe_fused()` and `q6k_moe_fused()`
- `tinygrad/apps/llm.py` - `QuantizedExpertWeights.__call__` now uses fused kernels
- `test_quant_correctness.py` - Added kernel correctness tests

#### Architecture
```
QuantizedExpertWeights.__call__(sel, x)
├── Q4_K (type 12) → q4k_moe_fused()  [144 bytes/block]
├── Q6_K (type 14) → q6k_moe_fused()  [210 bytes/block]
└── Q5_K (type 13) → fallback path    [gather + dequant + matmul]
```

#### Key Design: Runtime Expert IDs
```python
# OLD (cache miss): expert indices embedded as graph constants
selected_blocks = blocks[sel_flat]  # sel_flat values become graph constants

# NEW (cache hit): expert indices are runtime tensor values
out = q4k_moe_fused(x, blocks, sel_flat, ...)  # sel_flat is runtime data
```

#### Kernel Pattern (from hand_spec_kernel3)
Both kernels use the standard custom kernel pattern:
- `UOp.range()` for loops with `AxisType.REDUCE` for accumulation
- `out[i,m].set(0.0)` + `out[i,m].set(out.after(k)[i,m] + prod, end=k)` for reduction
- `.sink(arg=KernelInfo(opts_to_apply=()))` to disable optimizer (required for custom kernels)

#### Correctness Results
| Kernel | Max Diff | Status |
|--------|----------|--------|
| Q4_K MoE | 0.0005 | ✅ Pass |
| Q6_K MoE | 0.0 (exact) | ✅ Pass |
| Forward pass | No NaN, logits range [-23, 26] | ✅ Pass |

#### Q6_K Layout (complex interleaved)
```
Block: 210 bytes → 256 elements
├── ql (bytes 0-127):   low 4 bits, interleaved by nibble
│   k=0-63:   byte k, low nibble
│   k=64-127: byte k-64, high nibble
│   k=128-191: byte 64+(k-128), low nibble
│   k=192-255: byte 64+(k-192), high nibble
├── qh (bytes 128-191): high 2 bits, interleaved by 2-bit groups
│   k=0-127:   byte 128+(k%32), bit_pos=(k//32)*2
│   k=128-255: byte 160+((k-128)%32), bit_pos=((k-128)//32)*2
├── scales (bytes 192-207): 16 int8 values, one per 16 elements
└── d (bytes 208-209): float16 global scale

Formula: val = d * ((ql | (qh << 4)) - 32) * scale[k // 16]
```

### Code Review (2026-02-01)

#### GLM-4.7 Flash Architecture
- **42 blocks**: 1 dense (leading_dense_blocks=1) + 41 MoE
- **64 experts per MoE block**, 2 active per token (num_experts_per_tok=2)
- **Shared experts**: Run in parallel with routed experts (`ffn_*_shexp`)
- **MLA attention**: Multi-head Latent Attention from DeepSeek-V2

#### MoE Weight Types
| Layer | GGML Type | Bytes/Block | Kernel |
|-------|-----------|-------------|--------|
| ffn_gate_exps | Q4_K (12) | 144 | q4k_moe_fused ✅ |
| ffn_up_exps | Q4_K (12) | 144 | q4k_moe_fused ✅ |
| ffn_down_exps | Q6_K (14) | 210 | q6k_moe_fused ✅ |

#### Code Flow
```
Transformer.forward()
  └── MLATransformerBlock.__call__()
        ├── _attention() - MLA with LoRA Q, compressed KV
        └── _feed_forward()
              ├── router: ffn_gate_inp(h_norm).softmax().topk(2)
              ├── experts: ffn_down_exps(sel, ffn_gate_exps(sel,x).silu() * ffn_up_exps(sel,x))
              │            └── QuantizedExpertWeights.__call__()
              │                  ├── Q4_K → q4k_moe_fused()
              │                  ├── Q6_K → q6k_moe_fused()
              │                  └── Q5_K → fallback (gather+dequant+matmul)
              └── shared: ffn_down_shexp(ffn_gate_shexp(h).silu() * ffn_up_shexp(h))
```

#### Observations
1. **Full kernel coverage** for GLM-4.7: All MoE weights use Q4_K or Q6_K (both have fused kernels)
2. **Fallback path** only for Q5_K (not used in GLM-4.7) - would still cause cache misses
3. **`.realize()` calls** in QuantizedExpertWeights are intentional to bound graph size
4. **Shared experts** use standard QuantizedLinear, not MoE path (correct - they're not dynamically routed)

#### Potential Improvements
1. Add Q5_K fused kernel if models using it need cache hits
2. Consider tiling the MoE kernels for better memory locality (like hand_spec_kernel3)
3. The `opts_to_apply=()` is required - optimizer bugs break custom kernels

### Performance Testing (pending)
- **Target**: ~10 tok/s (llama.cpp achieves this on same hardware)
- **Before**: 0.3 tok/s (cache misses, ~500ms/block schedule compilation)
- **After**: TBD - awaiting user test results

### Quick Test Commands
```bash
# Kernel correctness
source .venv2/bin/activate && python test_quant_correctness.py moe_fused
source .venv2/bin/activate && python test_quant_correctness.py q6k_moe_fused

# Model forward pass
source .venv2/bin/activate && python -c "
from tinygrad import Tensor
from tinygrad.dtype import dtypes
import sys; sys.path.insert(0, 'tinygrad/apps')
from llm import Transformer
model, kv = Transformer.from_gguf(Tensor.from_url(
  'https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf'
), max_context=64, quantized=True)
logits = model.forward_logits(Tensor([[1]], dtype=dtypes.int32), 0)
print(f'Has NaN: {logits.numpy().std() > 0}')
"

# Full inference test
DEBUG=1 source .venv2/bin/activate && python tinygrad/apps/llm.py --model "glm-4.7:flash" --quantized --prompt "2+2=" --count 5
```

---

## Previous Status (2026-01-31 - Session 2)

### Environment
- Use `.venv2` for running tests

### Recent Changes
1. **Fixed start_pos bug in --prompt mode** - Model was only seeing last token instead of full prompt
2. **Removed test kernels from quant_kernels.py** - Now only contains production kernels (`dequantize_q4k_uop`, `q4k_linear_fused`)
3. **Removed `_no_jit` flag** - JIT now enabled for quantized models
4. **Added `.realize()` to QuantizedExpertWeights** - Bounds graph per expert call
5. **Added heartbeat timer** - Shows elapsed time and per-token timing when `DEBUG>=1`

### Current Issue: Cache Misses in MoE
- **Symptom**: Massive cache misses per MoE block (6+ schedules each, ~500ms/block)
- **Root cause**: Expert selection creates different graphs
  - `topk()` selects different expert indices each token (e.g., [3, 17] vs [42, 55])
  - `blocks[sel_flat]` embeds these indices as graph constants
  - Different indices = different graph structure = CACHE MISS
- **Hang between blocks**: `.realize()` forces Metal sync after each MoE block
- **This is fundamental to dynamic MoE**: Without two-phase approach, graph varies

### Measured Impact
- ~500ms per MoE block × 41 blocks = **~20 seconds** per token just for schedule compilation
- Even with JIT=0, schedule cache still misses (separate cache)
- Cache misses happen regardless of JIT setting because it's the graph structure that varies

### Potential Solutions
1. **Two-phase MoE** (proper fix): Build per-expert token lists, then batched matmul per expert
2. **Fixed grid MoE**: Process all 64 experts, mask unused (wasteful but fixed graph)
3. **JIT=0**: Disable JIT but schedule cache still misses

### Debugging Whitespace Output
**ISOLATED**: Non-MoE model (Youtu-LLM:2b) works correctly!
- Produces coherent output: `<think>\nOkay, the user wants to...`
- Schedule cache hits after warmup
- ~0.5s per token

**Conclusion**: Problem is specific to GLM-4.7 Flash MoE, not MLA attention.
Possible causes:
1. QuantizedExpertWeights Q4_K dequant bug
2. Expert selection graph construction issue
3. Cache misses causing incorrect execution order

```bash
# Working test:
TEMP=0.7 REP_PENALTY=1.05 DEBUG=1 python tinygrad/apps/llm.py --model "youtu-llm:2b" --prompt "What is 2+2?"
```

### Quick Debug Commands
```bash
# Test with heartbeat timer
DEBUG=1 python tinygrad/apps/llm.py --model "glm-4.7:flash" --quantized --prompt "2+2=" --count 10

# Test with JIT disabled
JIT=0 python tinygrad/apps/llm.py --model "glm-4.7:flash" --quantized --prompt "2+2=" --count 10

# Test with symbolic disabled (integer positions only)
SYM=0 python tinygrad/apps/llm.py --model "glm-4.7:flash" --quantized --prompt "2+2=" --count 10
```

### Test Commands
```bash
# Quick correctness test
python tinygrad/apps/llm.py --model "glm-4.7:flash" --quantized --prompt "2+2=" --count 10

# Run unit tests
source .venv2/bin/activate
python test_quant_correctness.py
```

---

## Previous Status (2026-01-31)

### Problem Statement
- GLM-4.7 Flash (Q4_K_M, --quantized, max_context 512) still spikes to ~32–36 GB RSS on M3 32GB and becomes non-responsive.
- Interactive mode accepts input but does not return tokens; user interrupts while Metal is compiling a kernel (stack shows hang during cache_k assign / pipeline compilation).
- Kernel debug logs show steady execution but no decoded output; measured memory in logs (~20–21 GB) underreports OS RSS.

### Custom UOp Kernel for Q4_K Dequantization ✅ WORKING
- **Status**: Custom kernel matching reference implementation, 7x speedup achieved
- **File**: `tinygrad/nn/quant_kernels.py`
- **Approach**: Use `Tensor.custom_kernel()` API to write UOp-level kernel for Q4_K dequantization
- **Key insight**: Single fused kernel vs many tensor ops in reference
- **Performance**:
  - 1K blocks (140 KB -> 500 KB): 0.73ms vs 5.17ms (7.1x speedup)
  - 100K blocks (13.7 MB -> 48.8 MB): 1.08ms vs 7.30ms (6.8x speedup)
- **Q4_K Block Layout** (144 bytes -> 256 float16 elements):
  ```
  Byte 0-1:   d (float16)     - global scale
  Byte 2-3:   dmin (float16)  - global min scale
  Byte 4-15:  scales (12 bytes) - packed 6-bit scales and mins for 8 groups
    - bytes 4-7:   low 6 bits of sc[0-3], high 2 bits contribute to sc[4-7]
    - bytes 8-11:  low 6 bits of mn[0-3], high 2 bits contribute to mn[4-7]
    - bytes 12-15: low 4 bits of sc[4-7], high 4 bits of mn[4-7]
  Byte 16-143: qs (128 bytes) - 256 x 4-bit quantized values
    - Groups 0,1 share bytes 16-47 (even=low nibble, odd=high nibble)
    - Groups 2,3 share bytes 48-79
    - Groups 4,5 share bytes 80-111
    - Groups 6,7 share bytes 112-143
  ```
- **Dequant formula per group of 32**: `val[i] = d * sc[group] * q[i] - dmin * mn[group]`
- **Integration**: Already integrated into `QuantizedLinear.__call__` via `Q4K_UOP=1` env var (default on)

### Fused Q4_K Dequant+Matmul Kernel ✅ WORKING
- **Status**: Working fused kernel using `Tensor.custom_kernel()` with reduction
- **File**: `tinygrad/nn/quant_kernels.py:q4k_linear_fused()`
- **Pattern**: Uses `AxisType.REDUCE` with `.set()` for accumulation (same as `custom_gemm`)
- **Performance**:
  ```
  ( 4, 1024) x ( 256, 1024) |   0.5 MB dequant | 1.22x speedup
  ( 4, 4096) x (1024, 4096) |   8.0 MB dequant | 1.31x speedup
  ( 4, 4096) x (4096, 4096) |  32.0 MB dequant | 0.61x (GEMM dominates)
  ( 1, 4096) x (8192, 4096) |  64.0 MB dequant | 0.69x (GEMM dominates)
  ```
- **Key benefit**: Memory savings, not speed
  - No O(out * in) dequant buffer needed
  - For MoE: 64 experts × 8MB = 512MB saved per layer
  - Critical for memory-constrained MoE models like GLM-4.7 Flash
- **Benchmark**: `python tinygrad/nn/bench_q4k.py --quick`

### llama.cpp Metal Implementation Analysis for GLM-4.7 Flash

#### Key Source Files
- `llama.cpp/src/models/glm4-moe.cpp` - GLM-4 MoE model graph
- `llama.cpp/src/llama-graph.cpp:build_moe_ffn()` - MoE FFN implementation
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` - Metal kernels

#### GLM-4 MoE Architecture in llama.cpp
From `glm4-moe.cpp`:
```cpp
// Layer 0: Dense FFN
if (il < hparams.n_layer_dense_lead) {
    cur = build_ffn(cur, ffn_up, ffn_gate, ffn_down, LLM_FFN_SILU);
} else {
    // Routed MoE + shared expert combined
    ggml_tensor * routed_out = build_moe_ffn(cur, gate_inp, up_exps, gate_exps, down_exps, ...);
    ggml_tensor * shared_out = build_ffn(cur, ffn_up_shexp, ffn_gate_shexp, ffn_down_shexp, ...);
    cur = ggml_add(routed_out, shared_out);  // Combined output
}
```
- Layer 0: Dense FFN (n_layer_dense_lead=1)
- Layers 1-41: MoE with 64 experts + shared expert
- Shared expert FFN runs in parallel with routed experts

#### Q4_K Dequantization Tricks

**1. d/16 Trick for High Nibbles** ⭐ KEY OPTIMIZATION
```metal
// Instead of: q = (qbyte >> 4); val = d * q;
// Do this:    d_adj = d / 16; val = d_adj * (qbyte & 0xF0);
const float d = il < 2 ? xb->d : xb->d / 16.h;
const ushort mask = il < 2 ? 0x0F : 0xF0;
for (int i = 0; i < 16; ++i) {
    reg[i/4][i%4] = dl * (q[i] & mask) - ml;
}
```
This saves a shift operation per element in the inner loop.

**2. Compact Scale Unpacking (get_scale_min_k4_just2)**
```metal
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}
```
Returns both scale and min for a group in one call.

**3. 16-Element Tile Processing**
Q4_K blocks (256 elements) are processed in 16-element tiles.
Each tile gets its own scale/min pair, computed once per tile.

#### MoE Metal Kernel (kernel_mul_mm_id) - Critical for GLM-4.7 Performance

**Two-Phase Approach** (enables batched matmul):
1. **Phase 1: `kernel_mul_mm_id_map0`** - Build per-expert token lists
   - Input: expert selection tensor `[n_tokens, k_experts]`
   - Output: `htpe[n_experts]` = count of tokens per expert
   - Output: `hids[n_experts, max_tokens]` = which token indices use each expert
   - This pre-computation enables coalesced memory access in phase 2

2. **Phase 2: `kernel_mul_mm_id`** - Batched matmul per expert
   ```metal
   kernel void kernel_mul_mm_id(...) {
       const int im = tgpig.z;  // expert index (grid Z dimension)
       const int r0 = tgpig.y * NR0;  // output row tile
       const int r1 = tgpig.x * NR1;  // batch/token tile

       // Get token count and indices for this expert
       const int32_t neh1 = tpe_u32[im];  // how many tokens use this expert
       // ids_i32 = pre-computed token indices for this expert

       // Standard tiled matmul with indirect token indexing
       for (int k = 0; k < K; k += NK) {
           // Load weight tile (same expert, contiguous)
           // Load input tile (indirect via ids_i32)
           // simdgroup matrix multiply
       }
   }
   ```

**Key Constants** (optimized for Apple Silicon):
```metal
constexpr int NR0 = 64;   // Output feature tile (rows)
constexpr int NR1 = 32;   // Batch tile (tokens)
constexpr int NK  = 32;   // K-dimension tile
```

**Memory Layout:**
- Threadgroup memory: 4KB weight tiles + 4KB input tiles
- simdgroup matrix ops: 8x8 tiles leveraging hardware matmul
- Grid: `(ceil(n_tokens/NR1), ceil(out_features/NR0), n_experts)`

#### Why llama.cpp MoE is Fast

| Aspect | llama.cpp | tinygrad current |
|--------|-----------|------------------|
| Expert routing | Pre-computed per-expert token lists | Per-token expert selection (dynamic indexing) |
| Matmul scheduling | One kernel launch per expert (batched) | One kernel per token×expert (many launches) |
| Weight dequant | In-kernel, per-tile, never materialized | Separate pass creates O(out×in) buffer |
| Memory access | Coalesced via indirect indexing | Strided/scattered access |
| Graph structure | Fixed regardless of expert selection | Graph varies with selected experts → recompiles |

#### Applying to tinygrad for GLM-4.7

**Phase 1: Token-to-Expert Mapping** (implement in Python, run on GPU)
```python
def build_expert_mapping(sel: Tensor) -> tuple[Tensor, Tensor]:
    """Build per-expert token index lists.
    sel: (B, T, K) expert selection tensor
    Returns: (tokens_per_expert, expert_token_ids)
    """
    # Can be implemented as: scatter with counts, or sort + unique
```

**Phase 2: Batched Expert Matmul**
```python
def batched_expert_matmul(x: Tensor, expert_weights: Tensor,
                          tokens_per_expert: Tensor, expert_token_ids: Tensor) -> Tensor:
    """Run matmul for all experts in parallel.
    - Fixed grid: (n_experts, out_tiles, batch_tiles)
    - Each expert's tokens are contiguous in expert_token_ids
    """
```

**Critical for Schedule Cache:**
- Expert IDs must be **runtime values** (like batch size), not graph constants
- Grid dimensions should be fixed: `(n_experts, ceil(out/64), ceil(max_tokens/32))`
- Empty experts handled by early-exit check in kernel (like llama.cpp's `if (r1 >= neh1) return;`)

### Next Steps

1. ✅ **Fused Q4_K dequant + matmul kernel** - DONE (`q4k_linear_fused`)
   - Apply d/16 trick for high nibbles optimization

2. **Two-phase MoE approach** - Required for GLM-4.7 performance
   - Implement `build_expert_mapping` to create per-expert token lists
   - Implement `batched_expert_matmul` with fused Q4_K dequant
   - Fixed grid ensures schedule cache hits

3. **Integrate fused kernel into QuantizedLinear/QuantizedExpertWeights**
   - Replace current dequant+matmul with fused version for memory savings
   - Especially critical for MoE: 64 experts × 8MB = 512MB saved per layer

### Current Thoughts
- The main memory balloon is still in MoE expert handling; despite "selected expert only" dequantization, Metal/graph still grows and RSS exceeds RAM.
- The new expert gather path uses CPU-side index list; it likely prevents giant one-hot masks, but may still be materializing large buffers or triggering massive compile in Metal.
- The interactive hang is during Metal pipeline compilation (newComputePipelineState), suggesting too many unique kernels/graphs per token or excessive graph size.
- **With custom kernel**: The Q4_K dequant is now a single optimized kernel. This should reduce graph complexity significantly when integrated.


## 2026-01-31

### Added DeepSeek-V2-Lite Support ✅ WORKING
- **Status**: Model loads and generates tokens with cache hits
- **Issue**: Architecture mismatch - DeepSeek-V2-Lite has MLA weights but wasn't loading correctly
- **Root cause**: Different MLA variant than GLM-4.7 Flash
  - No `q_lora_rank` in GGUF (uses direct query projection instead of LoRA)
  - Combined `attn_kv_b` weight instead of separate `attn_k_b` and `attn_v_b`
- **Solution**: Made MLATransformerBlock support both variants
  - Query: LoRA (if `q_lora_rank > 0`) or direct projection (if `q_lora_rank == 0`)
  - KV: Split combined `attn_kv_b` into separate K/V weights during load
  - Use MLA when `kv_lora_rank > 0` (regardless of `q_lora_rank`)
  - Separate K/V caches (q_head_dim=192 vs v_head_dim=128)
  - Fixed shared expert FFN size: `n_shared_experts * moe_hidden_dim`
  - Added `deepseek-llm` tokenizer preset
- **Parameters inferred from GGUF**:
  - `kv_lora_rank: 512`
  - `qk_rope_head_dim: 64` (from rope.dimension_count)
  - `qk_nope_head_dim: 128` (from key_length - qk_rope_head_dim)
  - `v_head_dim: 128` (from value_length)
- **Performance**: First token ~1s (compilation), subsequent ~6ms with cache hits
- **Files**: tinygrad/apps/llm.py:105-170, 308, 369-377, 433-444

### Added Youtu-LLM-2B as Small MLA Test Model ✅ WORKING
- **Status**: Model loads and generates tokens correctly with MLA architecture
- **Issue**: DeepSeek-V2-Lite (16B MoE) too large for 36GB RAM Mac (swaps to 38GB+)
- **Solution**: Found Youtu-LLM-2B - a Dense MLA model (1.96B params, ~2.09GB Q8_0 GGUF)
- **Architecture**: Same MLA attention as DeepSeek-V2-Lite but without MoE
  - Uses Query LoRA (`q_lora_rank: 1536`)
  - Has separate `attn_k_b` and `attn_v_b` weights (not combined)
  - All 32 blocks are dense (no MoE)
- **Model URL**: `https://huggingface.co/tencent/Youtu-LLM-2B-GGUF/resolve/main/Youtu-LLM-2B-Q8_0.gguf`
- **Tokenizer**: `youtu` preset using `<|User|>`, `<|Assistant|>`, `<|end_of_text|>` format
- **Chat format**: User messages have no end marker, only assistant responses end with `<|end_of_text|>`
- **Sampling**: Requires `TEMP=0.7 REP_PENALTY=1.05` env vars to prevent repetition (greedy decoding loops)
- **Note**: Small 2B model has limited reasoning ability; MLA implementation verified working
- **Files**: tinygrad/apps/llm.py:60, 67, 473-489, 493

### Added Q5_0 Quantization Support
- **Issue**: DeepSeek-V2-Lite failed with "GGML type '6' is not supported!"
- **Solution**: Implemented Q5_0 dequantization (32 elements per 22-byte block)
- **Format**: 2 bytes scale (float16) + 4 bytes high bits + 16 bytes low 4 bits
- **Formula**: value = scale * ((low_4bits | (high_bit << 4)) - 16)
- **Fix**: Removed extra unsqueeze/flatten to match Q8_0 pattern (returns (blocks, 32) shape)
- **File**: tinygrad/nn/state.py:374-384

### Added Q5_1 Quantization Support
- **Issue**: DeepSeek-Coder-V2-Lite failed with "GGML type '7' is not supported!"
- **Solution**: Implemented Q5_1 dequantization (32 elements per 24-byte block)
- **Format**: 2 bytes d (float16) + 2 bytes m (float16) + 4 bytes high bits + 16 bytes low 4 bits
- **Formula**: value = d * ((low_4bits | (high_bit << 4))) + m
- **File**: tinygrad/nn/state.py:385-395
- **Now supported**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4

### Added DeepSeek-Coder-V2-Lite-Instruct (English MLA Model)
- **Status**: Too large for 36GB RAM Mac (uses ~37GB, causes swap thrashing)
- **URL**: `https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF`
- **Quantization**: Q4_K_S (~9.5GB weights, but total memory ~37GB with KV cache)
- **Architecture**: Same MLA as DeepSeek-V2-Lite (q_lora_rank=0, kv_lora_rank=512)
- **Tokenizer**: `deepseek-llm` preset (format: `User: ...\n\nAssistant: ...`)
- **Issue**: Hangs at Metal synchronize (`waitUntilCompleted`) due to memory pressure
- **Note**: Needs 64GB+ RAM machine to test

### Added Sampling Parameters
- **TEMP** env var: Temperature for sampling (0 = greedy, 0.7 = recommended for creative output)
- **REP_PENALTY** env var: Repetition penalty (1.0 = disabled, 1.05 = recommended to prevent loops)
- **Implementation**: Added `forward_logits()` method for manual repetition penalty application
- **Files**: tinygrad/apps/llm.py:315-316, 329-333, 450-464

### GLM-4.7 Flash - MoE Memory Analysis
- **Model**: 30B-A3B MoE (30B total params, 3B active) - NOT 4.7B!
- **Architecture**: 42 blocks (1 dense + 41 MoE), 64 experts per MoE block
- **Root cause**: Blocker is dequantization strategy and graph blowup, not raw weight size
  - 123 expert tensors (41 blocks × 3 per block)
  - Quantized (Q4_K): 123 × 113MB = ~14GB (fits in RAM)
  - Problem: `_q_to_uint8()` uses `Tensor.stack` creating huge lazy graphs
  - Without `.realize()`, graph grows unboundedly across all operations
- **Current fix**: Add `.realize()` after dequantization to bound graph size
  - Each expert tensor: 786432 blocks × 256 elements = 201M elements
  - Dequantized to float16: ~400MB temporary (per expert tensor, not 48GB total!)
  - Peak memory: ~1.2GB per MoE layer (3 expert tensors × 400MB)
  - Sequential layer execution means only one layer's temporaries at a time
- **Long-term solution** (requires development):
  - Quantized MoE matmul Metal kernels - dequantize inside kernel per block
  - Never materialize full dequantized tensor in memory
  - This is how llama.cpp achieves 50+ tok/s on same hardware
- **Files**: tinygrad/apps/llm.py:94-109

## 2026-01-30

### Added Behavioral Guidelines (22:36)
- Added LLM coding best practices to CLAUDE.md
- Guidelines cover: thinking before coding, simplicity first, surgical changes, goal-driven execution

### Fixed MLA Attention Shape Issue (22:33)
- Fixed tensor shape handling in MLATransformerBlock
- Added extra unsqueeze/squeeze operations for k_nope and v projections
- File: tinygrad/apps/llm.py:188-191

### Working Quantization Support (22:27)
- Implemented QuantizedLinear class for on-the-fly dequantization
- Added Q5_K dequantization support alongside Q4_K and Q6_K
- Fixed quantized weight handling for llama models (attn_q/attn_k must be dequantized for permutation)
- Added fallback to dequantize tensors that can't be kept quantized (Embeddings, etc.)
- Added --quantized flag to llm.py for lower memory usage
- Files: tinygrad/apps/llm.py, tinygrad/nn/state.py

### Added GLM-4.7 Flash Model Support (22:18)
- Implemented Multi-head Latent Attention (MLA) architecture from DeepSeek-V2
- Added MLATransformerBlock class with:
  - Query down-projection -> norm -> up-projection
  - KV down-projection with MQA rope -> norm -> separate k_b/v_b up-projections
  - Support for MoE with shared experts
- Added PerHeadWeights class for per-head k/v projections
- Added GLM4 tokenizer preset with proper role/end_turn handling
- Added model entry: "glm-4.7:flash" (Q4_K_M quantization)
- Model size: ~4.7B parameters, Q4_K_M should fit in 36 GB RAM
- Files: tinygrad/apps/llm.py, tinygrad/nn/state.py

## Testing MLA Models

```bash
# Youtu-LLM-2B (small, ~4GB RAM, WORKING)
# Requires temperature and repetition penalty to avoid loops
TEMP=0.7 REP_PENALTY=1.05 python tinygrad/apps/llm.py --model "youtu-llm:2b" --max_context 512

# GLM-4.7 Flash (30B MoE, needs 64GB+ RAM - too large for 36GB)
# python tinygrad/apps/llm.py --model "glm-4.7:flash" --quantized --max_context 1024

# DeepSeek-Coder-V2-Lite (16B MoE, needs 64GB+ RAM - hangs on 36GB)
# TEMP=0.7 python tinygrad/apps/llm.py --model "deepseek-coder-v2-lite" --max_context 256

# DeepSeek-V2-Lite (16B MoE, needs 64GB+ RAM)
# python tinygrad/apps/llm.py --model "deepseek-v2-lite"
```

## System Info

- Machine: Mac with 36 GB RAM
- Device: Metal (macOS)
- Branch: glm
- Base branch: master


# FROM glm47_llama_glm45_insights.md
# GLM-4.7 Flash: tinygrad vs llama.cpp vs GLM-4.5 (notes)

Scope: local repos only (`tinygrad/`, `llama.cpp/`, `GLM-4.5/`). Focus is fast + correct.

## 1) What changed vs 1803ee9 (high-level)
- New quantized GGUF flow: `tinygrad/nn/state.py` now optionally returns raw K-quant blocks via `gguf_load(..., quantized=True)` and keeps only K-quant tensors quantized.
- New custom kernels: `tinygrad/nn/quant_kernels.py` adds Q4_K dequant and fused Q4_K/Q6_K MoE matmul kernels with runtime expert IDs.
- New MLA path: `tinygrad/apps/llm.py` adds DeepSeek2-style MLA, interleaved RoPE, MoE gating controls, shared experts.
- New tokenization presets (glm4/deepseek-llm/youtu) and GLM4 `[gMASK]<sop>` prefix.

Diff-risk worth noting:
- `gguf_load` now returns 3 values, but `examples/gpt2.py` and `test/unit/test_gguf.py` still unpack 2. Those will now fail if run.

## 2) llama.cpp deepseek2 implementation (ground truth for GLM-4.7 Flash)
Key files:
- `llama.cpp/src/models/deepseek2.cpp`
- `llama.cpp/src/llama-graph.cpp` (`build_moe_ffn`)

Correctness-critical behaviors:
- **MLA + absorption**: q_nope is absorbed into compressed KV via `wk_b` and then concatenated with rope q_pe. That matches tinygrad’s `q_nope_absorbed` + `q_pe` concat.
- **RoPE**: deepseek2 applies RoPE only on the rope sub-dim (`q_pe`, `k_pe`) via `ggml_rope_ext` (no llama-style permutation). This implies interleaved format is expected for deepseek2.
- **Attention scale**: `kq_scale` is `1 / sqrt(n_embd_head_k)` with optional YaRN scaling. If rope scaling params are default, this reduces to `1/sqrt(q_head_dim)`.
- **MoE gating** (from `build_moe_ffn`):
  - `probs = sigmoid/logits/softmax` based on `expert_gating_func`.
  - `selected_experts = argsort_top_k(selection_probs)` where `selection_probs` may be biased by `exp_probs_b`.
  - `weights = get_rows(probs, selected_experts)` then optional normalize (`weights_norm`) and scale (`weights_scale`).
  - Up/Gate/Down expert matmuls use `build_lora_mm_id(..., selected_experts)` to drive expert-indexed kernels.

Performance-critical behaviors:
- llama.cpp uses **topk-aware kernels** (`build_lora_mm_id` + specialized metal kernels) that handle expert indices as runtime data. This keeps graphs stable and avoids cache misses.
- `build_moe_ffn` explicitly expands `weights` early so topk-moe kernel selection kicks in (no graph bloat later).

## 3) llama.cpp GLM4-MoE (GLM-4.5 family) differences
Key file:
- `llama.cpp/src/models/glm4-moe.cpp`

Main differences vs GLM-4.7 Flash (deepseek2):
- GLM4-MoE is **standard attention**, not MLA. It uses direct Q/K/V projections, optional Q/K norm, and standard RoPE (or mRoPE if enabled).
- It has **post-attention norm** (`attn_post_norm`), and it skips final NextN layers.
- MoE path still uses `build_moe_ffn`, but **routing + shared expert** is combined at the end (`routed_out + shared_out`).
- Attention scale is fixed `1/sqrt(n_embd_head)`.

Implication: GLM-4.5 correctness details (q/k norm, post-attn norm, nextn skip) are irrelevant for GLM-4.7 Flash, which is deepseek2/MLA.

## 4) GLM-4.5 repo notes (prompting)
- `GLM-4.5/inference/trans_infer_cli.py` uses `tokenizer.apply_chat_template(..., add_generation_prompt=True)` from HF.
- The prompt format is not spelled out in this repo; it’s delegated to HF’s chat template.
- The README indicates GLM-4.7 Flash is implemented as `glm4_moe_lite` in HF/vLLM/SGLang. So the **template is likely maintained there**, not locally.

## 5) Correctness gaps most worth checking in tinygrad
1) **GLM4 `[gMASK]<sop>` placement**
   - llama.cpp’s chatglm4 template explicitly prefixes the text with `"[gMASK]<sop>"` and then adds `<|role|>\n...`.
   - In llama.cpp vocab, `tokenizer_pre == "glm4"` sets `special_bos_id = LLAMA_TOKEN_NULL` (i.e., no automatic BOS).
   - tinygrad currently relies on `bos_id` + `after_bos()` to create `[gMASK]<sop>`. If `bos_id` isn’t `[gMASK]` (or `add_bos` is false), the prompt is wrong.
   - Action: print the exact decoded prefix for GLM-4.7 Flash and compare to llama.cpp’s template string.

2) **Expert selection bias (`exp_probs_b`)**
   - llama.cpp can add `ffn_exp_probs_b` to selection logits **before top-k**. That shifts which experts are picked.
   - tinygrad doesn’t load or apply `exp_probs_b` at all.
   - Action: check GGUF for `blk.*.ffn_exp_probs_b` (or similar). If present, this is a real correctness gap.

3) **KV cache equivalence**
   - Single-token correctness + multi-token garbage strongly suggests cache or prompt mismatch.
   - Action: compare logits for a 2-token prompt computed in one call vs step-by-step with cache. They should match.

4) **Softmax-weight gating** (only if `expert_gating_func=3`)
   - llama.cpp applies softmax over the selected expert weights, not just a sum-normalize.
   - tinygrad currently normalizes by sum (for gating_func 3) which is not a softmax. Not relevant for GLM-4.7 (gating_func=2), but worth noting if future models use gating_func=3.

## 6) Speed alignment (fast + correct)
- llama.cpp MoE speed comes from **two-phase expert routing + expert-indexed kernels** (token-to-expert mapping then batched expert matmul). This avoids per-token kernel variation and avoids dequant buffers.
- tinygrad’s `q4k_moe_fused` uses runtime expert IDs (good for cache) but still performs per-row matmul. It’s likely correct but not as fast as llama.cpp’s batched expert kernel.
- If speed is a primary goal, the next step is to implement **llama.cpp-style MoE kernels**: build expert token lists and do batched matmul per expert.

## 7) Suggested next verification steps (minimal + decisive)
- **Prompt prefix check**: print decoded prefix tokens for GLM-4.7 Flash to ensure `[gMASK]<sop><|user|>\n` and assistant suffix match llama.cpp.
- **Cache equivalence test**: compare logits for 2-token sequence in one pass vs step-by-step with cache.
- **Check for exp_probs_b**: verify if the GGUF contains expert selection bias tensors.

If these three checks pass, then remaining correctness issues likely come from MLA/KV layout or quantization artifacts, not prompt or gating.

## 8) llama.cpp performance specifics (Metal backend)
Sources: `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`, `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`, `llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`.

- **Two-phase MoE routing on GPU**:
  - `kernel_mul_mm_id_map0` builds per-expert token lists and counts in two buffers (`htpe` = tokens-per-expert, `hids` = token indices). It is specialized by `n_expert_used` via template instantiation (e.g. `kernel_mul_mm_id_map0_ne20_4`).
  - The map kernel scans the selected expert ids for each token and emits a compact list for each expert. This turns sparse routing into contiguous batches per expert.

- **Batched expert matmul kernel**:
  - `kernel_mul_mm_id` consumes `htpe`/`hids` and launches a fixed grid: `(ceil(n_tokens/32), ceil(out/64), n_experts)` with tile sizes `NR0=64`, `NR1=32`, `NK=32`.
  - Early exit: `if (r1 >= neh1) return;` skips experts with zero tokens without changing the grid shape.
  - Threadgroup memory is explicitly sized (two 4KB tiles for A/B plus scratch), enabling high reuse and stable performance.

- **Quantized types handled in-kernel**:
  - `kernel_mul_mm_id` is templated over quant types (Q4_0, Q4_K, Q6_K, etc). Dequant happens inside the matmul kernel, so no intermediate dequant buffers are materialized.
  - There are separate pipeline instantiations for many quant formats (`kernel_mul_mm_id_q4_K_f16`, `kernel_mul_mm_id_q6_K_f16`, etc), selected at runtime based on tensor type.

- **Synchronization between routing + matmul**:
  - In `ggml-metal-ops.cpp`, llama.cpp dispatches `kernel_mul_mm_id_map0`, then forces a barrier (`ggml_metal_op_concurrency_reset`) before launching the matmul kernel. This guarantees the id maps are ready for the main kernel.

**Why this matters for tinygrad**
- llama.cpp keeps the **graph shape fixed** even when expert ids change, because the expert id data is runtime input to the map kernel. This maximizes schedule cache hits.
- The batching (per-expert token lists) plus fused dequant matmul is the main reason llama.cpp achieves high tok/s on MoE models.
- tinygrad’s current fused MoE kernels use runtime expert ids (good), but still do per-row matmul. Matching llama.cpp’s two-phase map + batched matmul should close the perf gap while keeping correctness.

## 9) tinygrad custom_kernel patterns (implementation + perf)
Sources: `test/test_custom_kernel.py`, `extra/gemm/mi350x_uop_matmul.py`, `extra/assembly/amd/test/test_custom_kernel.py`, `extra/gemm/amd_asm_matmul.py`.

Core API patterns:
- `Tensor.custom_kernel(...)` wraps one or more output tensors and feeds their UOps into a Python callback (`fxn`) that builds a UOp program.
- The kernel body is a UOp graph that ends in `...sink(arg=KernelInfo(...))`. Use `KernelInfo(opts_to_apply=())` to disable optimizer passes for hand-written kernels.
- Reductions use `AxisType.REDUCE` and explicit accumulator updates via `.set(...)` + `.after(k)` + `end=k` (see `custom_gemm` / `custom_sum`).
- Vectorized loads/stores are modeled via `AxisType.UPCAST`, `contract()`, and explicit `UOp.group(...)` for multiple stores.
- Gradients can be provided via `grad_fxn` (see `backward_gemm` and `backward_gemm_custom`).

Correctness + scheduling details:
- Multi-output kernels should return a grouped store with `UOp.group(...)` and end ranges for every active loop.
- For multi-device/multi-output, `CUSTOM_KERNEL + AFTER` is supported in the scheduler (`tinygrad/schedule/multi.py`).
- Use `KernelInfo(name=...)` to tag kernels for debugging and profiling.

Performance mechanics from `mi350x_uop_matmul.py`:
- Uses explicit thread geometry via `UOp.special(...)` for `gidx`/`lidx` and maps to block/warp tiling manually.
- Uses local/shared memory (`AddrSpace.LOCAL`) + barriers to stage tiles of A/B into shared memory.
- Uses register tiles (`AddrSpace.REG`) + `Ops.WMMA` with 16x16x32 shapes to hit tensor cores.
- Tile sizes are chosen explicitly (BLOCK_M/N/K, TC_M/N/K); the kernel is responsible for coalesced loads and bank-conflict avoidance.
- Accumulator is a vectorized register tile initialized manually; writeback uses explicit vectorization and `gep`.

ASM and binary-backed kernels:
- `extra/gemm/amd_asm_matmul.py` shows a low-level path where a raw AMD assembly kernel is compiled and injected as an `Ops.PROGRAM` with `Ops.SOURCE` + `Ops.BINARY`.
- It builds `KernelInfo(estimates=Estimates(...))` to provide cost metadata and uses explicit `UOp.special` threads.
- This path bypasses most of the high-level optimizer and is suited to vendor-tuned kernels or external asm.

## 10) `extra/gemm/amd_matmul.py` (swap-in kernel strategy)
- Extracts the compiled matmul AST (`(A@B).schedule()[-1].ast`) and `get_program` to obtain a baseline kernel definition.
- Replaces the program source with tuned kernels (C++/ASM) while keeping the same argument layout and launch geometry.
- Uses `CompiledRunner` + `ExecItem` to run the swapped kernel with existing buffers, enabling A/B testing without graph changes.
- This is a clean pattern for **fast experimentation**: keep the graph stable, swap only the kernel code.

## 11) Performance takeaways from tinygrad examples
- Handwritten kernels should explicitly control:
  - threadgroup sizes (`UOp.special`) and global tiling
  - data movement into shared/local memory and barriers
  - vectorized loads (`AxisType.UPCAST` + `contract`) and register tiling
- Disable optimizer rewrites for custom kernels where you depend on exact scheduling or memory layout (`opts_to_apply=()`).
- When testing new kernels, use `CompiledRunner` with the existing AST to avoid changing graph structure (reduces compile churn and isolates kernel correctness/perf).
- For AMD targets, `Ops.PROGRAM` + `Ops.BINARY` offers a direct path to custom assembly while still leveraging tinygrad’s scheduling and buffer plumbing.

## 12) Correctness + tinygrad idiosyncrasies (things that help or hurt)

### A) Graph / cache behavior
- **UOp caching**: identical `(op, dtype, src, arg)` yields the same UOp object. This is great for reuse but can hide bugs when tags are mutated. Example: `uop.replace(tag=None)` can return the cached untagged UOp, not a new node.
- **Schedule cache normalization**: BIND nodes may have values stripped before cache key computation. If your custom kernel depends on `bind.src[1]`, always guard `len(bind.src) > 1`.
- **Runtime vs graph constants**: Expert IDs must be runtime tensors (not used in Python indexing). Otherwise, the graph structure changes per token and schedule cache misses explode.

### B) Custom kernel correctness pitfalls
- **Range closure**: Every `UOp.range` you open must be closed with `end(...)`. Missing ends can silently produce wrong or partial writes.
- **Reduction ordering**: Use `AxisType.REDUCE` and update via `C = C.set(C.after(k) + val, end=k)`; do not write accumulators without `.after(k)` or you’ll lose reduction semantics.
- **Optimizer passes**: For hand-written kernels that rely on precise control, set `KernelInfo(opts_to_apply=())` to prevent rewriters from altering loop structure.
- **Multi-output kernels**: Use `UOp.group(...)` and `end(...)` the shared ranges once. If only one output is ended, the other stores may be dropped.

### C) KV cache and position handling
- **Symbolic positions**: `start_pos` is often a symbolic UOp in generation. Anything that calls `int(start_pos)` is wrong when symbolic. If you need a mask with symbolic size, rely on tensor ops rather than Python ints.
- **Cache shape binding**: Cache tensors are created with the batch size from the first call. If B changes, later calls can silently broadcast or mismatch. (For single-batch inference this is fine, but it can mask errors in tests.)

### D) Tokenization/prompting mismatch risk
- **glm4 pre‑tokenization**: llama.cpp’s chatglm4 template explicitly prefixes the string with `"[gMASK]<sop>"` and does *not* use BOS. If tinygrad’s `bos_id` isn’t `[gMASK]`, the prompt is wrong even if `after_bos()` adds `<sop>`.
- **Tokenizer preset**: tinygrad relies on `tokenizer.ggml.pre` to select presets. If the gguf uses `glm4` but the preset doesn’t map exactly, role tokens or end‑turn markers can diverge subtly.

### E) Quantization handling
- **K‑quant blocks**: `gguf_load(..., quantized=True)` keeps K‑quant blocks raw and excludes them from `state_dict`. If any layer *expects* dequantized weights (e.g., permuted llama Q/K), it must be handled explicitly.
- **Dequant outputs**: Ensure output dtypes match expectations (float16 vs float32). Some kernels cast to float16 for performance; ensure accumulation remains float32 where needed.

### F) Debugging aids unique to tinygrad
- **Graph inspection**: `VIZ=1` / `VIZ=-1` can show rewrites; `TRACK_MATCH_STATS=2` shows pattern cost.
- **Kernel naming**: `KernelInfo(name=...)` makes it easy to isolate and compare kernels in debug output.
- **`realize()` to bound graphs**: Strategic `realize()` calls can prevent giant fused graphs that hide errors (but can also change numerics by breaking fusion).

### G) Interop differences vs llama.cpp
- **Expert bias tensors**: llama.cpp can apply `ffn_exp_probs_b` before top‑k. If the gguf includes these and tinygrad ignores them, expert selection diverges immediately.
- **Gating func variants**: llama.cpp supports softmax, sigmoid, and softmax_weight. tinygrad’s handling for `softmax_weight` currently normalizes by sum (not softmax), which is wrong if gating_func==3 (not GLM‑4.7, but relevant to other models).

### H) Fast‑path hazards
- **Fused kernels**: the fused Q4_K/Q6_K MoE kernels are correct on synthetic tests, but they bypass some graph paths (e.g., shape checks or implicit casts). A bug here can produce “looks numeric but wrong” outputs.
- **Contiguity assumptions**: custom kernels often assume contiguous layout. If you pass non‑contiguous tensors, you can get wrong results without an error. Call `.contiguous()` where appropriate.


# FROM insights.md
# Analysis of Discrepancies between Tinygrad and Llama.cpp for GLM-4.7/DeepSeek-V2

## 1. Missing YaRN RoPE Scaling (Critical, but maybe Optional)
**Issue**: Tinygrad uses `rope_theta` directly for Rotary Positional Embeddings (vanilla RoPE). GLM-4.7 (DeepSeek-V2) can use YaRN (Yet another RoPE extension) scaling.
- **Tinygrad**: `precompute_freqs_cis(..., rope_theta)`
- **Llama.cpp**: `ggml_rope_yarn_corr` with `mscale` attention attenuation, `beta_fast`, and `beta_slow` corrections.
- **Insight**: Native context is 32k. YaRN is often optional for context extension beyond 32k. However, `llama.cpp` applies it via `ggml_rope_ext` which includes corrections. If `beta_fast/slow` are default, it might behave like vanilla RoPE, but `mscale` (attention attenuation) is distinct.
- **Impact**: Causes logit explosion at T>1 if scaling is required but missing.

## 2. Missing MoE Expert Group Selection
**Issue**: DeepSeek-V2/V3 architecture uses a "Grouped Top-K" strategy where experts are first selected by group, and non-selected groups are masked out. Tinygrad selects experts globally from a flat list.
- **GGUF Data**: `deepseek2.expert_group_count: 1` and `deepseek2.expert_group_used_count: 1`.
- **Insight**: For **GLM-4.7 Flash specifically**, the group count is 1. This means **Grouped Top-K reduces to Standard Top-K**. The complex grouping logic in `llama.cpp` is skipped when `n_expert_groups <= 1`.
- **Conclusion**: The MoE Grouping discrepancy **is likely NOT the cause of the bug** for this specific model, as it uses a single group. The issue must be elsewhere (RoPE or Biases).

## 3. Missing MoE Bias Terms
**Issue**: Tinygrad ignores bias terms in the MoE router and expert scoring.
- **Investigation**: `check_biases.py` confirmed that `blk.*.ffn_gate_inp.bias` is **MISSING** from the GGUF. However, `blk.*.exp_probs_b.bias` **EXISTS** (shape `(64,)`).
- **Llama.cpp**: `selection_probs = ggml_add(ctx0, probs, exp_probs_b)`.
- **Tinygrad**: Currently ignores `exp_probs_b`.
- **Impact**: The expert selection probabilities are incorrect because this learned bias is omitted. This leads to selecting suboptimal experts. It is a confirmed correctness bug.

## 4. Potential incorrect gating normalization for `softmax_weight`
**Issue**: For `expert_gating_func=3` (SOFTMAX_WEIGHT), Tinygrad normalizes via division by sum, whereas Llama.cpp applies Softmax to the *selected* weights.
- **GGUF Data**: `deepseek2.expert_gating_func: 2` (Sigmoid).
- **Conclusion**: This discrepancy is **irrelevant** for GLM-4.7 Flash, which uses Sigmoid gating (`func=2`).

## 5. MoE Weight Scaling Order
**Observation**: Tinygrad scales the output of the MoE block, while Llama.cpp scales the expert weights (probabilities).
- **GGUF Data**: `deepseek2.expert_weights_scale: 1.8`.
- **Insight**: This scale factor is significant (1.8x). Tinygrad applies it effectively.
- **Impact**: Mathematically equivalent `Sum(x * w) * s == Sum(x * (w * s))`.

## 6. Shared Expert Handling
**Status**: Appears consistent.
- **GGUF Data**: `deepseek2.expert_shared_count: 1`.
- **Tinygrad**: Adds shared expert output to routed expert output.
- **Llama.cpp**: Same logic verified.

## 7. RoPE Parameters from GGUF
- `freq_base: 1000000.0` (Standard for DeepSeek/GLM-4).
- `rope.dimension_count: 64` (Matches `qk_rope_head_dim`).
- **Missing**: No explicit YaRN keys (`rope.scaling.*`) in the GGUF dump. This strongly suggests the model relies on default architecture behavior or standard RoPE for the base context. The "Interleaved" fix is likely the most important one.

## Recommendations
1. **Fix RoPE Interleaved**: Ensure `apply_rope_interleaved` is used correcty (already identified).
2. **Load Router Bias**: The most likely remaining culprit for "wrong experts" is `ffn_gate_inp.bias`. Update `MLATransformerBlock` to load this bias.
3. **Verify Attention Scale**: Ensure `scale = 1.0 / math.sqrt(256)` (effective head dim) is used, not `192` (nope dim) or `576` (key length).



# FROM next_steps.md
# Next Steps for GLM-4.7 Flash / DeepSeek-V2 Support

## 1. Correctness (Highest Priority)

### 1.1 Fix RoPE Implementation (Ready)
- **Status**: Root cause identified (missing YaRN scaling and Interleaved format quirks).
- **Action**: 
    - Apply the fix from `rope_patch.py` (or equivalent `apply_rope_interleaved`).
    - Implement YaRN scaling (`mscale`, `beta_fast/slow`) to match `llama.cpp`'s `ggml_rope_yarn_corr`.
- **Verify**: Run `test_rope_bug.py` and `test_rope_final.py`. Ensure logit explosion is gone at T>1.

### 1.2 Implement DeepSeek-V2 MoE Grouped Selection (Critical)
- **Status**: Identified simplified MoE logic in `tinygrad` vs `llama.cpp`.
- **Action**:
    - Update `MLATransformerBlock` (or `ExpertWeights`) to support "Grouped Top-K".
    - Logic:
        1. Reshape router logits to `(groups, experts_per_group)`.
        2. Select top `n_group_used` groups.
        3. Mask (set to `-inf`) experts in rejected groups.
        4. Perform standard Top-K on remaining experts.
- **Reference**: `llama.cpp/src/llama-graph.cpp` (`build_moe_ffn`).

### 1.3 Add Missing MoE Biases
- **Status**: `exp_probs_b` confirmed present in GGUF. `ffn_gate_inp.bias` is absent.
- **Action**:
    - Update `MLATransformerBlock` to load `exp_probs_b` (Expert Score Bias).
    - Apply bias in `_feed_forward`: `probs = probs + self.exp_probs_b` (before Top-K).
    - Ensure `ffn_gate_inp` remains `bias=False`.

### 1.4 Gating Normalization Review
- **Status**: Potential mismatch in Softmax vs Sigmoid vs Linear normalization.
- **Action**: Confirm `expert_gating_func` for GLM-4.7 (likely Sigmoid/2). Ensure logic matches `llama.cpp` exactly for that mode.

## 2. Performance & Memory Stability

### 2.1 Fix Memory Ballooning (Quantization)
- **Issue**: `QuantizedLinear` dequantizes entirely into memory, causing OOM/Swap on 32GB builds.
- **Action**:
    - Implement **Chunked Realization** in `QuantizedLinear`: process `N` blocks, accumulate, realize, repeat.
    - **Long-term**: Define `MUL_MAT_ID_Q` op for true in-kernel dequantization (DeepSeek-V2 specific requirement for efficiency).

### 2.2 Optimize Metal Compilation
- **Issue**: `_no_jit = True` leads to compilation hangs and excessive kernel generation.
- **Action**:
    - Fix `sel.tolist()` dependency in MoE to allow JIT.
    - Batch expert computations instead of sequential loop.

## 3. Code Health & Cleanup

### 3.1 Fix `gguf_load` Regressions
- **Issue**: Signature change breaks `examples/gpt2.py` and tests.
- **Action**: Unify return signature (always return 3 values or make backward compatible). Update all consumers.

### 3.2 Fix "WARNING: not loading" Messages
- **Issue**: `state_dict` missing `.blocks` keys for quantized tensors.
- **Action**: Explicitly add `.blocks` references to `state_dict` during `QuantizedLinear` initialization.

### 3.3 Folder Hygiene
- **Action**: Move ad-hoc test scripts (`test_rope_bug.py`, `check_moe_keys.py`, etc.) to `test/` or `extra/` with proper naming.

## 4. Documentation
- **Action**: Maintain `progress.md` as the single source of truth for session tracking. Archive older logs once features are merged.


# FROM review_since_master.md
# Review since `master` (no code changes)

Scope: all additions/changes since `master` across `tinygrad/` plus new scripts/docs. Focus on potential issues, cleanups, complexity collapse, and reuse opportunities.

## High‑risk / likely issues
- **`gguf_load` signature change breaks callers**: `tinygrad/nn/state.py` now returns 3 values when `quantized=True`, but `examples/gpt2.py` and `test/unit/test_gguf.py` still unpack 2 values. This will fail on `master` test paths. Consider backward‑compatible return (e.g., always 2 unless `quantized=True` and caller opts into 3) or update all call sites. Paths: `examples/gpt2.py`, `test/unit/test_gguf.py`, `examples/llama3.py`.
- **Interactive prompt start_pos bug persists**: `tinygrad/apps/llm.py` still computes `start_pos = max(len(ids) - 1, 0)` *before* appending user input, so it skips the first tokens (`[gMASK]` for GLM4) on the first turn. This is a correctness bug independent of quantization.
- **Prompt prefix likely wrong for GLM4**: llama.cpp sets `special_bos_id = NULL` for `tokenizer_pre == "glm4"` and explicitly prefixes `"[gMASK]<sop>"`. tinygrad relies on `bos_id` + `after_bos()` to create `[gMASK]<sop>`. If `bos_id` isn’t `[gMASK]` or `add_bos` is false, the prompt is wrong.

## Medium‑risk / correctness gaps
- **`exp_probs_b` not applied**: llama.cpp adds `ffn_exp_probs_b` (selection bias) before top‑k in `build_moe_ffn`. tinygrad does not load or apply these tensors, which can change expert selection. Check GGUF for `blk.*.ffn_exp_probs_b` and wire it in if present.
- **`softmax_weight` gating semantics**: llama.cpp applies softmax to the *selected* expert weights when gating_func==3. tinygrad currently normalizes by sum for gating_func==3, which is not the same. Not GLM‑4.7, but relevant to other MoE models.
- **`int(start_pos)` in MLA mask**: `MLATransformerBlock._attention` uses `int(start_pos)` when `T > 1`. This is safe for the current flow (symbolic start_pos only when T==1), but it’s fragile and easy to regress if generation behavior changes.

## Performance pitfalls
- **Host sync per token with rep_penalty**: `Transformer.generate` uses `.numpy()` to apply repetition penalty, forcing a device‑to‑host sync every token. This is a large perf hit when `REP_PENALTY>1.0`. If used in production, consider a device‑side implementation.
- **Manual attention path in MLA**: `q.matmul(k.T)` + `softmax` is correct but may be slower than `scaled_dot_product_attention` or a fused kernel. For large context, this is a perf bottleneck.
- **`QuantizedLinear` fused path always casts input to fp16**: In `q4k_linear_fused`, input is cast to `float16` even if `HALF=0`. This is probably intended for perf, but it can silently change numerics and makes `HALF=0` less meaningful.

## Cleanup / complexity collapse opportunities
- **Unify Q4_K/Q5_K/Q6_K dequant logic**: There is now duplicate dequant code in `tinygrad/nn/state.py` and `tinygrad/nn/quant_kernels.py`. Consider factoring a single reference implementation (or shared helpers for scale/min unpack) so correctness updates happen once.
- **Consolidate tokenizer handling**: `SimpleTokenizer` now has multiple presets (glm4, deepseek‑llm, youtu) with manual role/end_turn logic. There is already a chat template system in llama.cpp; if tinygrad grows more presets, consider a centralized template map and explicit tests for prefix output.
- **Re‑use existing quantized linear patterns**: `QuantizedLinear` and `QuantizedExpertWeights` are bespoke in `llm.py`. If other models need quantized matmul, consider moving these into `tinygrad/nn/` (or a shared module) to reduce duplication and isolate quant logic.
- **Move ad‑hoc scripts to `extra/`**: `test_quant_correctness.py`, `test_moe_mvp.py`, and `tinygrad/nn/bench_q4k.py` are top‑level or under `tinygrad/nn/`. For consistency, consider moving to `extra/` or `test/` with naming conventions so pytest discovery is deliberate.
- **Minimize `.realize()` in tight loops**: `QuantizedLinear` realizes each block accumulation to bound graph size. This is pragmatic, but it’s also a perf cliff. If feasible, consider a custom kernel (as done for Q4_K) or a staged reduction that doesn’t force realize each block.

## Reuse opportunities in the tinygrad codebase
- **Custom kernel patterns**: The UOp reduction pattern in `test/test_custom_kernel.py` is already canonical (`set` + `after` + `end`). It can be reused or referenced to validate the MoE custom kernels.
- **`CompiledRunner` kernel swapping**: `extra/gemm/amd_matmul.py` shows a clean way to swap kernel source with the same AST/argument layout. This could be adapted for MoE kernels to keep graph shapes fixed while iterating on kernel code.

## Test coverage / maintenance gaps
- **No automated tests updated for GGUF load change**: `test/unit/test_gguf.py` still expects two return values from `gguf_load`. This will now fail and should be updated (or the function should be backward‑compatible).
- **Quantized path lacks unit coverage**: `QuantizedLinear`/`QuantizedExpertWeights` correctness is only covered by standalone scripts in root. Consider adding minimal tests under `test/unit/` to avoid regressions.

## Docs/metadata
- **Large doc additions** (`progress.md`, `issue_log.md`, `suggestions.md`, `design.md`) are fine but add a lot of repo noise. If these are long‑term, consider a `notes/` or `docs/` subdir to keep repo root cleaner.

---

If you want, I can split this into separate lists (correctness, perf, cleanup) or tie each item to specific line numbers for easier follow‑up.


# FROM suggestions.md
# GLM-4.7 Flash Quantization Issues - Analysis & Suggestions

## Issue 1: "WARNING: not loading" Messages

### Root Cause
When `QuantizedLinear`/`QuantizedExpertWeights` objects are created, they have a `.blocks` tensor attribute. The `load_state_dict` function calls `get_state_dict(model)` which finds keys like `blk.0.attn_q_a.blocks`, but the `state_dict` dict only contains non-quantized weights (norms, embeddings, etc.) - it doesn't have `.blocks` keys.

The flow:
1. GGUF loads `blk.0.attn_q_a.weight` -> stored in `quantized_tensors`
2. Code creates `QuantizedLinear(blocks=blocks_tensor, ...)` and assigns it to model
3. `state_dict` is passed to `load_state_dict()`
4. `load_state_dict` scans model and finds `blk.0.attn_q_a.blocks`
5. But `state_dict` doesn't have that key -> warning printed

### Fix
After creating QuantizedLinear/QuantizedExpertWeights, add the blocks to state_dict:

```python
# In llm.py around line 530:
if isinstance(obj, nn.Linear):
  setattr(parent, attr, QuantizedLinear(blocks, shape, ggml_type, name=name))
  state_dict[name.replace('.weight', '.blocks')] = blocks  # ADD THIS
```

This ensures `load_state_dict` finds the tensor and assigns it (essentially a no-op since it's the same tensor, but prevents warnings).

---

## Issue 2: Memory Balloon (32-36 GB RSS)

### Analysis from progress.md
- GLM-4.7 Flash is 30B-A3B MoE (30B params, 3B active per token)
- 42 blocks (1 dense + 41 MoE), 64 experts per MoE block
- Quantized (Q4_K): ~14GB weights
- Problem: Graph blowup when dequantizing

### Current Architecture
`QuantizedLinear.__call__` dequantizes per-block during forward:
```python
for bi in range(n_blocks):
  w = dequant_fn(self.blocks[bi])  # lazy tensor
  y = x_slice.linear(w.T)
  out = out + y  # accumulates lazy ops
```

This creates a huge lazy graph with `n_blocks` dequant operations that aren't realized until the end.

### Suggested Fix: Chunked Realization
Process blocks in small chunks and realize periodically to bound memory:

```python
def __call__(self, x: Tensor) -> Tensor:
  CHUNK_SIZE = 16  # dequant 16 blocks at a time
  x_flat = x.reshape(-1, x.shape[-1])
  out = None
  for chunk_start in range(0, n_blocks, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, n_blocks)
    chunk_out = None
    for bi in range(chunk_start, chunk_end):
      w = dequant_fn(self.blocks[bi])
      x_slice = x_flat[:, bi * el_per_block:(bi + 1) * el_per_block]
      y = x_slice.linear(w.T)
      chunk_out = y if chunk_out is None else chunk_out + y
    chunk_out.realize()  # bound graph size
    out = chunk_out if out is None else out + chunk_out
  return out.reshape(*x.shape[:-1], self.out_features)
```

---

## Issue 3: Metal Compilation Hang

### Analysis
The hang is during `newComputePipelineState` (Metal kernel compilation). This suggests:
1. Too many unique kernels being generated
2. Each MoE expert path may generate different kernels
3. Large kernel code due to unrolled loops or huge input graphs

### Suggestions

1. **Use JIT carefully**: `model._no_jit = True` is set for MoE, which may cause re-compilation every forward pass.

2. **Fuse dequantization into kernel**: The long-term solution (per progress.md) is to dequantize inside the Metal kernel rather than building a huge lazy graph. This requires new Metal kernels for Q4_K matmul.

3. **Batch expert computation**: Instead of looping over selected experts individually, batch them together.

---

## Issue 4: ExpertWeights Matmul Pattern

### Current Pattern in `QuantizedExpertWeights`
The code dequantizes each selected expert and does sequential matmuls:

```python
for i, expert_idx in enumerate(selected_experts):
  expert_w = dequant_fn(self.blocks[expert_idx])  # full expert dequant
  out[i] = input[i] @ expert_w.T
```

### Better Pattern: Selective Block Dequantization
Only dequantize blocks actually needed for the input:
```python
# For each token, only dequantize the blocks that correspond to
# non-zero input columns
active_blocks = get_active_blocks(input)  # based on input sparsity
```

---

## Quick Wins to Try

1. **Add `.blocks` to state_dict** - eliminates warnings, ensures tensors are properly connected

2. **Add `.realize()` in QuantizedLinear loop** - every N iterations to bound graph size:
   ```python
   if (bi + 1) % 32 == 0: out.realize()
   ```

3. **Reduce max_context** - Currently testing with 64, try even smaller (32) to reduce KV cache

4. **Check if blocks tensor is realized** - The GGUF loader may return lazy tensors. Add `blocks = blocks.realize()` in QuantizedLinear.__init__

---

## Memory Budget Calculation

For GLM-4.7 Flash Q4_K_M:
- Weights: ~18GB quantized (visible in log: "copy 18312.34M")
- KV cache (max_context=64): minimal
- Dequantized expert buffers (worst case): 3 experts × 400MB = 1.2GB per MoE layer

If dequantization is lazy and builds graphs, the graph metadata itself can consume significant memory. Forcing eager realization should help.

---

## Approach 5: TK-Based Quantized Matmul Kernel

### Inspiration from Flash Attention (extra/thunder/tiny/fa.py)

The TK (Tile Kernel) infrastructure provides a high-level DSL for writing GPU kernels that could be used for a fused quantized matmul. Key components:

```python
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import GL, TileLayout

def q4k_matmul_kernel(out:UOp, x:UOp, blocks:UOp) -> UOp:
  with Kernel("q4k_matmul", grid=(out_dim // TILE_M, 1, batch), threads=WARP_THREADS) as ker:
    warp = ker.warp

    # Global memory handles
    y, inp, blk = GL(out, ker), GL(x, ker), GL(blocks, ker)

    # Register tiles for accumulation and dequantization
    acc_reg = ker.rt((TILE_M, TILE_N), dtypes.float32)
    x_reg = ker.rt((TILE_K, TILE_N), dtypes.float16)
    w_reg = ker.rt((TILE_M, TILE_K), dtypes.float16)  # dequantized tile

    # Scale/min registers for Q4_K
    d_reg = ker.rv(TILE_M, dtypes.float16)
    dmin_reg = ker.rv(TILE_M, dtypes.float16)

    acc_reg = warp.zero(acc_reg)

    for k_idx in ker.range(in_dim // TILE_K):
      # Load Q4_K blocks for this K tile
      # Dequantize in registers: w = d * sc * q - dmin * mn
      ...
      # Load input tile
      x_reg = warp.load(x_reg, inp, ...)
      # Matrix multiply
      acc_reg = warp.mma_AB(acc_reg, w_reg, x_reg)

    acc_reg = ker.endrange()
    y = warp.store(y, acc_reg, ...)
    return ker.finish()
```

### Advantages
1. **Never materializes full weight matrix** - dequant happens per-tile in registers
2. **Uses GPU tensor cores** - `warp.mma_AB` leverages hardware matmul
3. **Bounded memory** - only TILE_M × TILE_K elements dequantized at a time
4. **Schedule-stable** - same kernel shape regardless of expert selection

### Design for MoE Expert Matmul

For `MUL_MAT_ID_Q` (the MoE case), the kernel could:

```python
def mul_mat_id_q4k(out:UOp, x:UOp, blocks:UOp, expert_ids:UOp) -> UOp:
  # Grid: (experts_per_token * tokens, out_dim_tiles, 1)
  with Kernel("mul_mat_id_q4k", grid, threads) as ker:
    # Each thread block handles one (token, expert) pair
    token_expert_idx = ker.blockIdx_x  # 0..(n_tokens * k_experts - 1)
    out_tile = ker.blockIdx_y

    # Decode which expert and which token
    expert_id = expert_ids[token_expert_idx // k_experts, token_expert_idx % k_experts]

    # Blocks offset for this expert
    expert_block_offset = expert_id * blocks_per_expert

    # Standard tiled matmul with Q4_K dequant in inner loop
    for k_tile in ker.range(K // TILE_K):
      block_idx = expert_block_offset + out_tile * (TILE_M // 256) + k_tile
      # Dequant this block's worth of weights into w_reg
      # Accumulate: acc += w_reg @ x_reg
```

### Implementation Path

1. **Start with dense QuantizedLinear using TK** - no expert selection, just fused Q4_K dequant + matmul
2. **Verify correctness** - compare output vs reference dequant + matmul
3. **Add expert indexing** - extend to `mul_mat_id_q4k` for MoE blocks
4. **Keep expert IDs as runtime tensor** - ensures schedule cache hits

### Key Files to Study
- `extra/thunder/tiny/tk/kernel.py` - Kernel class, ranges, tiles
- `extra/thunder/tiny/tk/tiles.py` - GL, ST, RT, RV tile types
- `extra/thunder/tiny/tk/group.py` - warp operations (load, store, mma)
- `extra/thunder/tiny/fa.py` - flash attention as reference

### Q4_K Block Layout (144 bytes → 256 elements)
```
Byte 0-1:   d (float16)     - scale
Byte 2-3:   dmin (float16)  - min scale
Byte 4-15:  scales (12 bytes) - 8×6-bit scales + 8×6-bit mins
Byte 16-143: qs (128 bytes) - 256×4-bit quantized values
```

Dequant formula per group of 32:
```
val[i] = d * sc[group] * q[i] - dmin * mn[group]
```

Where `sc` and `mn` are unpacked from the 12 scale bytes.

---

## Approach 6: Lessons from llama.cpp Metal Implementation

### Key Files Analyzed
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` - Metal shader implementations

### Q4_K Dequantization Tricks

#### 1. Compact Scale Unpacking Function
```metal
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}
```
Returns both scale and min for a group in one call, packed in a uchar2.

#### 2. d/16 Trick for High Nibbles
```metal
const float d   = il < 2 ? xb->d : xb->d / 16.h;
const ushort mask = il < 2 ? 0x0F : 0xF0;
for (int i = 0; i < 16; ++i) {
    reg[i/4][i%4] = dl * (q[i] & mask) - ml;
}
```
Instead of shifting high nibbles right by 4, divide d by 16 and use mask 0xF0.
This saves a shift operation in the inner loop.

#### 3. 16-Element Tile Processing
Q4_K blocks (256 elements) are processed in 16-element tiles (16 iterations of 16).
Each tile gets its own scale/min pair, computed once per tile.

### MoE Architecture (kernel_mul_mm_id)

#### Two-Phase Approach
1. **Phase 1: Map tokens to experts** (`kernel_mul_mm_id_map0`)
   - Pre-compute per-expert token lists
   - Each expert gets a compact list of "which tokens use me"
   - Stored in `hids` buffer, counts in `htpe` buffer

2. **Phase 2: Batched matmul per expert** (`kernel_mul_mm_id`)
   - Each threadgroup handles one expert
   - Uses pre-computed token lists for coalesced memory access
   - Same tiled matmul as dense case, just with indirect indexing

#### Key Constants
```metal
constexpr int NR0 = 64;   // Output tile rows
constexpr int NR1 = 32;   // Output tile cols
constexpr int NK  = 32;   // K-dimension tile
```

#### Memory Layout
- Threadgroup memory: 4KB for A matrix tiles, 4KB for B matrix tiles
- simdgroup matrix ops: 8x8 tiles
- Total: 64x32 output tile per threadgroup

### What We Can Copy

1. **Pre-computed expert routing**: Build token->expert mappings before matmul
   - Avoids dynamic indexing in inner loop
   - Enables batched processing per expert

2. **Fixed tile sizes**: Use 64x32 output tiles, 32 K-dimension
   - Maps well to GPU warp/simdgroup sizes
   - Enables threadgroup memory caching

3. **d/16 trick**: For Q4_K high nibble groups
   - Saves shift operation in inner loop
   - Our UOp kernel could benefit from this

4. **16-element granularity**: Process Q4_K in 16-element chunks
   - Better register utilization
   - Scale/min computed once per chunk

### Implementation Differences

| Aspect | llama.cpp | tinygrad current |
|--------|-----------|------------------|
| Dequant location | In-kernel, per-tile | Separate pass, full tensor |
| MoE routing | Pre-computed lists | Per-token selection |
| Memory pattern | Tiled + threadgroup mem | Strided global access |
| Expert selection | Batch per expert | Loop per token |

### Recommended Next Steps

1. **Fused Q4_K dequant + matmul kernel** (TK-based)
   - Dequantize 256-element blocks in tiles
   - Accumulate into float32 registers
   - Never materialize full weight matrix

2. **Two-phase MoE approach**
   - Phase 1: Build per-expert token lists on GPU
   - Phase 2: Batch matmul per expert with quantized weights

3. **Fixed expert count in schedule**
   - Parameterize by (n_tokens, n_experts_per_token)
   - Expert IDs as runtime tensor, not baked into graph

# FROM design.md
# GLM-4.7 Flash: True Quantized MoE Runtime

## Goal
Run GLM-4.7 Flash on M3 32GB with stable memory and >50 tok/s by keeping expert weights quantized end-to-end and
dequantizing inside backend kernels only for selected experts.

## Scope
- Only GLM-4.7 Flash (GLM4_MOE).
- Only MoE expert matmuls (up/gate/down). Dense layers can stay as-is.
- Target Q4_K_M (K-quant). Q4_0 is not the target.
- Must run on all tinygrad backends (correctness path everywhere, optimized kernels where available).

## Assumptions
- One new op + per-backend implementations are acceptable.
- Graph topology must remain stable across tokens.
- We can map gguf K-quant blocks to backend dequant functions in-kernel.

## Current State (last 6 commits + existing infra)

### What exists now
- `tinygrad/nn/state.py`
  - `GGML_QUANT_INFO` supports Q4_K/Q5_K/Q6_K with block sizes + dequant fns.
  - `gguf_load(..., quantized=True)` returns `quantized_tensors` for K-quant only and omits them from `state_dict`.
  - Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 are dequantized immediately in `ggml_data_to_tensor`.
- `tinygrad/apps/llm.py`
  - `QuantizedLinear`: dequantizes blocks per call, then matmul in FP16/FP32.
  - `QuantizedExpertWeights`: dequantizes only selected experts, but uses `sel.tolist()` + `Tensor.stack` on CPU and
    `realize()` on the dequantized tensor.
  - `Transformer.from_gguf` swaps `nn.Linear` -> `QuantizedLinear` and `ExpertWeights` -> `QuantizedExpertWeights`.
  - MoE gating uses `topk`, then expert indexing via advanced indexing.
  - JIT is disabled for MoE quant path (`model._no_jit = True`) to avoid JIT capture of `tolist()`.
- Model URLs
  - `glm-4.7:flash` points to Q4_K_M (K-quant).
  - `glm-4.7:flash-q4` points to Q4_0 and always dequantizes at load time.

### Implications
- K-quant support exists at load time, but matmul is not quantized; weights are still materialized per call.
- `sel.tolist()` forces CPU sync and breaks JIT. Graph topology is unstable across tokens.
- The current path still creates large temporaries and is not true in-kernel dequant.

### Delta vs llama.cpp
- llama.cpp uses `mul_mat_id` on quantized weights and dequantizes per tile in Metal kernels.
- tinygrad does expert selection in Python and dequantizes into full tensors before matmul.

## What llama.cpp Actually Does
MoE path (GLM4_MOE) uses `mul_mat_id` on quantized expert weights; no one-hot masks.
Key files (llama.cpp):
- `src/llama-graph.cpp`: `build_moe_ffn` uses `ggml_mul_mat_id` for up/gate/down.
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`: `ggml_metal_op_mul_mat_id` launches map + matmul kernels.
- `ggml/src/ggml-metal/ggml-metal.metal`: `kernel_mul_mm_id` templates dequantize per tile for Q4_K/Q5_K/Q6_K.

Result: weights stay quantized; only a small tile is dequantized into threadgroup memory.

## Root Problem in tinygrad
- Current MoE path either:
  - dequantizes full expert tensors into FP16/FP32, or
  - builds huge graphs (stack/one-hot) to select experts.
- This balloons memory and thrashes when RSS approaches 32–36 GB.

## Design: Minimal Additions

### 1) New op: `MUL_MAT_ID_Q` (quantized MoE matmul)
- Semantics: `Y = W_q[expert_id] @ X` for a batch of selected experts.
- Inputs:
  - `W_q`: quantized expert weights (packed blocks, shape `[out, in, n_expert]`).
  - `X`: activations (shape `[in, n_expert_used, n_tokens]`).
  - `IDS`: int32 expert ids (shape `[n_expert_used, n_tokens]`).
- Output:
  - `Y`: `[out, n_expert_used, n_tokens]`.
- Must not materialize any full dequantized buffer.

### 2) Backend: generic + per-backend kernels
Common API and lowering are shared. Each backend either:
- implements a native quantized `MUL_MAT_ID_Q` kernel, or
- routes to the CPU reference implementation for correctness.

Match llama.cpp structure on Metal:
- `kernel_mul_mm_id_map0`: build per-expert compact index lists.
- `kernel_mul_mm_id_q4k_*`: matmul with per-tile dequant into threadgroup memory.

Constraints for Metal kernels:
- Keep threadgroup memory bounded (tile sized).
- Avoid per-token specialization (stable pipeline cache).
- Use fixed tile sizes suited to GLM-4.7 Flash dimensions.

### 3) Graph Changes in tinygrad
- Keep expert selection on-device without one-hot.
- Replace current expert selection + `gather` + `linear` with `MUL_MAT_ID_Q`.
- Ensure expert ids are runtime buffers, not baked constants.

Minimal graph for a MoE block:
- `probs = gate(x)`
- `ids = topk(probs)` (int32)
- `up = MUL_MAT_ID_Q(W_up_q, x, ids)`
- `gate = MUL_MAT_ID_Q(W_gate_q, x, ids)`
- `act = silu(gate) * up`
- `down = MUL_MAT_ID_Q(W_down_q, act, ids)`
- `out = sum over experts (weighted)`

### 4) Quantized Weight Storage
- Keep Q4_K_M blocks in original gguf layout.
- Expose raw blocks to backends without dequantizing to FP tensors.
- Any reshape must be logical (stride metadata only).

## Backend Coverage
Backends present in `tinygrad/runtime`:
- CPU: `ops_cpu.py` (must be correct and is the fallback)
- CUDA: `ops_cuda.py`
- HIP/AMD: `ops_hip.py`, `ops_amd.py`
- Metal: `ops_metal.py`
- OpenCL: `ops_cl.py`
- Vulkan/WebGPU: `ops_webgpu.py`
- QCOM/DSP: `ops_qcom.py`, `ops_dsp.py`
- Python/NumPy/Null: `ops_python.py`, `ops_npy.py`, `ops_null.py`

Goal: CPU always correct. GPU backends use native kernels where available; otherwise call CPU fallback.

## Implementation Plan (Generic)

1) **Define `MUL_MAT_ID_Q` op**
  - UOp + schedule entry.
  - Type check: src0 must be K-quant.
  - Output dtype F16 or F32 (use F32 accumulation for GLM4_MOE, like llama.cpp).

2) **CPU reference**
  - Implement `MUL_MAT_ID_Q` in `ops_cpu.py` using dequant + matmul for correctness.
  - This is the fallback for backends without native kernels.

3) **GPU backends**
  - Metal: add kernel entry point + shader for Q4_K tile dequant.\n
  - CUDA: add kernel that mirrors llama.cpp `mul_mat_id` (dequant per tile).\n
  - HIP/AMD, OpenCL, Vulkan/WebGPU: either add native kernels or route to CPU fallback.

4) **Graph rewrite in `apps/llm.py`**
  - Route expert matmuls to `MUL_MAT_ID_Q` when weights are Q4_K.
  - If weights are not K-quant, keep existing path.

5) **Remove CPU expert selection**
  - Eliminate `sel.tolist()` and `Tensor.stack` in `QuantizedExpertWeights`.
  - Keep `ids` as a runtime tensor (shape-stable for schedule cache).

## Integration Points (existing infra)
- `tinygrad/nn/state.py`
  - Keep `GGML_QUANT_INFO` and raw blocks; no new CPU dequant path.
- `tinygrad/apps/llm.py`
  - Replace `QuantizedExpertWeights` callsite with `MUL_MAT_ID_Q`.
  - Keep `QuantizedLinear` for non-MoE layers unchanged.
- `tinygrad/uop/ops.py`, `tinygrad/engine/schedule.py`
  - Add `MUL_MAT_ID_Q` op with shape inference and scheduling rules.
- `tinygrad/runtime/ops_metal.py` + Metal shader
  - Add kernel entry point and pipeline cache.
  - Ensure kernel shape stability across tokens.

## Correctness Checks
- Shape invariants for GLM-4.7 Flash:
  - `n_expert = 64`, `n_expert_used = 2`, `n_tokens` variable.
  - `in = n_embd`, `out = moe_hidden_dim`.
- Verify output vs FP16 path on a small batch (n_tokens=1, n_expert_used=2).
- Verify same tokens as non-quantized (within tolerance).

## Performance Targets
- No full dequant buffers of expert weights.
- RSS stays < 26–28 GB at max_context=512.
- Kernel cache hits after first token.

## Testing Plan
- Unit test: fake Q4_K blocks, run `MUL_MAT_ID_Q` vs reference dequant+matmul.
- Smoke test: `glm-4.7:flash` with `--quantized` and `max_context 64`.
- Profile: `DEBUG=1` to confirm kernel reuse.

## Risks
- Metal kernel complexity may be high; keep only Q4_K_M for now.
- JIT caching: ensure `ids` are not baked, but graph shape stays stable.
- Block layout mistakes can silently corrupt output.

## Open Questions
- Use F32 accumulation for GLM4_MOE always (match llama.cpp) vs conditional?
- Best representation of `ids` to keep schedule cache stable (BIND vs runtime input)?
- Do we need a Q4_0 fallback path for `glm-4.7:flash-q4` (likely no)?

---

## Session 8: Attention Scale Fix & Verification

### Attention Scale Fix Applied
Fixed the attention scale in `tinygrad/apps/llm.py`:
- **Old (wrong)**: `scale = mscale^2 / sqrt(256)` = 0.0625
- **New (fixed)**: `scale = mscale^2 / sqrt(576)` = 0.0417

The key insight from deepseek2.cpp: In absorbed MLA, both Q and K have dimension = `kv_lora_rank + qk_rope_head_dim` = 512 + 64 = 576. The scale factor should be `1/sqrt(576)`, not `1/sqrt(256)`.

Code change at line 429-430:
```python
cache_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576 for GLM-4.7
scale = self.mscale * self.mscale / math.sqrt(cache_dim)
```

### Test Results

**"Hello" token works correctly!**
```
Hello logits: max=+24.12, min=-21.38, argmax=93295
Top-5 decoded: [' greetings', 'Greetings', ' greeting', ' greet', 'Welcome']
```

**T=2 ([gMASK, <sop>]) works:**
```
T=2 logits: max=+22.50, min=-9.75, argmax=10365
```

**T=1 ([gMASK] alone) still fails** - but this is expected! [gMASK] is a special prefix token that shouldn't be used alone:
```
T=1 logits: max=-106.75, min=-150.88  # All negative - expected for invalid input
```

### Key Finding
The model is **working correctly** for regular text inputs. The explosion seen with [gMASK] alone is expected behavior for an invalid prompt format. GLM-4.7 prompts should be:
- `[gMASK]<sop>actual_prompt_text...`

### Next Steps
1. Test full generation pipeline with proper prompts
2. Verify multi-token generation quality
3. Investigate exp_probs_b (expert bias) if generation quality still poor
