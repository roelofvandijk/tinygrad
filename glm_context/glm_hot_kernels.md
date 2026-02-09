# GLM-4.7-Flash-Q4_0 Hot Kernel Analysis (Feb 8, 2026)

**17.93 tok/s** (55.8 ms/tok), 1404 kernels/tok, 6 ICBs, 46 MoE + 1 dense block

Theoretical: **80 tok/s** (1.24 GB / 100 GB/s = 12.4 ms). Current efficiency: **22%**.

---

## Top 6 Kernels by Warm-Up Time

| # | Kernel | us/call | BW | Calls | Tot ms | What | Source |
|---|--------|---------|-----|-------|--------|------|--------|
| 1 | `r_96_16_4_4_128_2048` | **373** | 34 GB/s | 138 | 51.5 | MoE gate·silu·up fused (multireduce) | [mla.py:139-141](tinygrad/apps/mla.py#L139-L141) |
| 2 | `r_2048_16_2_2_3_16` | **234** | 30 GB/s | 126 | 29.4 | Expert down (Q4_0 packed-dot) | [quantized.py:126-137](tinygrad/apps/quantized.py#L126-L137) |
| 3 | `r_384_16_4_4_4_16` | **157** | 45 GB/s | 138 | 21.6 | Expert gate (Q4_0 packed-dot) | [quantized.py:126-137](tinygrad/apps/quantized.py#L126-L137) |
| 4 | `r_384_16_4_4_4_16n1` | **152** | 47 GB/s | 138 | 21.0 | Expert up (Q4_0 packed-dot) | [quantized.py:126-137](tinygrad/apps/quantized.py#L126-L137) |
| 5 | `r_128_16_4_4_96` | **74** | 86 GB/s | 138 | 10.3 | MoE weighted sum | [mla.py:145](tinygrad/apps/mla.py#L145) |
| 6 | `r_2048_16_2_2_3_16` (shared) | ~234 | ~30 GB/s | 12 | — | Shared expert down | [mla.py:147](tinygrad/apps/mla.py#L147) |

Total for top 4 expert kernels: **~123ms** warmup = ~56% of all kernel time.

---

## Kernel #1: MoE Gate·Silu·Up Fused (THE bottleneck)

**`r_96_16_4_4_128_2048`** — 373us, 34 GB/s, **multireduce**

Opts: `GROUP(0,16), LOCAL(0,4), UPCAST(0,4)` — our Q4_0 GROUP heuristic fires ([heuristic.py:91-110](tinygrad/codegen/opt/heuristic.py#L91-L110))

### What it does

This kernel fuses the entire `gate.silu() * up` computation from [mla.py:139-141](tinygrad/apps/mla.py#L139-L141):
```python
gate = self.ffn_gate_exps(sel, h_norm).silu().contiguous()  # ← first reduce (128 iters)
up = self.ffn_up_exps(sel, h_norm).contiguous()             # ← second reduce (2048 iters)
gated = gate * up
```

Despite the `.contiguous()` calls, the scheduler **re-fuses** gate and up into one multireduce kernel. Why: the `.contiguous()` after `.silu()` DOES break fusion (ALU op prevents `found_contiguous` in [rangeify.py:547-554](tinygrad/schedule/rangeify.py#L547-L554)), but somehow the scheduler still produces a fused kernel for GLM. The key difference from ds2-lite: GLM's expert dimensions are different (4 experts selected vs 6), leading to different scheduler decisions.

### Metal source (abridged)

```metal
kernel void r_96_16_4_4_128_2048(
  device float* data0_1536,        // output: 4×384 = 1536 floats
  device float* data1_2048,        // routing probs
  device float* data2_1,           // expert_weights_scale (1.8)
  device half*  data3_2048,        // activation (h_norm)
  device half*  data4_3145728,     // gate expert weights (3M halfs = fp16, NOT Q4_0!)
  device half*  data5_3145728,     // up expert weights (3M halfs = fp16, NOT Q4_0!)
  ...
) {
  threadgroup float temp0[256];    // GROUP(0,16) shared memory
  float acc0[4], acc1[4], acc2[4]; // 3 accumulators

  // FIRST REDUCE: gate matmul (128 iterations — GROUP splits to 8/thread)
  for (int Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    half val1 = *(data3_2048 + ...);      // activation
    half val2 = *(data4_3145728 + ...);    // gate weight
    acc0[i] += (probs * scale * act) * gate_w;  // 4 outputs
  }

  // SECOND REDUCE: up matmul (2048 iterations — FULLY SERIAL, no GROUP!)
  for (int Ridx1 = 0; Ridx1 < 2048; Ridx1++) {
    half val7 = *(data3_2048 + Ridx1);    // activation (re-read!)
    half val11 = *(data5_3145728 + ...);   // up weight
    acc1[i] += (probs * scale * act) * up_w;  // 4 outputs
  }

  // GROUP reduction for first reduce only
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int Ridx103 = 0; Ridx103 < 16; Ridx103++) { ... }

  // Output: silu(gate_result) * up_result
  *(data0 + ...) = silu(acc_gate) * acc_up;
}
```

### Why it's slow

1. **Multireduce**: Two reductions (128 + 2048 iters). GROUP only helps the first (128→8 per thread). The second (2048) runs fully serial. See [postrange.py:155-158](tinygrad/codegen/opt/postrange.py#L155-L158) for the constraint.

2. **It reads fp16 expert weights, not Q4_0 blocks.** `data4_3145728` = 3M halfs = 6MB = fp16 dequanted weights. This means the Q4_0 packed-dot path ([quantized.py:126-137](tinygrad/apps/quantized.py#L126-L137)) is NOT being used here. The scheduler is fusing the dequant+matmul differently for GLM.

3. **96 workgroups only** (gidx0 range). For a ~6MB read, need 1000+ workgroups to saturate bandwidth. 96 workgroups = severe underutilization.

4. **Activation re-read**: `data3_2048` is read twice (once per reduce), but it's only 4KB so stays in cache.

### Why `.contiguous()` didn't split it

The `.contiguous()` in [mla.py:139](tinygrad/apps/mla.py#L139) after `.silu()` should prevent re-fusion. But GLM's expert setup differs from ds2-lite: 4 experts (not 6) with different dimensions. The scheduler may be making different fusion decisions. Need to verify with `DEBUG=4` whether the `.contiguous()` is actually producing separate schedule items or being elided.

---

## Kernel #2: Expert Down (Q4_0 packed-dot)

**`r_2048_16_2_2_3_16`** — 234us, 30 GB/s

Opts: `GROUP(0,16), LOCAL(0,2), UPCAST(0,2)` — our heuristic fires but with small LOCAL/UPCAST

### Metal source (abridged)

```metal
kernel void r_2048_16_2_2_3_16(
  device half* data0_8192,            // output: 4×2048 = 8192 halfs
  device unsigned char* data1_7077888, // Q4_0 blocks (expert down weights)
  device half* data2_6144,            // gate output
  device half* data3_6144,            // up output
  ...
) {
  // Q4_0 dequant+matmul: scale * ((nib & 0xF) - 8) * gate * up
  for (int Ridx0_0 = 0; Ridx0_0 < 3; Ridx0_0++) {      // outer: 3 blocks
    for (int Ridx0_1 = 0; Ridx0_1 < 16; Ridx0_1++) {    // inner: 16 packed bytes
      unsigned char val4 = *(data1 + ...);                // packed Q4_0 byte
      acc0[0] += scale * (((nib & 0xF) - 8) * gate * up + ((nib >> 4) - 8) * gate * up);
    }
  }
  // GROUP reduction
  threadgroup_barrier(...);
}
```

### Why it's slow

1. **Only LOCAL(0,2), UPCAST(0,2)** — global dim 2048 gives 2×2=4 which divides, but this means only 4 outputs per workgroup. Compare with gate/up which get LOCAL(0,4), UPCAST(0,4) = 16 outputs per workgroup.

2. **2048 workgroups** — good for parallelism, but each does very little work (3×16=48 iterations per reduce, then GROUP reduction). High ratio of sync-to-compute.

3. **Scattered byte reads** — `data1_7077888` (6.75MB Q4_0 blocks), accessed byte-by-byte. 2048 threads × 48 bytes each = scattered across 6.75MB.

4. **Reads BOTH gate and up outputs** — `data2_6144` and `data3_6144`. This is the fused `down(gate_out * up_out)` where gate and up outputs are separate buffers (from our `.contiguous()` split).

### Potential: reduce reads by fusing gate·up into down

If gate·up·down were one kernel, gate and up results would be registers, not 6MB memory reads. But that creates a triple-reduce kernel — even worse for the optimizer.

---

## Kernels #3 & #4: Expert Gate/Up (Q4_0 packed-dot)

**`r_384_16_4_4_4_16`** / **`r_384_16_4_4_4_16n1`** — 157/152us, 45/47 GB/s

Opts: `GROUP(0,16), LOCAL(0,4), UPCAST(0,4)` — good match

These are the well-optimized Q4_0 expert matmuls from [quantized.py:126-137](tinygrad/apps/quantized.py#L126-L137).

### Dimensions

GLM: 4 experts × 1536 output × 2048 input. Q4_0: 2048/32 = 64 blocks per row.
- 384 = 4 experts × 96 output groups (1536/16)
- Reduce: 4×4×4×16 = 1024 (= 64 blocks × 16 bytes per block)

### Why they're at 45-47 GB/s (not 100+)

Same ceiling as ds2-lite: scattered byte reads in tinygrad-generated code. Each thread reads individual `unsigned char` values from Q4_0 blocks. llama.cpp's MSL kernel reads `uint16_t` (2 bytes) and uses SIMD reduction, achieving 145+ GB/s.

The heuristic approach cannot fix the memory access pattern — that's determined by codegen.

---

## Kernel #5: MoE Weighted Sum

**`r_128_16_4_4_96`** — 74us, 86 GB/s

Opts: `GROUP(0,16), LOCAL(0,4), UPCAST(0,4)` — our heuristic fires (has bitwise ops? or is this MV?)

This is `(expert_out * probs).sum(axis=2) * scale` from [mla.py:145](tinygrad/apps/mla.py#L145). After the `.contiguous()` on expert_out, this becomes a separate kernel. 86 GB/s is reasonable — it's a simple weighted sum of 4 expert outputs.

---

## Kernel Budget (estimated per-token)

| Component | us/call | ×calls/tok | ms/tok | % |
|-----------|---------|------------|--------|---|
| Gate·silu·up fused `r_96_*_2048` | 373 | 46 | **17.2** | 31% |
| Down proj `r_2048_*_3_16` | 234 | ~42 | **9.8** | 18% |
| Gate Q4_0 `r_384_*` | 157 | 46 | **7.2** | 13% |
| Up Q4_0 `r_384_*n1` | 152 | 46 | **7.0** | 13% |
| Weighted sum `r_128_*_96` | 74 | 46 | **3.4** | 6% |
| Shared expert down | 234 | 12 | **2.8** | 5% |
| Attention (all) | ~50 | ~140 | **3.5** | 6% |
| Norms + topk + ewise | ~12 | ~400 | **4.8** | 9% |
| **Total** | | 1404 | **~55.8** | 100% |

Expert gate+up+down = **41.2ms/tok** = **74% of total time**.

---

## Potential Solutions (ranked by expected impact)

### 1. Custom `mul_mat_id` CompiledRunner for Q4_0 MoE (expected: 18→30+ tok/s)

Replace tinygrad's generated expert kernels with hand-written Metal (like the existing Q4K/Q6K runners in [metal_q4k.py](tinygrad/nn/metal_q4k.py) and [metal_q6k.py](tinygrad/nn/metal_q6k.py)).

**Why this works**: The existing Q4K MSL kernel hits 145 GB/s for dense matvec. An `mul_mat_id` variant would:
- Use `uint16_t` reads (coalesced) instead of `unsigned char` (scattered)
- SIMD-reduce within threadgroup instead of serial accumulation
- Handle expert selection (gather) inside the kernel, eliminating separate gather kernels
- Extend `CompiledRunner` for JIT batching ([metal_q4k.py pattern](tinygrad/nn/metal_q4k.py))

**Target kernels**: `r_384_*` (gate/up, 45 GB/s → 145 GB/s = 3.2x), `r_2048_*` (down, 30 GB/s → 145 GB/s = 4.8x)

**Expected saving**: gate+up from 14.2ms → ~4.5ms, down from 9.8ms → ~2ms. Total: -17.5ms = **18→26 tok/s**.

### 2. Break the gate·silu·up multireduce (expected: eliminate kernel #1)

Kernel #1 (`r_96_*_2048`) reads fp16 expert weights, not Q4_0. This suggests the scheduler is NOT using the Q4_0 packed-dot path for the fused kernel. If we can force the split so gate and up use the packed-dot path (kernels #3/#4), kernel #1 disappears.

**Investigation needed**: Why does GLM's scheduler fuse gate+up differently than ds2-lite? The `.contiguous()` in [mla.py:139](tinygrad/apps/mla.py#L139) should break it. Check with `DEBUG=4` whether found_contiguous is eliding it.

**If split works**: kernel #1 (17.2ms) → kernels #3+#4 already exist (14.2ms combined). Net saving: ~3ms.

**Combined with solution 1**: If custom MSL handles gate+up+down, the fused kernel is replaced entirely.

### 3. Improve down kernel LOCAL/UPCAST (expected: small)

Down kernel gets only LOCAL(0,2), UPCAST(0,2). Global dim is 2048, so 2048 % (4×4) = 0, meaning (4,4) should work. The issue might be that the heuristic tries GROUP(0,16) first, and after GROUP the global shape changes.

**Try**: Larger GROUP first, or try LOCAL/UPCAST on a different axis. But ceiling is still ~50 GB/s due to scattered reads. Low priority.

### 4. Kernel count reduction (expected: 1404→~1000)

1404 kernels × ~40us overhead = ~56ms just in dispatch. Reducing kernel count by 400 saves ~16ms.

Targets from [bottlenecks.md](glm_context/bottlenecks.md#L335-L343):
- Eliminate cache_v (V = slice of K): -47 kernels
- Fuse RMSNorm: -230 kernels
- Fuse QK + softmax: -46 kernels
- Expert gather consolidation: -138 kernels

### 5. Different quantization path for GLM experts

GLM's GGUF has Q4_K experts (gate/up) and Q6_K (down), not Q4_0. But the current Q4_0 GGUF uses Q4_0 for everything. If we switch to Q4_K GGUF with custom Q4K MSL runners (already exist), we get better accuracy AND custom kernel speed.

The Q4K runner in [metal_q4k.py](tinygrad/nn/metal_q4k.py) already extends `CompiledRunner` and hits 145 GB/s for dense. An MoE variant would need expert selection logic.

---

## Resolved: Kernel #1 Was Shared Expert (Q5_K → fp16), Not Expert Gate+Up

**Root cause**: The "Q4_0" GGUF has mixed quantization types. Expert gate/up weights are Q4_0 (type 2), but **shared expert weights are Q5_K (type 13)**. `QuantizedLinear` only has packed-dot for Q4_0, so Q5_K falls through to `_ensure_dequant_cache` → fp16. The fused `gate.silu() * up` creates a multireduce reading 12MB fp16 per call.

**Fix** ([mla.py:146-149](tinygrad/apps/mla.py#L146-L149)): Split shared expert gate and up with `.contiguous()`:
```python
shexp_gate = self.ffn_gate_shexp(h_norm).silu().contiguous()
shexp_up = self.ffn_up_shexp(h_norm).contiguous()
out = out.contiguous() + self.ffn_down_shexp(shexp_gate * shexp_up)
```

**Result**: 17.93 → **20.9 tok/s (+17%)**. The 373us multireduce (34 GB/s) becomes two MV-matched matmuls at 72us (89 GB/s) + 94us (91 GB/s). The +46 extra kernels add ~1.6ms overhead, but the shared expert speedup saves ~9.5ms.

### Open Questions

1. **Call count mismatch**: kernel #2 (down) has 126 calls (= 42 × 3), not 138 (= 46 × 3). Why 42 not 46?

2. **Q4_0 GROUP heuristic has ~0% impact on GLM.** The 17.93 → 20.9 tok/s improvement is entirely from the shared expert split. The GROUP heuristic was already firing on expert kernels but not improving them meaningfully.
