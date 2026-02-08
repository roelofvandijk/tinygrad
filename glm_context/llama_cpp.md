# llama.cpp Reference Implementation

## Core Architecture: Separate Ops, Not One Fused Kernel

llama.cpp does **NOT** fuse gate+silu+up+down+sum+residual into one kernel.
Instead it uses a pipeline of individually optimized operations:

```
1. gate = mul_mat_id(gate_exps, input, expert_ids)  // separate GEMV per expert
2. up   = mul_mat_id(up_exps,   input, expert_ids)  // separate GEMV per expert
3. act  = swiglu(gate, up)                           // fused activation
4. down = mul_mat_id(down_exps, act,   expert_ids)   // separate GEMV per expert
5. weighted = down * expert_weights                   // elementwise
6. out = weighted[:,0,:] + weighted[:,1,:] + ... + weighted[:,k-1,:]  // explicit adds
```

Source: `reference_material/llama.cpp/src/llama-graph.cpp:1093-1360`

---

## `mul_mat_id` — The MoE Primitive

**Not gather→matmul. A single fused kernel with built-in expert indexing.**

```c
// ggml.c:3203
// as  -> [cols, rows, n_expert]          // all expert weights stacked
// b   -> [cols, n_expert_used, n_tokens] // input (same for all experts)
// ids -> [n_expert_used, n_tokens]       // which expert for each slot
// c   -> [rows, n_expert_used, n_tokens] // output
c = ggml_mul_mat_id(ctx, as, b, ids);
```

The kernel directly reads from `weights[:, :, expert_id]` — no intermediate gather buffer.

---

## Metal Backend: Per-Expert Threadgroup Dispatch

Two paths in `ggml-metal-ops.cpp:1980-2166`:

### Decode (single token): `mul_mv_id`
```
threadgroups: (ne01/nr0) × 1 × (n_expert_used × n_tokens)
threads_per: 32 × nsg × 1
```

Each Z-threadgroup handles ONE (expert, token) pair. Full SIMD group parallelism (32 × nsg threads) for the inner reduction.

For deepseek-v2-lite (dim=2048, 6 experts):
- **llama.cpp**: ~(2048/nr0) × 6 threadgroups, each does parallel GEMV with 32×nsg threads
- **tinygrad**: 2048 threadgroups × 16 threads, each serially loops 6×1408 = 8,448 iterations

### Batch (prefill): `mul_mm_id`
Two-phase:
1. **Map kernel** (`mul_mm_id_map0`): builds compact expert→token mapping
2. **GEMM kernel**: `(n_tokens/32) × (ne01/64) × n_experts` threadgroups, 128 threads each

---

## Weighted Sum: Explicit Adds, Not Reduction

```cpp
// llama-graph.cpp:1346-1350
ggml_tensor * moe_out = cur_experts[0];
for (uint32_t i = 1; i < n_expert_used; ++i) {
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
}
```

For k=6: `e0 + e1 + e2 + e3 + e4 + e5` — five add operations. Each is a highly optimized elementwise kernel. No reduction dimension needed.

Note from code comment:
> explicitly use hparams.n_expert_used instead of n_expert_used to avoid potentially a large number of add nodes during warmup

---

## Shared Expert: Completely Separate Pipeline

The shared expert (DeepSeek/GLM) runs as a fully independent computation:
```
shared_out = down_shexp(silu(gate_shexp(x)) * up_shexp(x))
```
Never fused with MoE experts. Runs as separate matmuls. This avoids constraining threadgroup sizing of MoE kernels.

---

## What This Means for tinygrad

### The Problem
tinygrad fused everything into one kernel:
```metal
// Part 1: MoE down proj — ALL 16 threads compute identically (redundant!)
for (Ridx2 = 0; Ridx2 < 6; Ridx2++) {        // 6 experts, serial
  for (Ridx1 = 0; Ridx1 < 1408; Ridx1++) {   // inner dim, serial
    acc0 += data3[Ridx2*1408+Ridx1] * data5[gidx0*1408+Ridx1+expert*2883584];
  }
  acc1 += acc0 * prob[Ridx2];                  // weighted accumulation
}
// Part 2: Shared expert — 16 threads properly parallel (176 each)
// Part 3: threadgroup reduction, residual add
```

GROUPTOP=16 chosen for Part 2 (shared expert), leaving Part 1 fully serial. Result: 4.5ms at 31 GB/s.

### Why llama.cpp Is Faster
1. **Down projection is its own kernel** — gets full GEMV parallelism
2. **Each expert dispatches separately** — no serial expert loop
3. **Shared expert is separate** — doesn't constrain MoE threadgroup sizing
4. **Explicit adds** — k-1 tiny elementwise kernels instead of one sum reduction

### Actionable Options for tinygrad

**Option A: Break fusion with `.contiguous()`** — Already exists as `MOE_ADDS=2`. Lets down projection get its own threadgroup sizing.

**Option B: Explicit adds (llama.cpp style)** — `weighted[:,:,0,:] + ... + weighted[:,:,k-1,:]`. Eliminates reduction dimension.

**Option C: Separate shared expert** — `.contiguous()` between MoE output and shared expert. Shared expert becomes single-reduce, eligible for GROUP/MV.

**Option D: Custom `mul_mat_id` kernel** — Metal kernel with per-expert Z-threadgroups like llama.cpp. Most work but highest ceiling.

---

## CUDA `mmid.cu` Preprocessing

For CUDA, llama.cpp runs a preprocessing kernel (`mm_ids_helper`) that:
1. Groups tokens by expert — builds compact sorted representation
2. Computes `expert_bounds[i:i+1]` — where each expert's tokens start/end
3. Dispatches standard GEMM per expert with contiguous token blocks

Avoids scattered access — each expert's matmul operates on contiguous memory. Template-specialized for n_expert_used = 2, 4, 6, 8, 16, 32.

Source: `reference_material/llama.cpp/ggml/src/ggml-cuda/mmid.cu:22-116`

---

## Fused TopK-MoE Kernel (CUDA only)

```cuda
// reference_material/llama.cpp/ggml/src/ggml-cuda/topk-moe.cu
// Fuses: softmax → iterative argmax → weight normalization
// One warp (32 threads) processes one token
// Iterative argmax: find max, mask to -inf, repeat k times
// Warp-level reductions for max finding
```

Not available on Metal — Metal uses `ggml_argsort_top_k` instead.

---

## ICB Barrier Approach

`ggml-metal-common.cpp:280-359`:
- Track read/write buffer ranges per kernel
- Only insert `memoryBarrierWithScope` when RAW/WAR/WAW conflict detected
- Look ahead N_FORWARD=8 nodes to reorder and maximize concurrency
- Independent kernels pipeline freely, near-zero per-kernel overhead

---

## Quantized Matmul Kernel Design

### Key techniques (Metal Q4_0 GEMV)
- **ushort reads**: 2 bytes → 4 nibbles at once (vs 1 byte → 2 nibbles)
- **SIMD reduction**: `simd_sum` hardware shuffle (vs threadgroup memory barriers)
- **Multi-row**: amortizes activation reads across nr0 output rows
- **Stride-4 block interleaving**: 4 adjacent threads read consecutive blocks (coalesced)
- **The 1/256 trick**: leaves nibble in high byte, multiplies by 1/256 instead of explicit shift

### Activation quantization (for GEMM path)
```cpp
quantize_row_q8_K(activations, quantized_activations, n);
// Then: q4_K weights × q8_K activations
```
Reduces memory bandwidth 4x for matmul. Not used in decode (batch=1 GEMV) path.
