# llama.cpp MoE Implementation — Insights for tinygrad

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

## Weighted Sum: Explicit Adds, Not Reduction

```cpp
// llama-graph.cpp:1346-1350
ggml_tensor * moe_out = cur_experts[0];
for (uint32_t i = 1; i < n_expert_used; ++i) {
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
}
```

For k=6: `e0 + e1 + e2 + e3 + e4 + e5` — five add operations.
Each is a highly optimized elementwise kernel. No reduction dimension needed.

Note from the code comment:
> explicitly use hparams.n_expert_used instead of n_expert_used
> to avoid potentially a large number of add nodes during warmup
> ref: https://github.com/ggml-org/llama.cpp/pull/14753

## Shared Expert: Completely Separate Pipeline

The shared expert (DeepSeek/GLM) runs as a fully independent computation:
```
shared_out = down_shexp(silu(gate_shexp(x)) * up_shexp(x))
```
This is never fused with the MoE experts. It runs as separate matmuls.

## What This Means for tinygrad's `r_2048_16_1408_6_176`

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

GROUPTOP=16 was chosen for Part 2 (shared expert), leaving Part 1 fully serial.
Result: 4.5ms at 31 GB/s (should be ~100 GB/s).

### Why llama.cpp Is Faster Here
1. **Down projection is its own kernel** — gets full GEMV parallelism (32×nsg threads for the reduction)
2. **Each expert dispatches separately** — no serial expert loop
3. **Shared expert is separate** — doesn't constrain threadgroup sizing of MoE
4. **Explicit adds** — k-1 tiny elementwise kernels instead of one sum reduction

### Actionable Options for tinygrad

**Option A: Break fusion with `.contiguous()`**
```python
expert_out = self.ffn_down_exps(sel, gated).float()   # [B, T, k, dim]
expert_out = expert_out.contiguous()                    # <-- force separate kernel
out = (expert_out * probs.unsqueeze(-1)).sum(axis=2)
```
Already exists as `MOE_ADDS=2`. This lets the down projection kernel get its own threadgroup sizing.

**Option B: Explicit adds (llama.cpp style)**
```python
expert_out = self.ffn_down_exps(sel, gated).float()
weighted = expert_out * probs.unsqueeze(-1)
# k explicit adds instead of .sum(axis=2)
out = weighted[:,:,0,:] + weighted[:,:,1,:] + ... + weighted[:,:,k-1,:]
```
Eliminates the reduction dimension entirely.

**Option C: Separate shared expert**
```python
# Don't let shared expert fuse with MoE
moe_out = (expert_out * probs.unsqueeze(-1)).sum(axis=2) * scale
moe_out = moe_out.contiguous()  # break fusion
shared_out = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).float().silu() * self.ffn_up_shexp(h_norm).float()).float()
return h + (moe_out + shared_out).cast(h.dtype)
```

**Option D: Custom `mul_mat_id` kernel**
Write a Metal kernel that dispatches per-expert threadgroups like llama.cpp.
Most work but highest ceiling.

## CUDA `mmid.cu` Preprocessing

For CUDA, llama.cpp runs a preprocessing kernel (`mm_ids_helper`) that:
1. Groups tokens by expert — builds compact sorted representation
2. Computes `expert_bounds[i:i+1]` — where each expert's tokens start/end
3. Then dispatches standard GEMM per expert with contiguous token blocks

This avoids scattered access — each expert's matmul operates on contiguous memory.
Template-specialized for common values: n_expert_used = 2, 4, 6, 8, 16, 32.

Source: `reference_material/llama.cpp/ggml/src/ggml-cuda/mmid.cu:22-116`

## Fused TopK-MoE Kernel (CUDA only)

```cuda
// reference_material/llama.cpp/ggml/src/ggml-cuda/topk-moe.cu
// Fuses: softmax → iterative argmax → weight normalization
// One warp (32 threads) processes one token
// Iterative argmax: find max, mask to -inf, repeat k times
// Warp-level reductions for max finding
```

Not available on Metal — Metal uses `ggml_argsort_top_k` instead.
