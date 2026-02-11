# Pre-Separation Pattern: Enabling Gather→Reduce Fusion

## Summary
Pre-separating Q4_0 blocks into separate scale and packed tensors enables the tinygrad
scheduler to fuse expert gathers INTO downstream dequant+matmul reduce kernels.

**Result**: 21.3 → 26.7 tok/s (+25%) on GLM-4.7-Flash Q4_0

## How We Found It

### Step 1: Metal Source Analysis (DEBUG=5)
Examined expert Q4_0 matmul kernel `r_5120_16_4_4_4_16` using `DEBUG=5`.
Discovered **stride-18 byte access pattern**: GROUP threads access different Q4_0 blocks
with stride 18 bytes (2 scale + 16 packed), causing ~5.6% cache line utilization.

### Step 2: Pre-Separation Hypothesis
Q4_0 blocks are 18 bytes: `[d: fp16 (2 bytes)][packed: 16 × uint8]`.
Hypothesis: separate into contiguous scale tensor (stride-2) and packed tensor (stride-16)
for better memory coalescing.

### Step 3: Implementation
```python
# Before: combined blocks, sliced after gather
eb_4d = expert_blocks.reshape(E, O, bpr, 18)
gathered = eb_4d[sel_flat]        # ONE gather
scale = gathered[:, :, :, :2]    # consumer 1 → prevents fusion
packed = gathered[:, :, :, 2:]   # consumer 2 → prevents fusion

# After: pre-separated, gather each independently
scale_all = eb_4d[:, :, :, :2].contiguous().realize()   # (E, O, bpr, 1)
packed_all = eb_4d[:, :, :, 2:].contiguous().realize()   # (E, O, bpr, 16)
scale = scale_all[sel_flat]   # single consumer → scheduler fuses into reduce
packed = packed_all[sel_flat]  # single consumer → scheduler fuses into reduce
```

### Step 4: Surprise — It's NOT About Coalescing
Initial theory was stride-18 → stride-2/16 coalescing improvement.
**Actual mechanism**: gather→reduce FUSION.

**Proof**: Adding `.contiguous()` after the gather (which prevents fusion but keeps
the coalescing benefit) dropped back to 21.3 tok/s. The coalescing alone provides
essentially 0% benefit. ALL the speedup comes from fusion.

### Step 5: Understanding Why
The tinygrad scheduler can fuse an elementwise operation (gather/index) INTO a downstream
reduce kernel when the gathered tensor has exactly ONE consumer. With combined 18-byte blocks:

```
gathered_blocks[sel_flat]  →  [:, :, :, :2]  (consumer 1: scale)
                           →  [:, :, :, 2:]  (consumer 2: packed)
```

Two consumers = scheduler must materialize the gather as a separate kernel.

Pre-separated:
```
scale_all[sel_flat]  → feeds directly into dequant+matmul (ONE consumer)
packed_all[sel_flat] → feeds directly into dequant+matmul (ONE consumer)
```

One consumer each = scheduler fuses each gather into its downstream reduce.

## Where Else This Applies

### QuantizedExpertWeights Fallback Path (Q4_K, Q5_K, Q6_K)
File: `quantized.py:144-148`

The fallback path gathers combined blocks then passes to `dequant_fn`, which slices into
4+ parts. Same anti-pattern:

**Q4_K (144 bytes/block)**:
- `blocks[:,0:2]` → d (scale)
- `blocks[:,2:4]` → dmin
- `blocks[:,4:16]` → s (super-block scales)
- `blocks[:,16:144]` → q (packed weights)

**Q6_K (210 bytes/block)**:
- `blocks[:,:128]` → xl (low quantized)
- `blocks[:,128:192]` → xh (high quantized)
- `blocks[:,192:208]` → scales
- `blocks[:,-2:]` → d (scale)

Each of these creates multiple consumers of the gathered tensor.
Pre-separating would enable gather→reduce fusion for these types too.

**Currently not relevant for GLM Q4_0** (uses ggml_type=2 which is already fixed).
Would help: GLM Q4_K_M, GLM Q6_K, DeepSeek-V2-Lite Q4_K_M.

### NOT Applicable
- `split_kv_b` in mla.py — done at load time, no gather
- `q.split(nope, rope)` in attention — split of linear output, not a gather
- `gate_up.split()` in MoE — split of matmul output, not a gather
- Dense QuantizedLinear — no gather (direct matmul with weight)

## Files Modified
- `tinygrad/apps/quantized.py` — `_ensure_q4_0_separated()` for both QuantizedLinear and QuantizedExpertWeights
- `tinygrad/apps/mla.py` — `merge_gate_up_experts/shared_expert` init `_q4_0_scale=None, _q4_0_packed=None`
