# Problematic Kernels — deepseek-v2-lite-Q4_0

Model: deepseek-v2-lite-Q4_0, 1.9 GB params, 728 kernels/token, 27 MoE blocks + 1 dense block
Baseline: **28.8 tok/s** (34.8ms/tok). Target: **57+ tok/s** (2x).
Theoretical minimum at 100 GB/s: 19ms (52 tok/s).

## Token Time Budget

| Category | ms/tok | % | Kernels/tok |
|----------|--------|---|-------------|
| Expert gate+up (fused) | 16.9 | 45% | 27 |
| Expert down | 7.2 | 19% | 27 |
| Shared expert + weighted sum (fused) | ~6.0 | 16% | 27 |
| Attention (q_a, kv_a, q_b, attn_out) | ~3.0 | 8% | ~108 |
| Small reductions (norms, softmax, topk) | ~1.0 | 3% | ~263 |
| Elementwise | ~0.7 | 2% | ~276 |
| **Total** | **~34.8** | **100%** | **728** |

Expert kernels = **80% of token time**. Everything else is noise.

---

## Kernel #1: Expert Gate+Up Fused (THE BOTTLENECK)

**Name**: `r_880_32_3_64_16_64_16`
**Calls/token**: 27 (one per MoE block)
**Time**: 626us each = **16.9ms/token (45%)**
**Bandwidth**: 33 GB/s (33% of Metal peak 100 GB/s)
**Type**: MULTIREDUCE (two separate reduction groups fused)

### What it computes

```
gated = gate_experts(sel, h_norm).silu() * up_experts(sel, h_norm)
```

This is gate + silu + up + element-multiply, ALL fused into one kernel. Two Q4_0 packed-dot
matmuls (gate and up) each with reduce dims [64, 16], plus silu activation and multiply.

### Optimizer opts applied

```
UPCAST(0, 3)   — 3 output elements per thread
UNROLL(3, 0)   — fully unroll inner reduce dim 16 (but ONLY for the 2nd reduce = up expert)
LOCAL(0, 32)   — 32 threads per threadgroup
```

NO GROUP. NO GROUPTOP. Each thread does the full reduction serially.

### Grid dimensions

- 880 threadgroups (= 6 experts * 1408 outputs / 3 upcast / 32 local... wait, 6*1408/3 = 2816, /32 = 88, *10 = 880 — the 10 comes from gidx0/88 for expert routing index)
- 32 threads per threadgroup
- Total: 28,160 threads

### Why it's slow

**1. Multireduce prevents GROUP**: The kernel has TWO reduction groups (gate and up). GROUP(0, N)
only splits the FIRST reduction's outer dim. The second reduction is unaffected. Both reductions
run serially, with GROUP sync overhead paid but only one reduction benefits. This is why all
previous GROUP experiments regressed.

**2. Asymmetric unrolling**: The first reduce (gate) has an inner loop `for Ridx0_1 in range(16)`.
The second reduce (up) is fully unrolled — all 16 packed bytes loaded at once with `half4`
vectorized reads. The first reduce reads ONE byte at a time. This means the gate half of the
kernel is ~2x slower than the up half.

**3. No cooperative reduction**: 32 threads each do 64*16 = 1024 iterations serially for EACH
of the two reductions. No threadgroup shared memory, no partial sum accumulation.

**4. Scattered byte reads across threads**: Thread `lidx0=0` reads row 0 of the weight matrix.
Thread `lidx0=1` reads row 3456 bytes away (= 3 upcasted rows * 64 blocks * 18 bytes/block).
With 32 threads, reads span 32 * 3456 = 110,592 bytes. Metal L1 cache is 32KB. Each thread's
reads evict other threads' data. Cache line utilization is terrible.

### Metal source (first reduce loop — the slow one)

```metal
for (int Ridx0_0 = 0; Ridx0_0 < 64; Ridx0_0++) {
    int alu6 = (Ridx0_0*18);
    int alu7 = (alu1+alu6);
    // Read scale bytes (2 bytes per block, 3 output rows)
    unsigned char val1 = (*(data1+(alu7+1)));        // scale_hi row 0
    unsigned char val2 = (*(data1+(alu7+1152)));      // scale_lo row 1
    unsigned char val3 = (*(data1+(alu7+1153)));      // scale_hi row 1
    unsigned char val4 = (*(data1+(alu7+2304)));      // scale_lo row 2
    unsigned char val5 = (*(data1+(alu7+2305)));      // scale_hi row 2
    unsigned char val6 = (*(data1+alu7));             // scale_lo row 0

    for (int Ridx0_1 = 0; Ridx0_1 < 16; Ridx0_1++) {
        // ONE packed byte per row per iteration (not vectorized!)
        unsigned char val7 = (*(data1+(alu8+2)));       // packed[j] row 0
        unsigned char val8 = (*(data1+(alu8+1154)));     // packed[j] row 1
        unsigned char val9 = (*(data1+(alu8+2306)));     // packed[j] row 2

        // Activation reads (half precision, contiguous)
        half val10 = (*(data4+(alu9+16)));   // x_hi
        half val11 = (*(data4+alu9));        // x_lo

        // scale * ((nibble_lo - 8) * x_lo + (nibble_hi - 8) * x_hi)
        acc0[0] += scale * (((val7 & 0xF) - 8) * x_lo + ((val7 >> 4) - 8) * x_hi);
        acc0[1] += scale * (((val8 & 0xF) - 8) * x_lo + ((val8 >> 4) - 8) * x_hi);
        acc0[2] += scale * (((val9 & 0xF) - 8) * x_lo + ((val9 >> 4) - 8) * x_hi);
    }
}
```

### Why MV heuristic doesn't match

MV (heuristic.py:72-73) requires: `mulop.src[act_i].op is Ops.INDEX`

The outermost MUL in this kernel is `MUL(scale_bitcast_chain, ADD(MUL(lo-8, x), MUL(hi-8, x)))`.
Neither operand is an INDEX — one is a bitcast chain from INDEX, the other is ADD of MULs.
The activation INDEXes are buried TWO levels deep inside the ADD.

### Why GROUPTOP doesn't fire

GROUPTOP (heuristic.py:92) requires: `prod(output_shape[i] for i in upcastable_dims) <= 2048`

Output for this kernel: 6 experts * 1408 outputs = 8448 > 2048. Threshold blocks it.

### Ideas to speed up

**Idea A: Break the fusion (split gate and up)**
Put `.contiguous()` between gate and up so they become separate kernels. Each would be a
single-reduce kernel eligible for better optimization.
- **Tested**: 28.8 tok/s — NO change. The individual kernels still don't get MV because the
  packed-dot pattern doesn't match `MUL(INDEX, INDEX)`.
- **Why**: tinygrad's `found_contiguous` optimization elides `CONTIGUOUS(RESHAPE(...))`, so the
  scheduler may re-fuse them anyway. Even when they do split, each single-reduce kernel gets
  the same UPCAST+UNROLL+LOCAL opts without GROUP.

**Idea B: Extend MV heuristic to walk through dequant chains**
Instead of requiring `mulop.src[act_i].op is Ops.INDEX`, walk the backward_slice to find INDEX.
- **Tested**: 19 tok/s — REGRESSION. The extended MV fired on multireduce kernels too, applying
  GROUP to only one of two reductions. Sync overhead exceeds benefit.
- **Fix needed**: Only apply to single-reduce kernels. But gate+up IS multireduce.

**Idea C: Raise GROUPTOP threshold to 16384**
- **Tested**: 12.8 tok/s — SEVERE regression. GROUPTOP applied to ALL kernels including attention
  that was already well-optimized with MV.

**Idea D: BEAM search**
Let BEAM find optimal opts for each kernel. BEAM=2 would try GROUP, different UPCAST/LOCAL combos.
- **Status**: Not fully tested. BEAM is slow (minutes per kernel). Need BEAM_KERNEL_NAME targeting.

**Idea E: Different tensor expression**
Restructure the packed-dot computation so the outermost MUL IS `MUL(INDEX, INDEX)`:
```python
# Instead of: scale * (lo * x_lo + hi * x_hi)
# Try: scale_expanded * lo * x_lo + scale_expanded * hi * x_hi
# Where scale_expanded is pre-broadcast to match packed dims
```
This would make each term a `MUL(weight_INDEX, activation_INDEX)` matchable by MV.
But it doubles the scale multiplications and may not change the fused kernel structure.

**Idea F: Pre-dequant to int8 at load time**
At model load: `int8_lo = (packed & 0xF) - 8; int8_hi = (packed >> 4) - 8`
Store as two separate int8 tensors. Then the kernel is `scale * (int8_w * x).sum()`.
This removes bitwise ops from the hot loop. The `MUL(int8_w_INDEX, x_INDEX)` might match MV.
Data size: same as Q4_0 (16 bytes packed → 16 int8_lo + 16 int8_hi = 32 bytes, but scale
shared = 2 bytes → 34 vs 18 = 1.9x more data). Probably not worth the bandwidth increase.

**Idea G: Completely different approach — dequant to fp16 cache, use regular matmul**
Pre-dequant expert weights to fp16 at model load. Regular matmul gets MV at 80 GB/s.
- Math: fp16 = 3.56x more bandwidth. At 80 GB/s vs 33 GB/s effective: 80/3.56 = 22.5 GB/s
  effective vs 33 GB/s Q4_0. **Worse**. Dead end.

**Idea H: Fix the root cause — teach optimizer about fused dequant matmul**
Add a new heuristic path in hand_coded_optimizations specifically for Q4_0 packed-dot patterns.
Detect: kernel has AND + SHR ops, single reduce, large output. Apply GROUP + LOCAL + UPCAST
similar to MV but tuned for the Q4_0 access pattern.
- **The multireduce problem**: gate+up fused kernel has TWO reduce groups. Need to either:
  (a) only optimize single-reduce kernels (down + shared expert), or
  (b) force the scheduler to NOT fuse gate+up (requires scheduler changes, not just heuristics)

---

## Kernel #2: Expert Down Projection

**Name**: `r_15_32_4_16_4_44_16` (profile name: `r_2_32_3_16_4_44_16`)
**Calls/token**: 27
**Time**: 266us each = **7.2ms/token (19%)**
**Bandwidth**: 39 GB/s
**Type**: SINGLE REDUCE (one reduction group)

### What it computes

```
expert_out = down_experts(sel, gated)  # (6, 1408) -> (6, 2048)
```

Single Q4_0 packed-dot matmul. Reduce dims: [44, 16] (44 blocks of 32 elements = 1408 input).

### Optimizer opts applied

```
UPCAST(1, 4)      — 4 output elements per thread
UNROLL(1, 0)       — fully unroll inner reduce dim 16
LOCAL(0, 4)        — 4 threads in x
LOCAL(1, 16)       — 16 threads in y
```

NO GROUP. Each thread does 44 outer iterations serially. Inner loop fully unrolled (reads all 18
bytes per block + vectorized half4 activation reads).

### Grid dimensions

- 32 gidx0 (2048 outputs / 4 upcast / 16 local_y) × 15 gidx1 (some combo of 6 experts)
- 4 × 16 = 64 threads per threadgroup
- Total: 480 threadgroups × 64 = 30,720 threads

### Why it's slow

**1. No cooperative reduction**: 64 threads, each doing 44 serial iterations. With GROUP(0, 4),
each thread would only do 11 iterations, then sync. 4x less serial work per thread.

**2. This IS a single-reduce kernel** — GROUP can work here without the multireduce problem.

### Metal source (outer loop)

```metal
for (int Ridx0_0 = 0; Ridx0_0 < 44; Ridx0_0++) {
    // Read 18 bytes of Q4_0 block per output row (x4 upcast)
    // = 72 byte reads per iteration, scattered across 4 output rows
    // stride between rows: 792 bytes (= 44 blocks * 18 bytes)
    unsigned char val0..val70;  // 71 byte reads!

    // Read activation: vectorized half4 (8 reads of 8 bytes = 64 bytes)
    half4 val72..val79;

    // 4 dequant-multiply-accumulate per byte
    acc0[0..3] += scale * ((nibble & 0xF) - 8) * x_lo + ((nibble >> 4) - 8) * x_hi
}
```

### Ideas to speed up

**Idea A: Apply GROUP via heuristic change (MOST PROMISING)**
Add targeted GROUP for single-reduce kernels with AND/SHR in backward_slice:
```python
# After MV fails, before GROUPTOP, add:
if has_bitwise_ops and single_reduce and output_product > 2048:
    try: k.apply_opt(Opt(OptOps.GROUP, 0, 4))  # 44/4 = 11 iters per thread
```
Expected: 44 → 11 iterations per thread + sync. If sync is ~5us and throughput doubles:
266us → ~140us → saves 3.4ms/token.

**Idea B: BEAM=2 targeted at this kernel**
```bash
BEAM=2 BEAM_KERNEL_NAME=r_15_32 python tinygrad/apps/llm.py ...
```

**Idea C: Increase UPCAST from 4 to 8**
More outputs per thread = more work amortization. 2048/8 = 256 output groups.
May not help — more registers needed, more byte reads per iteration.

---

## Kernel #3: Shared Expert + Weighted Sum Fused

**Name**: `r_5_32_2_16_4_6_88_16` (profile name varies: `r_22_32_4_64_16_64_16n*`, `r_288_2_16_4_4_8`)
**Calls/token**: 27 (but some blocks have different shapes)
**Time**: ~210-845us each = **~6ms/token (16%)**
**Bandwidth**: 26-28 GB/s
**Type**: MULTIREDUCE

### What it computes

```
# Weighted sum of expert outputs
out = (expert_out.contiguous() * probs.unsqueeze(-1)).sum(axis=2)
# PLUS shared expert (fused into same kernel):
out += down_shexp(gate_shexp(h_norm).silu() * up_shexp(h_norm))
```

The contiguous() call breaks fusion between expert_out and weighted_sum, BUT the weighted sum
fuses with the shared expert. So this kernel does:
1. Sum(expert_out * routing_probs, dim=experts)  — reduce over 6 experts
2. Shared expert gate+silu+up+down — Q4_0 packed-dot with reduce [88, 16]

### Why it's slow

Same as Kernel #1: multireduce prevents GROUP. The weighted sum (reduce over 6) is tiny but
the shared expert matmul (reduce 88*16 = 1408 elements) dominates. The two reductions can't
both benefit from GROUP.

### Ideas to speed up

**Idea A: Break shared expert out of weighted sum**
Add `.contiguous()` or `.realize()` on `out` before adding shared expert:
```python
out = (expert_out.contiguous() * probs.unsqueeze(-1)).sum(axis=2) * self.expert_weights_scale
out = out.contiguous()  # force break
if hasattr(self, 'ffn_gate_shexp'):
    out = out + self.ffn_down_shexp(...)
```
This would make the shared expert a separate single-reduce kernel eligible for GROUP/MV.

**Idea B: Dequant cache for shared expert only**
The shared expert has fixed weights (not selected per-token). Pre-dequant to fp16 at load time.
Shared expert size: 2 * 2816 * 2048 * 2 = 23 MB fp16. At 80 GB/s MV: 0.29ms vs current ~0.8ms.
Only 27 × 0.5ms = 13.5ms saved. BUT this adds 23 MB to param memory (from 1.9 GB to 1.92 GB).

---

## Kernel #4: Attention Q_B / KV_A Projections

**Name**: `r_10_128_16_4_4_32` and variants
**Calls/token**: ~108 total (4 per block × 27 blocks)
**Time**: ~36us each = **~3.9ms/token (8%)**
**Bandwidth**: ~60-80 GB/s (GOOD — MV fires correctly)

### What it computes

```
q = attn_q_b(attn_q_a_norm(attn_q_a(x_norm)))   # Q4_0 packed-dot
kv = attn_kv_a_mqa(x_norm)                        # Q4_0 packed-dot
```

### Why it's (relatively) fast

MV heuristic MATCHES these kernels because they use the fp16 dequant cache path (`QL_CACHE_ATTN_*`),
not the packed-dot path. The dequant cache produces `MUL(INDEX, INDEX)` which MV detects.

Opts: `GROUP(0, 16) + LOCAL(1, 4) + UPCAST(1, 4)` — cooperative reduction with threadgroup memory.

```metal
threadgroup float temp0[256];  // shared memory for partial sums
```

### Ideas

These kernels are already well-optimized. Not the bottleneck. MV_ROWS_PER_THREAD=1 gave +1 tok/s
by reducing scatter in these kernels (29.2 vs 28.2 tok/s).

---

## Kernel #5: Embedding / Output Head

**Name**: `r_6400_16_4_4_128` (embedding), `E_102400_2_8_16_2_4` / `E_409600_16_8_4` (dequant)
**Calls/token**: 1-4 each
**Time**: 3.4ms + 6.5ms + 6.4ms = **~16ms in warmup only** (amortized to ~0 after JIT)
**Bandwidth**: 98-148 GB/s (GOOD)

These only run during warmup (pre-JIT). After JIT captures, they're in the cached graph.
Not a steady-state bottleneck.

---

## Summary: Where the 2x Needs to Come From

Current 34.8ms breakdown:
- Expert gate+up: 16.9ms (45%) — multireduce, 33 GB/s, NO GROUP
- Expert down: 7.2ms (19%) — single-reduce, 39 GB/s, NO GROUP
- Shared expert: 6.0ms (16%) — multireduce, 28 GB/s, NO GROUP
- Everything else: 4.7ms (13%) — already 60-80 GB/s

Target 17.4ms (2x). Need to cut 17.4ms.

### Attack Plan (ordered by expected impact)

1. **Expert down: apply GROUP** (single-reduce, can work)
   - Expected: 39 → 70 GB/s = 266 → 148us = saves **3.2ms** (9%)

2. **Break shared expert fusion, apply GROUP/MV**
   - Split weighted-sum and shared expert with `.contiguous()`
   - Shared expert becomes single-reduce, eligible for GROUP
   - Expected: 28 → 60 GB/s on shared = saves **~3ms** (8%)

3. **Expert gate+up: force unfuse at scheduler level**
   - If gate and up become separate single-reduce kernels, each gets GROUP
   - Expected: 33 → 60 GB/s = 626 → ~345us = saves **7.6ms** (22%)
   - Requires: either scheduler change or `.contiguous()` that actually sticks

4. **BEAM search on remaining kernels**
   - Targeted BEAM=4 on expert kernels only
   - May find better UPCAST/LOCAL/UNROLL combos

5. **Pre-dequant shared expert to fp16** (fallback if GROUP doesn't help)
   - Shared expert at 80 GB/s MV = saves ~3ms

### Theoretical Best Case

If all expert kernels reach 70 GB/s (70% of Metal peak):
- Gate+up: 19.4MB / 70 GB/s = 0.28ms × 27 = 7.5ms (was 16.9ms)
- Down: 9.7MB / 70 GB/s = 0.14ms × 27 = 3.8ms (was 7.2ms)
- Shared: 4.8MB / 70 GB/s = 0.07ms × 27 = 1.9ms (was 6.0ms)
- Rest: 4.7ms
- **Total: 17.9ms = 56 tok/s** (exactly 2x!)

The math works. The question is: can we actually get GROUP to fire and help?

## Key Files

- `tinygrad/codegen/opt/heuristic.py` — MV heuristic (line 63-89), GROUPTOP (line 91-97)
- `tinygrad/apps/mla.py:139-146` — Expert MoE computation, shared expert
- `tinygrad/apps/quantized.py:126-137` — Q4_0 packed-dot expert implementation
- `tinygrad/engine/schedule.py` — Controls kernel fusion (found_contiguous elision)

## Previous Experiments Log

| Experiment | tok/s | vs baseline | Notes |
|-----------|-------|-------------|-------|
| Baseline | 28.8 | — | |
| GROUPTOP_MAX=16384 | 12.8 | -55% | Applied to ALL kernels |
| MV backward_slice walk | 19.0 | -34% | Caught multireduce |
| Targeted packed-dot GROUP | 18.8 | -35% | Still caught multireduce |
| QL_CUSTOM=1 (UOp kernel) | 12.2 | -58% | Custom UOp path is slow |
| JIT_BATCH_SIZE=728 | 28.4 | -1% | ICB transitions not bottleneck |
| MV_ROWS_PER_THREAD=1 | 29.2 | +1% | Small win on attention kernels |
| Split gate+up .contiguous() | 28.8 | 0% | Scheduler re-fuses |
| Remove expert .float() | 28.8 | 0% | Already fp16 in packed-dot |
| Remove shexp .float() | 28.9 | 0% | Already fp16 in packed-dot |
