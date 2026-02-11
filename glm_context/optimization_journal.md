# GLM Q4_0 Optimization Journal

Date baseline: 2026-02-11
Scope: decode-path optimization with `bench_block.py` only (no full model run), sequential benchmarks only.

## Measurement Discipline
- Benchmark command: `.venv2/bin/python bench_block.py <count> --model "glm-4.7:flash-unsloth-Q4_0"`
- Device: `METAL`
- Stability check: use `count=30` before accepting/rejecting a change.
- Primary metric: `full block with JIT` median (closest proxy to decode tok/s path).
- Secondary metric: `feed_forward only` median.

### Harness Pinning
- File: `bench_block.py`
- Change: pinned CLI `--model` to `glm-4.7:flash-unsloth-Q4_0` via `choices=[DEFAULT_BENCH_MODEL]`.
- Why:
  - Prevents accidental benchmark drift to non-target models/shapes.
  - Keeps optimization loop strictly on the decode path we care about.

## Confirmed Wins

### 1) Representative benchmark harness (GGUF-driven config)
- File: `bench_block.py`
- Change: stop using stale hardcoded dims; pull shapes from GGUF metadata.
- Real decode-like config now used:
  - `dim=2048`
  - `moe_hidden=1536`
  - `experts=64 x 4`
  - `expert_first=True` (from tensor metadata)
- Why this matters:
  - Prevents optimizing the wrong shapes.
  - Keeps schedule/kernel behavior aligned with the real model architecture.

### 2) One runtime path for Q4_0 experts
- File: `tinygrad/apps/quantized.py`
- Current behavior:
  - Q4_0 experts always use pre-separated DSL path (`scale` + `packed`), gather+dequant+reduce in one fused expression.
  - No optional runtime branch in expert path.
- Why this matters:
  - Avoids branchy performance variability.
  - Keeps optimization target stable.

### 3) Core heuristic tuning for MoE Q4_0 reduce kernels
- File: `tinygrad/codegen/opt/heuristic.py`
- Change in MoE dequant branch:
  - Prefer `LOCAL` order: `[8, 4, 16, 32]`
  - Prefer `GROUP` order: `[8, 16, 4]`
- Hot kernels affected:
  - `r_4_768_4_16_64` / `r_4_512_4_16_48` patterns
  - tuned schedules show `LOCAL(axis=1,arg=8)` + `GROUP(axis=1,arg=8)` and better decode-time behavior

#### A/B evidence (count=30)
- Old heuristic (`LOCAL [4,8,16,32]`, `GROUP [16,8,4]`):
  - full block median: `0.98 ms`
  - feed_forward median: `20.96 ms`
- Tuned heuristic (`LOCAL [8,4,16,32]`, `GROUP [8,16,4]`):
  - full block median: `0.92 ms`
  - feed_forward median: `19.99 ms`
- Net:
  - full-block median improved by ~`6%`
  - feed_forward median improved by ~`4.6%`

## Rejected Changes (Important)

### Forced `h_norm.contiguous().realize()` before MoE
- File tested: `tinygrad/apps/mla.py`
- Result:
  - Kernel count increased (`30 -> 34` in warmup path).
  - Full block median regressed to ~`2.29 ms`.
- Interpretation:
  - Forcing realization broke useful fusion/ICB behavior and added launch overhead.
  - Keep this out of decode path.

## Current Working State
- Keep the heuristic tuning above in `tinygrad/codegen/opt/heuristic.py`.
- Continue iterating only with `bench_block.py` and record every accepted/rejected hypothesis here.

## Kernel Anatomy (DEBUG=2)

### Feed-forward steady-state kernel set (15 kernels)
Source run: `DEBUG=2 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`

From cache-hit feed_forward pass (`scheduled 15 kernels in ~3.5ms`), dominant kernels are:
- `r_4_384_8_8_64_2n1` ~`329 us`, ~`279 GB/s`  
  MoE gate/up expert reduce (Q4_0 nibble unpack + multiply + reduce).
- `r_4_256_8_8_48_2` ~`176-183 us`, ~`254-264 GB/s`  
  MoE down expert reduce.
- `r_192_16_4_4_4_16n1` ~`83 us`  
  Secondary quantized reduce path.
- `r_128_16_4_4_3_16n1` usually ~`51 us` but occasionally spikes (scheduler/launch jitter).

Small non-dominant kernels:
- Router/topk/norm/silu/probability kernels (`r_16_128`, `r_64_16_4`, `E_16_32_3`, `r_4`, etc.) are each mostly single-digit to low tens of microseconds.
- Gather kernels (`E_1024_32_4_3`, `E_2048_32_4_3`, `E_4`) are present but not dominant.

Interpretation:
- Decode feed-forward is dominated by two expert reduce kernels and they are bandwidth-heavy.
- Time reduction comes from improving these kernels' launch shape and memory behavior, not from micro-optimizing router/topk math.

## BEAM Experiments (`BEAM_SKIP_MS=1`)

### BEAM perf check
Run: `BEAM=4 BEAM_SKIP_MS=1 .venv2/bin/python bench_block.py 10 --model "glm-4.7:flash-unsloth-Q4_0"`
- full block median: `1.32 ms`
- feed_forward median: `20.27 ms`

Compared with tuned hand heuristic (`count=30` runs around `0.85-0.92 ms` full block median), this BEAM setting regresses decode-path latency.

### BEAM debug pass observations
Runs attempted:
- `BEAM=4 BEAM_SKIP_MS=1 DEBUG=6 ...` (very large trace, stopped after enough evidence)
- `BEAM=4 BEAM_SKIP_MS=1 DEBUG=4 ...` (also largely BEAM probe output before useful decode-kernel logs)

Observed in BEAM traces:
- Many probe kernels (`kernel void test(...)`) and massive option exploration output.
- Frequent opt patterns in traced candidates were heavily local-oriented, especially:
  - `(Opt(op=OptOps.LOCAL, axis=0, arg=32),)` (dominant)
  - combinations with additional `LOCAL` splits and occasional `GROUP/GROUPTOP`.

Interpretation:
- With current objective + skip threshold, BEAM explores/chooses many launch shapes that are not beating the targeted hand-tuned MoE decode heuristic on this workload.
- Keep BEAM for occasional discovery, but treat hand-tuned decode heuristic as current best for this exact path.
- Important: `BEAM_SKIP_MS` is not usable for this workload because many critical kernels are already sub-millisecond; beam exits early once a candidate is below the threshold, which can freeze suboptimal shapes.

### GFLOPS-level comparison: bad BEAM shape vs tuned shape
Observed pairwise differences from trace snippets:

- BEAM-selected hot gate/up kernel:
  - `r_4_3072_16_64`: ~`652 us`, `217 GFLOPS`, `195 GB/s`
- Tuned hot gate/up kernel:
  - `r_4_384_8_8_64_2n1`: ~`329 us`, `423 GFLOPS`, `279 GB/s`
- Delta:
  - ~`1.98x` slower
  - ~`49%` lower GFLOPS
  - ~`30%` lower bandwidth

- BEAM-selected hot down kernel:
  - `r_4_2048_8_6_16`: ~`621 us`, `92 GFLOPS`, `57 GB/s`
- Tuned hot down kernel:
  - `r_4_256_8_8_48_2`: ~`177 us`, `322 GFLOPS`, `262 GB/s`
- Delta:
  - ~`3.5x` slower
  - ~`71%` lower GFLOPS
  - ~`78%` lower bandwidth

Additional red flags seen in the BEAM trace:
- Large timing variance and stalls in tiny kernels (`r_4`, `r_16_128`) during some iterations.
- Kernel naming/shape indicates BEAM picked much coarser tiling for expert kernels (large monolithic output loops), which underutilizes memory bandwidth on this decode shape.

## Variance Note
- `count=10` is noisy and can mislead (examples seen from ~`0.98 ms` to ~`1.9 ms` depending on warm state and outliers).
- `count=30` gives a more stable signal for accept/reject decisions.

## Trial: Explicit `k-1` Expert Adds vs `sum(axis=2)` (Rejected)

### Hypothesis
- Replacing MoE combine reduction with explicit adds (llama.cpp style) might avoid a bad reduce launch shape and reduce jitter.

### Change tested
- File: `tinygrad/apps/mla.py`
- Tested form:
  - `weighted = expert_out * probs.unsqueeze(-1)`
  - `out = weighted[:,:,0,:] + weighted[:,:,1,:] + ...` (loop over `K`)
- Reverted after benchmark.

### Sequential benchmark result (`BEAM=0 .venv2/bin/python bench_block.py 20`)
- Baseline (`sum(axis=2)`):
  - full block median: `1.81 ms`
  - feed_forward median: `18.85 ms`
- Explicit-add trial:
  - full block median: `1.82 ms`
  - feed_forward median: `19.58 ms`
- Net:
  - full block essentially flat
  - feed_forward regressed by ~`0.73 ms` (~`3.9%`)

### Kernel-level A/B (`DEBUG=2`)
- Hot kernels stayed the same class and speed range:
  - `r_4_384_8_8_64_2n1`: ~`419-425 GFLOPS`, ~`276-280 GB/s`
  - `r_4_256_8_8_48_2`: ~`315-321 GFLOPS`, ~`257-262 GB/s`
- Combine kernel changed from:
  - baseline: `r_16_32_4_4` (`['__truediv__','sum','__mul__']`)
  - trial: `E_16_32_4n1` (`['__truediv__','__add__','__mul__']`)
- But that kernel is only ~`8-11 us`, so it is not the bottleneck.
- Cache/uops increased (`13dd905d`/`26285` -> `34fecea2`/`26378`), with no gain on dominant kernels.

### "Almost-win?" assessment
- Not an almost-win for this workload.
- Reason: the only structural change is in a tiny non-dominant kernel; dominant expert dequant/reduce kernels do not improve.
- A small backend tweak to the combine kernel alone is unlikely to recover ~`0.7 ms`; wins must come from the two hot expert kernels.

## Metric sanity: why GB/s can go up while kernel gets slower

- tinygrad debug metrics are estimate-based:
  - `GFLOPS = op_est / time`
  - `GB/s = mem_est / time` (and printed as `mem|lds`)
- `mem_est` is static estimate from UOps, not direct hardware counters.
- Therefore a worse schedule can report higher `GB/s` if it moves more bytes for the same work.
- Practical rule for this project:
  - optimize by kernel time first (especially hot kernels),
  - use `GFLOPS/GB/s` only as supporting diagnostics.

## BEAM reduce-only probe (Rejected)

### Setup
- Command: `BEAM=4 BEAM_REDUCE_ONLY=1 DEBUG=6 .venv2/bin/python bench_block.py 1`

### What happened
- Search selected very different FF kernel shapes than the tuned hand heuristic.
- Hot kernels became variants like:
  - `r_4_192_16_64_16`
  - `r_4_2048_16_48`
- Additional kernels (`E_4_98304`, `E_4_196608`) became very expensive in the FF pass (multi-ms range), indicating bad decomposition/materialization for this workload.

### Result
- feed_forward median in this run: `26.28 ms` (clear regression).
- Compared to current tuned baseline (~`18.8-19.0 ms` FF), this is not viable.

### Conclusion
- Keep current hand-tuned MoE heuristic path.
- Use BEAM output as shape diagnostics only; do not adopt these BEAM-selected schedules for decode-path FF.

## Trial: Pre-weight experts before `ffn_down_exps` (Almost-win, Rejected)

### Hypothesis
- Rewrite `sum(p * (W*x))` into `sum(W * (p*x))` so expert-prob scaling can fuse into down path and reduce combine overhead.

### Change tested
- In `MLATransformerBlock._feed_forward`:
  - `weighted_gated = gated * probs.unsqueeze(-1)`
  - `expert_out = ffn_down_exps(sel, weighted_gated)`
  - `out = expert_out.sum(axis=2)`

### Observed kernel effect
- Down hot kernel changed from:
  - `r_4_256_8_8_48_2` (arg 5)
- To:
  - `r_4_256_8_8_48_2` (arg 7) with extra ops including `__truediv__` fused into hot path.
- This confirms normalization math moved into the bottleneck kernel.

### Result
- FF median moved within noise in repeated runs (roughly `18.6-18.9 ms`), no stable win.
- Not accepted because it does not produce consistent FF reduction and complicates hot-kernel math.

## Trial: force `probs` to fp16 before pre-weighting (Rejected)

### Hypothesis
- If pre-weighting is close, casting probs to fp16 might remove expensive normalization/cast pressure inside the hot down kernel.

### Result
- `BEAM=0 bench_block.py 20`:
  - full block median: `1.81 ms`
  - feed_forward median: `19.20 ms`
- FF regressed versus baseline (~`18.8-18.9 ms`).
- Reverted.

## Current anchor (post-revert)
- Baseline path restored in `mla.py`:
  - `expert_out = ffn_down_exps(sel, gated)`
  - `out = (expert_out * probs.unsqueeze(-1)).sum(axis=2)`
- Latest check:
  - full block median: `1.80 ms`
  - feed_forward median: `18.88 ms`

## Benchmark robustness update

- Added robust spread metrics to `bench_block.py` for both loops:
  - `p10/p90`, `IQR`, `MAD` per run.
- Added `--rounds` option:
  - repeats full benchmark rounds and reports aggregate median-of-round-medians.
- Rationale:
  - avoid accepting/rejecting changes from single noisy runs.
  - decision rule should use spread-aware metrics, not one median.

### Current robust anchor (`BEAM=0 bench_block.py 20`)
- full block:
  - median `1.83 ms`, p10/p90 `1.53/1.90`, IQR `0.23`, MAD `0.06`
- feed_forward:
  - median `19.07 ms`, p10/p90 `18.49/19.54`, IQR `0.59`, MAD `0.27`

## Big trial: custom kernel experts (Rejected)

### Attempt
- Added always-on custom Q4_0 expert kernel path in `quantized.py` (fused decode+dot per selected expert row).

### Result
- Regressed hard:
  - full block median around `4.00 ms`
  - feed_forward median around `19-29 ms` depending on run.
- Kernel trace showed custom kernels were extremely slow in this integration:
  - `custom_q4_0_expert_4_3072_2048` often multi-ms
  - `custom_q4_0_expert_4_2048_1536` often multi-ms

### Diagnosis
- Standalone microbench of custom expert kernel was fast (~`0.5 ms`) but integrated path introduced expensive surrounding behavior and did not beat fused DSL hot kernels.
- Reverted.

## Big trial: direct expert-id indexed custom kernel (prototype)

### Attempt
- Prototyped custom kernel that indexes full expert block tensor by `sel[n]` directly (avoids pre-gather materialization).
- Key fix for spec validity: cast expert id to `dtypes.index`.

### Result
- Prototype compiles and runs, but per-matmul time was still ~`0.5 ms` for both hot shapes:
  - `(O=3072, I=2048, N=4)` ~`0.53-0.58 ms`
  - `(O=2048, I=1536, N=4)` ~`0.49-0.56 ms`
- Combined this is still slower than current fused DSL hot kernels for decode path.

### Conclusion
- This is not yet a 2x path.
- True 2x likely needs a deeper core primitive (`mul_mat_id`-style lowering/scheduler path), not just custom-kernel substitution.

## Core design review from docs (2026-02-11)

Read sources:
- `glm_context/llama_cpp.md`
- `reference_material/llamacpp.md`
- `glm_context/pre_separation.md`
- `glm_context/architecture.md`
- `glm_context/bottlenecks.md`
- `CHALLENGE.md`

### Agreed design constraints
- Keep pre-separated Q4_0 tensors (`scale`, `packed`) and preserve gather→reduce fusion until a true primitive beats it.
- A real `mul_mat_id` path must be a core primitive/lowering path, not just swapping in a custom kernel call.
- Expert-id tensor should stay regular integer in memory and only cast to index where required in indexing expressions.
- Kernel-count pressure is as important as kernel GFLOPS; any path that adds barriers/materialization is likely a net loss.

### Core fixes applied
- Added API surface:
  - `UOp.mul_mat_id(...)` in `tinygrad/uop/ops.py`
  - `Tensor.mul_mat_id(...)` in `tinygrad/tensor.py`
- Fixed a rangeify edge case in `tinygrad/schedule/indexing.py`:
  - `realize_map` entries that remain `None` (from index-typed nodes skipped by explicit rangeify) are now treated as passthrough instead of asserting.

### Trial outcome and rollback
- Routing Q4 expert runtime through custom `mul_mat_id` custom-kernel substitution was re-tested and rejected.
- Reverted `QuantizedExpertWeights` back to the previous fast DSL pre-separated path.
- This keeps the production hot path stable while core primitive work continues.

### Validation after rollback
- Representative microbench:
  - `BEAM=0 .venv2/bin/python test_q4_0_bench.py --iters 4 --warmup 1`
  - gate_up median `5.24 ms`, down median `4.89 ms`, combined `10.14 ms`
- Real benchmark (required):
  - `BEAM=0 .venv2/bin/python bench_block.py 3 --model "glm-4.7:flash-unsloth-Q4_0"`
  - full block median `1.54 ms`
  - feed_forward median `19.23 ms`

## Primitive Trial: `mul_mat_id` Q4_0 experts (Accepted)

### Scope
- Files:
  - `tinygrad/apps/quantized.py`
  - `tinygrad/tensor.py`
  - `test/test_custom_kernel.py`
- Goal: one primitive path for Q4_0 expert matmul that is fast in real `bench_block` decode flow.

### What was wrong
- Custom primitive kernel originally emitted wraparound index math in the hot loop for expert-indexed loads.
- In generated Metal this appeared as `idx - size*(idx/size)` style modulo/division inside reduction loops.
- Effect: severe ALU overhead and low kernel throughput.

### Fix 1: index math rewrite (major win)
- In `custom_q4_0_mul_mat_id`:
  - clamp expert id once: `e = sel[n].cast(int).maximum(0).minimum(E-1)`
  - compute flat base index once: `base = ((e * O) + o) * bpr + br`
  - load from flattened `scale`/`packed` tensors.
- Result: removed expensive modulo/division from inner loops in generated Metal.

### Fix 2: launch config tuning for down kernel
- `3072x2048`: keep `LOCAL(1,16) + GROUP(1,16)`.
- `2048x1536`: changed from `GROUP(1,8)` to `LOCAL(1,16) + GROUP(1,16)`.
- Result: down kernel jumped from ~`129 GFLOPS` class to ~`260 GFLOPS` class in steady-state traces.

### Bench evidence (sequential)
- Command: `BEAM=0 .venv2/bin/python bench_block.py 20 --model "glm-4.7:flash-unsloth-Q4_0"`
- Before these two fixes (same primitive path):
  - full block median: `3.14 ms`
  - feed_forward median: `16.25 ms`
- After fixes:
  - full block median: `1.89 ms`
  - feed_forward median: `12.61 ms`
- Net improvement:
  - full block: ~`1.66x` faster
  - feed_forward: ~`1.29x` faster

### Kernel evidence (`DEBUG=2`)
- Full-block warm/cache-hit passes now show:
  - `custom_q4_0_mul_mat_id_4_3072_2048`: ~`284-290 GFLOPS`
  - `custom_q4_0_mul_mat_id_4_2048_1536`: ~`127-262 GFLOPS` (depends on pass; steady FF hit near ~`260`)
- The key change is removal of pathological low-GFLOPS behavior from the down kernel in steady state.

## Focused 3072x2048 matrix + full-model mitigation (2026-02-11)

### Goal
- Keep signal clean: change only `custom_q4_0_mul_mat_id_4_3072_2048` launch shape/reduction, hold other logic constant.
- Then mitigate adverse full-model effects (slow transition tokens) without undoing kernel wins.

### Controlled matrix (bench_block, sequential)
Model: `glm-4.7:flash-unsloth-Q4_0`  
Command: `BEAM=0 .venv2/bin/python bench_block.py 10 --model "glm-4.7:flash-unsloth-Q4_0"`

- Baseline before matrix (`LOCAL16 + GROUP16` for 3072):
  - full `1.87 ms`, ff `12.12 ms`
- Trial A (`LOCAL8 + GROUP8`):
  - full `1.80 ms`, ff `12.33 ms`
- Trial B (`LOCAL16 + GROUP8`):
  - full `1.88 ms`, ff `12.52 ms`
- Trial C (`LOCAL8 + GROUP16`):
  - full `1.77 ms`, ff `12.29 ms`  **best full-block**
- Trial D (`LOCAL8 only`, no GROUP):
  - full `2.20 ms`, ff `12.77 ms`  **rejected**

### Chosen 3072 setting
- Keep: `3072x2048 -> LOCAL(1,8) + GROUP(1,16)`.
- Keep: `2048x1536 -> LOCAL(1,16) + GROUP(1,16)`.

### Mitigation for full-model adverse impact
Observation from `26.log` vs `28.log`: steady tokens were faster with `mul_mat_id`, but transition tokens (4-7) were unstable/slow in some runs.

Mitigation tested:
- In `QuantizedExpertWeights.__call__`, stop forcing `sel_flat = sel.reshape(-1).cast(dtypes.int)`.
- Use `sel_flat = sel.reshape(-1)` and cast/clamp once inside kernel body.
- This removes an extra graph op on the caller side and avoids type churn at the call boundary.

### Results after mitigation
Bench anchor (`count=20`):
- `BEAM=0 .venv2/bin/python bench_block.py 20 --model "glm-4.7:flash-unsloth-Q4_0"`
- full block median: `1.77 ms`
- feed_forward median: `12.45 ms`

Full model (`.venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`):
- Before mitigation (same primitive path):
  - token 6-10 avg: `38.97 ms`
  - token 8-10 avg: `37.74 ms`
- After mitigation:
  - token 6-10 avg: `36.13 ms`  (~`7.3%` faster)
  - token 8-10 avg: `35.72 ms`  (~`5.35%` faster)
  - token 5 improved strongly in this run (`145.79 ms -> 65.49 ms`)

### Kernel sanity check
`DEBUG=2 bench_block.py 1` shows custom kernels remain healthy after mitigation:
- `custom_q4_0_mul_mat_id_4_3072_2048`: ~`269-292 us` in FF warm/cache-hit, up to ~`400+ GFLOPS` in some passes.
- `custom_q4_0_mul_mat_id_4_2048_1536`: ~`224-228 us`, ~`257-262 GFLOPS`.

### Current status
- Accepted:
  - `3072: LOCAL8+GROUP16`
  - `2048: LOCAL16+GROUP16`
  - caller-side `sel` cast removal
- These changes improve both benchmark-loop speed and observed full-model transition behavior while keeping hot kernel throughput high.

## Trial: selective `QuantizedLinear` custom default (Rejected, 2026-02-11)

### Hypothesis
- `custom_q4_0_linear` looked faster in microbench on several decode shapes, so auto-enabling it on FFN-shared shapes might improve block/model latency.

### Micro (representative shape slices)
- `QL_CUSTOM=0` vs `QL_CUSTOM=1`:
  - `attn_q_a (1536x2048)`: `1.893 ms` -> `1.634 ms`
  - `attn_q_b (3072x1536)`: `1.896 ms` -> `1.737 ms`
  - `shared_gateup (3072x2048)`: `1.913 ms` -> `1.679 ms`
  - `shared_down (2048x1536)`: `1.772 ms` -> `1.691 ms`
- Local kernel timing looked positive.

### End-to-end check (bench_block)
- Baseline (`QL_CUSTOM=0`):
  - full block `1.49 ms`, ff `12.30 ms`, warm kernels `33`.
- Full custom (`QL_CUSTOM=1`):
  - full block `2.66 ms`, ff `11.79 ms`, warm kernels `39`.
- Selective auto-enable for FFN-shared shapes only:
  - full block `2.17 ms`, ff `11.19 ms`, warm kernels `37`.

### Conclusion
- Rejected and reverted.
- This is another confirmed case of "micro faster, full pipeline slower": FF kernels got faster, but graph/kernel-count/fusion effects made block latency worse.
- Policy update: full-model benchmark is expensive, so gate it behind a clear block-level win first.

## Trial: `UNROLL` on `custom_q4_0_mul_mat_id` reduce axis (Rejected, 2026-02-11)

### Goal
- Cheap schedule nudge: keep math unchanged, add `UNROLL(..., arg=4)` to the two hot custom kernels:
  - `3072x2048`
  - `2048x1536`
- Intended as low-risk (Q4_0-only path, no Q5/Q6 impact).

### What happened
- Attempt 1: `Opt(UNROLL, axis=2, arg=4)`:
  - Fails at codegen opt application: `KernelOptError` (`UNROLL axis out of range`).
  - Cause: UNROLL axis indexes `unrollable_dims`, not raw kernel axis id.
- Attempt 2: `Opt(UNROLL, axis=0, arg=4)`:
  - Gets past axis mapping, then fails verification:
  - `RuntimeError: UOp verification failed ... Ops.INDEX dtypes.float.ptr(...).vec(4)`.
  - Indicates current custom-kernel lowering + postrange unroll interaction is invalid for this kernel form.

### Decision
- Reverted UNROLL additions entirely.

### Post-revert sanity
- Micro (`BEAM=0 .venv2/bin/python test_q4_0_bench.py --iters 20 --warmup 3`):
  - gate_up `1.15 ms`
  - down `0.93 ms`
  - combined `2.09 ms`
- Block (`BEAM=0 .venv2/bin/python bench_block.py 8 --model "glm-4.7:flash-unsloth-Q4_0"`):
  - full block `1.70 ms`
  - feed_forward `12.10 ms`

### Note
- No full-model run for this trial by design (compile/bench gate failed first).

## Revival Trial: explicit `K-1` adds for MoE combine (full-model A/B, Rejected, 2026-02-11)

### Change
- In `MLATransformerBlock._feed_forward`, replaced:
  - `out = (expert_out * probs.unsqueeze(-1)).sum(axis=2)`
- with explicit adds for small static `K` (decode `K=4`), fallback to `sum(axis=2)`.

### Full model A/B (same command, sequential)
- Command:
  - `BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`

- Baseline (`sum(axis=2)`):
  - t6..t10: `37.04, 36.18, 36.64, 36.27, 37.60 ms`
  - avg t6..t10: `36.75 ms`
  - avg t8..t10: `36.84 ms`

- Trial (explicit adds):
  - t6..t10: `38.73, 36.53, 38.20, 37.43, 37.67 ms`
  - avg t6..t10: `37.71 ms`
  - avg t8..t10: `37.77 ms`

### Result
- Regression in steady decode:
  - `+2.63%` on t6..t10 average
  - `+2.52%` on t8..t10 average
- Reverted to baseline `sum(axis=2)` implementation.

## Hot kernels + barrier/custom-linears sweep (2026-02-11, sequential)

### Current hot kernels (from latest recent profile artifacts)
- `custom_q4_0_mul_mat_id_4_3072_2048`
- `custom_q4_0_mul_mat_id_4_2048_1536`
- recurrent large-reduce kernels:
  - `r_192_16_4_4_128`
  - `r_128_16_4_4_10_16n1`
  - `r_128_16_4_4_96`

### Trial 1: remove `expert_out.contiguous()` in `_feed_forward`
- Effect:
  - kernel count dropped (`33 -> 32` full, `18 -> 17` ff)
  - ff improved slightly
  - but full-block median regressed in mirror A/B (`1.13 ms` vs baseline `1.06 ms` in strong run)
- Decision: rejected, reverted.

### Trial 2: remove `gated.contiguous()` in `_feed_forward`
- Effect:
  - kernel count dropped (`33 -> 32` full, `18 -> 17` ff)
  - ff improved slightly
  - full-block median regressed (`1.20 ms` vs baseline `1.06 ms`)
- Decision: rejected, reverted.

### Trial 3: remove `out.contiguous()` before shared-expert add
- Effect:
  - kernel count dropped (`33 -> 32` full, `18 -> 17` ff)
  - both full and ff medians worsened in aggregate (`full 1.36 ms`, `ff 12.47 ms`)
- Decision: rejected, reverted.

### Trial 4: targeted `QuantizedLinear` custom path only for `3072x2048`
- Effect:
  - kernel count increased (`33 -> 35` full, `18 -> 20` ff)
  - aggregate regressed (`full 1.68 ms`, `ff 12.81 ms`)
- Decision: rejected, reverted.

### Trial 5: targeted `QuantizedLinear` custom path only for `2048x1536`
- Effect:
  - kernel count increased (`33 -> 35` full, `18 -> 20` ff)
  - no full-block gain in immediate A/B (`1.39 ms` baseline-equivalent), ff not materially better
- Decision: rejected, reverted.

### Trial 6: launch-shape micro matrix for custom `mul_mat_id` kernels
- Micro-only sweep found apparent alternatives (e.g. up `(8,8)` or `(16,16)`).
- Block-level mirror checks were unstable and did not produce a robust win over current default.
- Decision: keep current file defaults:
  - `3072x2048 -> LOCAL(1,8) + GROUP(1,16)`
  - `2048x1536 -> LOCAL(1,16) + GROUP(1,16)`

### Net
- No accepted change in this sweep.
- Main lesson repeated: micro wins and lower kernel count alone are not sufficient; full-block median gate is decisive.

## Follow-up sweeps (2026-02-11, continued)

### Targeted launch-shape re-check (runtime monkeypatch, no source commit)
- Re-tested candidate launch configs against current defaults for `custom_q4_0_mul_mat_id`:
  - current: `3072 -> (LOCAL8, GROUP16)`, `2048 -> (LOCAL16, GROUP16)`
  - candidates: `(8,8)` and `(16,16)` on 3072 side.
- Outcome: no robust winner after mirror order checks; large run-to-run drift, and candidates did not consistently beat current on full-block aggregate.
- Decision: keep current default launch config in source.

### Q4_0 DSL packed read duplication (`QuantizedLinear`) re-check
- Trial: duplicate `_q4_0_packed` branch consumers (`lo` and `hi`) instead of sharing a single `packed` node.
- Mixed results across runs (some same-process wins, other immediate mirrors flat/worse).
- No stable improvement signal under repeated `bench_block` gates.
- Decision: keep baseline shared `packed` expression for now.

### Core behavior discovery: fixed opts bypass heuristics/beam
- `postrange.apply_opts` behavior:
  - if `opts_to_apply is not None`, it applies only those opts and skips heuristic/beam.
- Trialed `opts_to_apply=None` for custom `mul_mat_id` to let heuristics pick schedule.
- Result: correctness failure during linearization:
  - `RuntimeError: UOp verification failed ... Ops.INDEX ... vec(48)`.
- Interpretation: current fixed opts are effectively required for correctness for this kernel form, not only performance.

## Strong A/B rerun (2026-02-11, BEAM=0, alternating pairs)

### Hypothesis
- Pin a BEAM-discovered shape table in `/Users/rvd/src/rvd/tinygrad/tinygrad/codegen/opt/heuristic.py` to reproduce BEAM gains without BEAM.

### Method
- Alternating strong A/B with same command and machine state:
  - A: baseline heuristic
  - B: shape-pin patch
- Sequence: `A1/B1/A2/B2/A3/B3`, each run:
  - `BEAM=0 .venv2/bin/python bench_block.py 20 --model "glm-4.7:flash-unsloth-Q4_0"`

### Results
- Per-pair medians:
  - `A1 full 1.26 ms, ff 10.02 ms` vs `B1 full 1.35 ms, ff 10.99 ms`
  - `A2 full 1.21 ms, ff 9.90 ms` vs `B2 full 1.16 ms, ff 10.00 ms`
  - `A3 full 0.93 ms, ff 10.13 ms` vs `B3 full 1.30 ms, ff 9.97 ms`
- Aggregate:
  - Baseline A:
    - full median mean `1.133 ms`
    - ff median mean `10.017 ms`
  - Patched B:
    - full median mean `1.270 ms`
    - ff median mean `10.320 ms`
  - Delta (B-A):
    - full `+0.137 ms` (worse)
    - ff `+0.303 ms` (worse)

### Decision
- Rejected. Restored baseline heuristic.
- This confirms the earlier apparent jump was a fluke/noise episode, not a robust speedup.

## Non-custom kernel sweep (2026-02-11, sequential)

### Focus
- Hot non-custom decode kernels:
  - `r_32_32_3_512_4n1` (shared expert gate/up)
  - `r_128_16_4_4_96n1` (shared expert down)
  - `r_16_32_4_4` (small K combine/scale path)

### Trial 1: shape-targeted upcast preference (`3072x2048`: prefer `UPCAST=4` over `3`)
- File: `tinygrad/codegen/opt/heuristic.py`
- Strong alternating A/B (`A1/B1/A2/B2/A3/B3`, `BEAM=0 bench_block.py 20`)
  - Patched A medians:
    - full: `1.35, 1.60, 1.76` (mean `1.57`)
    - ff: `9.66, 9.68, 9.77` (mean `9.70`)
  - Baseline B medians:
    - full: `1.60, 1.86, 1.62` (mean `1.69`)
    - ff: `9.73, 10.01, 9.81` (mean `9.85`)
- Follow-up rerun (same day, same command) showed inversion:
  - patched: `full 1.58 ms`, `ff 10.09 ms`
  - baseline: `full 1.21 ms`, `ff 10.12 ms`
- Decision: rejected for now, reverted. Signal is too unstable and latest mirror favors baseline on full-block latency.

### Trial 2: pre-realize contiguous `(I,O)` dequant cache for non-Q4 linears
- Files: `tinygrad/apps/quantized.py`, `tinygrad/apps/mla.py`
- Hypothesis: avoid transposed-stride access in shared expert non-custom matmuls.
- Results:
  - one run looked strong on full (`0.97 ms`) but ff regressed (`10.69 ms`)
  - mirror A/B did not hold (`trial full 1.69 ms` vs baseline `1.48/1.58 ms`)
- Decision: rejected, reverted.

### Trial 3: move MoE probability weighting before down projection (linearity rewrite)
- File: `tinygrad/apps/mla.py`
- Rewrite: `down(gated) * probs -> down(gated * probs)` then `sum(axis=2)`.
- Result: no win (`full 1.62 ms`, `ff 9.97 ms`); kernel count did not improve.
- Decision: rejected, reverted.

### Trial 4: replace `probs / denom` with `probs * reciprocal(denom)`
- File: `tinygrad/apps/mla.py`
- Initial run looked mildly positive, but follow-up runs regressed and were unstable (`full ~1.60 ms`, `ff ~11.05 ms` in a rerun).
- Decision: rejected, reverted.

### Trial 5: matvec decode-shape specialization (`3072x2048 -> LOCAL8/GROUP16/UPCAST2`)
- File: `tinygrad/codegen/opt/heuristic.py`
- Result: no robust gain (`full 1.61 ms`, `ff 9.80 ms`).
- Decision: rejected, reverted.

### Trial 6: force fp16 accumulation for non-Q4 quantized linear fallback
- File: `tinygrad/apps/quantized.py`
- Result: regression (`full 1.63 ms`, `ff 10.58 ms`).
- Decision: rejected, reverted.

## Non-custom kernel sweep II (2026-02-11, sequential)

### Baseline anchors (before trials)
- `BEAM=0 bench_block.py 20` mirrors:
  - `full 1.82 ms, ff 10.12 ms`
  - `full 1.86 ms, ff 9.81 ms`

### Trial A: fold `expert_weights_scale` into `probs` before MoE combine
- File: `tinygrad/apps/mla.py`
- Change:
  - `probs *= expert_weights_scale`
  - remove trailing `* expert_weights_scale` after `sum(axis=2)`
- Result:
  - trial: `full 1.63 ms`, `ff 10.06 ms`
  - mirror baseline: `full 1.32 ms`, `ff 10.31 ms`
- Decision: rejected (full-block worse in immediate mirror).

### Trial B: cast `probs` to `h_norm.dtype` after normalization
- File: `tinygrad/apps/mla.py`
- Result:
  - trial: `full 1.64 ms`, `ff 9.92 ms`
  - mirror baseline: `full 1.22 ms`, `ff 10.09 ms`
- Decision: rejected (full-block regression despite slight ff gain).

### Trial C: selective fusion break for shared expert (`h_norm_shexp = h_norm.contiguous()`)
- File: `tinygrad/apps/mla.py`
- Result: `full 1.61 ms`, `ff 10.48 ms`
- Decision: rejected.

### Trial D: matvec shape pin for both non-custom shared expert kernels
- File: `tinygrad/codegen/opt/heuristic.py`
- Pin:
  - `[3072,2048] -> LOCAL8/GROUP8/UPCAST4`
  - `[2048,1536] -> LOCAL8/GROUP8/UPCAST4`
- Result:
  - trial: `full 1.42 ms`, `ff 10.56 ms`
  - mirror baseline: `full 1.45 ms`, `ff 10.25 ms`
- Decision: rejected (ff regression).

### Trial E: matvec shape pin only for `[2048,1536]`
- File: `tinygrad/codegen/opt/heuristic.py`
- Pin: `[2048,1536] -> LOCAL4/GROUP16/UPCAST2`
- Result: `full 1.72 ms`, `ff 10.18 ms`
- Decision: rejected.

### BEAM extraction and replay attempt (non-custom only)
- Ran: `BEAM=4 BEAM_REDUCE_ONLY=1 DEBUG=6 bench_block.py 1`
- Extracted BEAM-selected schedules:
  - `3072x2048` path: `GROUPTOP(0,16)` (kernel `r_3072_16_128`)
  - `2048x1536` path: `GROUP(0,16)+GROUP(0,4)` (kernel `r_2048_16_4_24`)
- Replayed these in heuristic under `BEAM=0`.
- Result:
  - pinned: `full 1.83 ms`, `ff 10.22 ms`
  - mirror baseline: `full 1.59 ms`, `ff 10.37 ms`
- Decision: rejected; BEAM-discovered shapes did not survive non-BEAM end-to-end timing.

## Core change: reuse cached BEAM schedules in normal path (2026-02-11)

### Motivation
- `BEAM=4 BEAM_REDUCE_ONLY=1` (cached) consistently produced much faster full-block medians than plain `BEAM=0`, but production path was not reading beam cache.
- Goal: keep one default path and automatically reuse known-good beam opts when cache exists, without running beam search online.

### Change
- File: `tinygrad/codegen/opt/postrange.py`
- In `apply_opts`, before heuristic path (`BEAM=0`), read `beam_search` diskcache for current AST key and apply cached opts directly when present.
- Fallback remains unchanged heuristic if cache key is absent.
- Guarded by:
  - `CACHELEVEL >= 1`
  - `BEAM_CACHE_READ` (default `1`)
  - `BEAM_CACHE_AMT` (default `4`)

### A/B (alternating, `BEAM=0 bench_block.py 20`)
- A1 patched: `full 1.42 ms`, `ff 10.22 ms`
- B1 baseline: `full 1.88 ms`, `ff 10.50 ms`
- A2 patched: `full 1.24 ms`, `ff 10.25 ms`
- B2 baseline: `full 1.26 ms`, `ff 10.34 ms`
- A3 patched: `full 1.57 ms`, `ff 10.21 ms`
- B3 baseline: `full 1.63 ms`, `ff 10.48 ms`

### Full-model validation
- Command:
  - `BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Steady-token section (t6..t10) with patch:
  - `37.27, 36.95, 38.09, 41.26, 42.72 ms`
  - first two tokens looked fine, then decode tailed off badly to `~23-24 tok/s`.
- This is worse than recent baseline behavior around high-20s tok/s.

### Decision
- Rejected and reverted.
- Interpretation: bench_block looked mildly positive/noisy, but full-model decode stability regressed, so this path is not production-safe.
