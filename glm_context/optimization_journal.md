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

## Trial: `QL_SHEXP_MSL_NR_3072_2048=2` + `QL_SHEXP_MSL_NR_2048_1536=1` (Rejected)

### Hypothesis
- Per-shape `NR` override might be a better default than current shared-expert MSL `NR=1`.

### A/B/A/B setup (sequential only)
- Command A (default):
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python bench_block.py 5 --model "glm-4.7:flash-unsloth-Q4_0"`
- Command B (override):
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 QL_SHEXP_MSL_NR=1 QL_SHEXP_MSL_NR_3072_2048=2 QL_SHEXP_MSL_NR_2048_1536=1 .venv2/bin/python bench_block.py 5 --model "glm-4.7:flash-unsloth-Q4_0"`
- Order: A -> B -> A -> B

### Results
- A1 decode proxy: `41.16 tok/s`
- B1 decode proxy: `40.36 tok/s`
- A2 decode proxy: `41.07 tok/s`
- B2 decode proxy: `40.88 tok/s`

### Conclusion
- In this stronger sequential A/B/A/B, the override is consistently slower than default by about `0.2-0.8 tok/s`.
- Keep default behavior as-is; do not promote this override to built-in default.

## Full-model validation (current default)

- Command:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Result (steady decode region):
  - `27.92, 31.66, 31.70, 32.42, 34.01, 31.38 tok/s`
  - practical steady-state band in this run: roughly `31-34 tok/s`

## Full-model A/B: default vs shared NR override

- A (default):
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - tok/s lines:
    - `19.84, 30.73, 32.20, 32.91, 32.56`
- B (shared override):
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 QL_SHEXP_MSL_NR=1 QL_SHEXP_MSL_NR_3072_2048=2 QL_SHEXP_MSL_NR_2048_1536=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - tok/s lines:
    - `31.04, 33.89, 33.57, 33.63, 33.79, 33.74`

Interpretation (single A/B pair):
- Using the stable tail (`last 4`):
  - default: `30.73, 32.20, 32.91, 32.56` (avg `32.10`)
  - override: `33.57, 33.63, 33.79, 33.74` (avg `33.68`)
- In this run, shared override is about `+1.58 tok/s` (~`+4.9%`) vs default.

## Accepted: make winning shared-expert NR settings default

### Change
- File: `tinygrad/apps/qk_linear_msl.py`
- Updated `_nr_for_qk` default behavior:
  - Q5_K shared expert shape `(3072, 2048)`: default `NR=2`
  - Q6_K shared expert shape `(2048, 1536)`: default `NR=1`
- Env overrides remain available:
  - global: `QL_SHEXP_MSL_NR`
  - per-shape: `QL_SHEXP_MSL_NR_3072_2048`, `QL_SHEXP_MSL_NR_2048_1536`

### Validation
- Kernel-default check (`no NR env overrides`):
  - `PYTHONPATH=. BEAM=0 DEBUG=2 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
  - Observed kernels:
    - `q5_k_linear_msl_1_3072_2048_nr2`
    - `q6_k_linear_msl_1_2048_1536_nr1`
- Full-model smoke:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - tail decode in this run: about `29-32 tok/s` (one noisy run; use repeated full-model runs for tighter comparison).

## Accepted: default NR for shared-expert MSL kernels

- File: `tinygrad/apps/qk_linear_msl.py`
- Change:
  - `_nr_for_qk` now hardcodes:
    - Q5_K `(3072, 2048)` -> `NR=2`
    - Q6_K `(2048, 1536)` -> `NR=1`
  - No env override path in this default mapping.

## Big trial: dense Q4_0 MSL kernel for regular `QuantizedLinear`

### Motivation from profile
- Large non-custom Q4-related reduce kernels still consumed significant warm-up time.
- Added dedicated MSL path to move dense Q4_0 off generic gather/dequant/reduce.

### Implementation
- New file: `tinygrad/apps/q4_linear_msl.py`
  - custom tag + lowering runner for dense Q4_0 linear:
    - `custom_q4_0_linear_msl`
    - `lower_q4_linear_msl_ast`
- Wiring:
  - `tinygrad/engine/realize.py`: lowerer dispatch for `q4_0_linear_msl`
  - `tinygrad/apps/quantized.py`: in Q4_0 packed-dot path, when `QL_MOE_MSL=1` on METAL:
    - route to `Tensor.custom_kernel(..., fxn=custom_q4_0_linear_msl)`

### Bench evidence
- `bench_block.py 2` with `QL_MOE_MSL=1 QL_SHEXP_MSL=1`:
  - observed decode proxy up to `43.51 tok/s` in one run
  - repeated run in same state around `42.31 tok/s`
- Kernel logs confirm usage:
  - `q4_0_linear_msl_1_768_2048_nr1`
  - `q4_0_linear_msl_1_5120_768_nr1`
  - `q4_0_linear_msl_1_2048_5120_nr1`

### Full model checks
- Example run:
  - `34.09, 34.82, 34.74, 35.07, 34.28 tok/s` (tail region)
- Another run showed lower `~29-33 tok/s`, indicating variance/thermal/system noise remains high.

## Rejected follow-up tuning

- `q4_moe` NR tuning:
  - `QL_MOE_MSL_NR_3072_2048=2` regressed decode proxy.
  - `QL_MOE_MSL_NR_2048_1536=2` regressed decode proxy.
- `q4_moe` threads:
  - `QL_MOE_MSL_THREADS=64` was not better than baseline.
- `q4_0_linear_msl` NR tuning:
  - forcing `NR=2` for `(2048, 5120)` regressed.
  - forcing `NR=2` globally regressed.

## Post-change full-model samples (dense Q4 MSL enabled)

- Command:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Sample A tail:
  - `29.72, 32.28, 33.49, 33.27, 33.05 tok/s`
- Sample B tail:
  - `30.71, 32.85, 34.49, 34.30, 34.79 tok/s`
- Observation:
  - still a noisy but roughly `33-35 tok/s` band in these samples; not near `45 tok/s` yet.

## Current top blockers from latest profile (`profile_model.py`)

- `q4_moe_mul_mat_id_msl_4_3072_2048_t32_nr1`: largest single kernel bucket.
- `q4_moe_mul_mat_id_msl_4_2048_1536_t32_nr1`: second major MoE bucket.
- `q4_0_linear_msl_1_2048_5120_nr1`: large dense Q4 bucket.
- `q5_k_linear_msl_1_3072_2048_nr2` and `q6_k_linear_msl_1_2048_1536_nr1`: still significant.
- Residual large reduce kernels remain (`r_9680_...`, `r_36_...`, `r_4_16_...`) and likely map to MLA/attention-side math that is still generic.
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

## Trial: `MLA_MOE_SPLIT_BOUNDARIES` in full model (2026-02-11)

### Setup
- Code: `tinygrad/apps/mla.py` has optional boundary barriers around MoE gate/up/down paths.
- Bench signal looked promising with boundaries off:
  - `MLA_MOE_SPLIT_BOUNDARIES=1`: ff `11.35 ms`, 19 kernels
  - `MLA_MOE_SPLIT_BOUNDARIES=0`: ff `9.94 ms`, 15 kernels

### Full-model A/B
- Baseline (previous run):
  - steady tokens t6..t10: `24.80, 24.52, 25.39, 25.44, 25.34 tok/s` (~`25.1 tok/s` avg)
- Trial (`MLA_MOE_SPLIT_BOUNDARIES=0`):
  - command: `BEAM=0 MLA_MOE_SPLIT_BOUNDARIES=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - steady tokens t6..t10: `20.56, 21.55, 21.35, 21.00, 21.55 tok/s` (~`21.2 tok/s` avg)

### Decision
- Rejected as default behavior.
- Despite fewer kernels and better bench feed-forward, full-model decode regressed by ~15% in steady tok/s.

## Trial: remaining `glm.py` ideas (2026-02-11)

### Idea 1: pairwise top-k vs generic `.topk`
- Patch trial: allow switching MoE selection between pairwise and `.topk`.
- `bench_block`:
  - pairwise: ff `10.51 ms`, 19 kernels
  - `.topk`: ff `29.05 ms`, 46 kernels
- full model (`MLA_PAIRWISE_TOPK=0`): steady t6..t10
  - `23.18, 23.81, 23.78, 23.54, 23.81 tok/s` (~`23.6 tok/s`)
- compared to pairwise baseline (~`25 tok/s`), `.topk` is slower.
- Decision: keep pairwise; remove switch.

### Idea 2: attention-length bucketing (`GLM_ATTN_MIN_BUCKET`)
- First observation: prior implementation only activated for integer `start_pos`; with `SYM=0` this disables JIT and collapses throughput (~`0.45 tok/s`).
- Added a symbolic-compatible fixed-bucket trial and tested:
  - command: `BEAM=0 GLM_ATTN_MIN_BUCKET=256 ... --benchmark 10`
  - steady t6..t10: `19.75, 23.42, 23.31, 23.55, 22.90 tok/s` (~`22.6 tok/s`)
- baseline on same code path with bucket disabled:
  - steady t6..t10: `25.01, 24.81, 24.87, 25.01, 25.06 tok/s` (~`25.0 tok/s`)
- Decision: reject bucketing for current decode benchmark regime.

### Cleanup
- Reverted trial toggles and symbolic-bucket logic from `tinygrad/apps/mla.py`.
- Kept the proven pairwise top-k path and existing merged MoE/Q4 expert fast path.

## Trial: shift contiguous later in MoE combine (2026-02-11)

### Hypothesis
- Similar to `gated` change: avoid early materialization and place `.contiguous()` after larger fused expressions.
- Specifically:
  - `expert_out = ffn_down_exps(sel, gated)` (no immediate contiguous)
  - `out = (expert_out * probs.unsqueeze(-1)).sum(axis=2).contiguous() * scale`
  - shared expert add: `out = (out + shexp).contiguous()` instead of `out.contiguous() + shexp_term`

### bench_block (20 iters)
- before:
  - full `1.74 ms`, ff `10.29 ms`, kernels `32` full / `17` ff
- after:
  - full `1.28 ms`, ff `10.15 ms`, kernels `31` full / `16` ff

### full model
- command: `BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- steady t6..t10: `22.71, 25.29, 25.02, 25.03, 25.14 tok/s`
- Compared to recent baseline (`~24.8-25.1 tok/s` steady), this looks neutral/slightly positive but noisy.

### Decision
- Keep for now as a low-risk cleanup with small-kernel-count reduction and no observed full-model regression.
- Revalidate with a stronger multi-run A/B before treating as a hard win.

## Trial batch: additional contiguous-placement candidates (2026-02-11)

### Baseline for this batch
- `bench_block.py 20 --model glm-4.7:flash-unsloth-Q4_0`
- full: `1.36 ms`, ff: `10.31 ms`, kernels: `31` full / `16` ff

### Candidate 1: dense FFN in `llm.py` (`TransformerBlock`)
- Change tested: `self.ffn_gate(...).silu().contiguous() * self.ffn_up(...)`
  -> `(self.ffn_gate(...).silu() * self.ffn_up(...)).contiguous()`
- Microbench (`TransformerBlock._feed_forward`, dim=2048, hidden=6144):
  - baseline: `median 3.641 ms`, kernels `84`
  - trial: `median 4.019 ms` then `3.650 ms`, kernels `63`
- Interpretation: fewer kernels but no clear speedup; also not on GLM MLA hot path.
- Decision: reject.

### Candidate 2: remove trailing block output contiguous in `MLATransformerBlock.__call__`
- Change tested: return `_feed_forward(_attention(...))` (no trailing `.contiguous()`)
- `bench_block`:
  - run1: full `1.34 ms`, ff `10.90 ms`
  - run2: full `2.02 ms`, ff `11.27 ms`
- Interpretation: ff path consistently slower.
- Decision: reject.

### Candidate 3: relax `mul_mat_id` source contiguity in `uop/ops.py`
- Change tested: stop forcing contiguous on already-buffer-like non-output srcs.
- `bench_block`:
  - run1: full `2.10 ms`, ff `10.48 ms`
  - run2: full `1.14 ms`, ff `10.56 ms`
- Interpretation: high full-block noise, but ff median regressed vs baseline.
- Decision: reject.

### Outcome
- No additional keepers from this batch.
- Existing good contiguous rewrite in `mla.py` remains the best result from this direction.

## Trial: Q5_K/Q6_K custom kernels in `QuantizedLinear` (2026-02-11)

### Goal
- Try the same custom-kernel strategy used for Q4 experts on Q5/Q6 shared-expert linears first.

### What was tested
- Added custom kernels for:
  - `custom_q5_k_linear_1_3072_2048`
  - `custom_q6_k_linear_1_2048_1536`
- Added preprocess/separation caches for Q5_K/Q6_K block fields and routed `ggml_type in {13,14}` through `Tensor.custom_kernel`.

### Why this looked plausible
- Model GGUF actually contains significant Q5/Q6:
  - ggml type `13` (Q5_K): `92` tensors (~`0.199 GB` quant bytes)
  - ggml type `14` (Q6_K): `47` tensors (~`0.379 GB` quant bytes)
- Shared experts are Q5/Q6 in this model family:
  - `ffn_gate_shexp`, `ffn_up_shexp` -> Q5_K
  - `ffn_down_shexp` -> Q6_K

### DEBUG=2 findings (bench loop)
- New custom kernels were directly on hot path and expensive:
  - `custom_q5_k_linear_1_3072_2048`: ~`1.20 ms`
  - `custom_q6_k_linear_1_2048_1536`: ~`0.60 ms` with spikes up to `~2.25 ms`
- Full block warmup capture rose to `34` kernels (from prior `~31` in current baseline path).
- One-iter debug run showed severe regression:
  - full block median: `5.34 ms`
  - feed_forward median: `17.82 ms`

### Decision
- **Rejected and reverted** Q5/Q6 custom path.
- Keep Q4 expert `mul_mat_id` custom primitive path only.

### Post-revert check (`BEAM=0 bench_block.py 20`)
- full block median: `1.25 ms`
- feed_forward median: `10.44 ms`
- warmup kernels: `31` full / `16` ff
- Confirms regression source was Q5/Q6 custom implementation.

## Q5/Q6 focused pass (2026-02-11, follow-up)

### Goal
- Improve the Q5_K/Q6_K shared-expert path first, without handwritten backend kernels.

### Ground truth on this model
- GGUF quant type counts/bytes (`glm-4.7:flash-unsloth-Q4_0`):
  - Q4_0 (`type 2`): `278` tensors, ~`15.80 GB` quant bytes
  - Q5_K (`type 13`): `92` tensors, ~`0.199 GB`
  - Q6_K (`type 14`): `47` tensors, ~`0.379 GB`
- Shared experts are Q5/Q6 in this checkpoint:
  - `ffn_gate_shexp`, `ffn_up_shexp` -> Q5_K
  - `ffn_down_shexp` -> Q6_K

### Trial 1: full custom Q5/Q6 kernels in `QuantizedLinear`
- Added custom kernels + pre-separated Q5/Q6 caches.
- Rejected: severe bench regression.
  - debug hot kernels:
    - `custom_q5_k_linear_1_3072_2048` ~`1.2 ms`
    - `custom_q6_k_linear_1_2048_1536` ~`0.6–2.2 ms` (spiky)
  - feed_forward jumped to ~`17.8 ms` in debug run.

### Trial 2: quant-aware DSL Q5/Q6 grouped dot path (no custom kernel)
- Kept packed Q5/Q6 separated and computed grouped reductions in DSL.
- Rejected: feed_forward regressed (~`12.2 ms` vs ~`10.3 ms` baseline).

### Trial 3: dequant-cache transpose-contiguous storage
- Cached dequantized weights as transposed contiguous and used `x.dot(cache)`.
- Rejected: no gain; feed_forward regressed (~`11.0 ms`).

### Trial 4: split shared Q6 down input (`(silu*up).contiguous()` before `ffn_down_shexp`)
- This changed kernel makeup as expected:
  - `r_2048_16_96` dropped from ~`140us` to ~`80us`
  - plus a small extra `silu*mul` kernel (~`8–10us`)
- Microbench looked near-neutral/slightly positive in some runs, but full-model A/B did not show a clear real tok/s win.
- Full-model comparison (`--benchmark 10`, BEAM=0):
  - split variant steady decode was not better than baseline in a reliable way.
- Decision: rejected for default path.

### Current best default (kept)
- Keep baseline Q5/Q6 path: dequantize-once cache + linear.
- Keep Q4 expert `mul_mat_id` custom path unchanged.

### Baseline reconfirmation after this pass
- `bench_block.py 20`:
  - full block ~`1.27 ms`
  - feed_forward ~`10.28 ms`
- full model (`--benchmark 10`) steady tokens are still ~mid-24 to ~25 tok/s depending on token/window in run.

### Key takeaway
- Q5/Q6 are present and important, but in this model they are not currently the dominant limiter vs the larger Q4 expert path and related non-shared MoE kernels.
- Custom or semi-custom Q5/Q6 rewrites were easy to make slower; wins seen in isolated kernel timing did not carry to end-to-end tok/s.

## Benchmark correctness fix: add decode-stack proxy (2026-02-11)

### Problem
- Single-block / feed-forward microbench could report local wins that did not survive full-model decode.
- If benchmark signal disagrees with model tok/s, benchmark is wrong for optimization decisions.

### Fix
- Updated `bench_block.py` to add a **decode-stack JIT proxy**:
  - Runs one decode token through a stack of MoE blocks (`46` by default for GLM-4.7 flash).
  - Uses same block shapes/ops and captures real kernel volume and graph batching behavior.
  - Keeps fast loop (no full model load/benchmark), but much closer to model decode critical path.

### Why this matters
- It now exposes the actual kernel scale directly:
  - warmup shows ~`1426` kernels (`31 kernels/block * 46 blocks`), then graph-batched execution.
- This explains the previously surprising `>1400 kernels` and gives a metric that tracks decode-step reality better than single-block medians.

### New baseline (BEAM=0, `bench_block.py 20`)
- full block JIT median: `1.53 ms`
- decode stack JIT median (`46 blocks`): `28.08 ms` -> `35.61 tok/s` proxy
- feed_forward median: `9.90 ms`

### Decision
- Keep decode-stack metric as the primary go/no-go signal for micro-iteration.
- Continue to confirm only promising changes on full model afterward.

## Split/Fuse + BEAM coupling experiment (2026-02-11)

### Goal
- Check whether split/fuse boundary decisions are independent from kernel opt search.

### Setup
- New generic harness: `search2.py` (module contract `bench_mask(mask, iters, beam)`).
- `bench_block.py` now exposes:
  - `SPLIT_POINTS` (17 MLA boundary bits)
  - `bench_mask(...)` using cached block/weights per process.
- Small run: exhaustive sample of 8 masks (`--max-masks 8`, `iters=4`, `repeat=2`).

### Results
- `beam=0`:
  - baseline `mask=0x00000`: `28.86 ms`
  - best in sample: `mask=0x1c000`: `27.37 ms`
- `beam=4`:
  - baseline `mask=0x00000`: `27.91 ms`
  - best in sample: `mask=0x04000`: `25.86 ms`
  - next: `mask=0x10000`: `25.96 ms`, `mask=0x1c000`: `26.05 ms`

### Interpretation
- Split/fuse is **not independent** of BEAM/opts.
- Ranking changed between `beam=0` and `beam=4`:
  - `0x1c000` was best at `beam=0` sample.
  - `0x04000` became best at `beam=4`.
- Therefore boundary search must be coupled with opt search (outer split mask, inner BEAM).

### Decision
- Keep `search2.py` and `bench_block.bench_mask` path for co-search workflow.
- Next step: larger mask set + stronger repeats using `beam=4`, then validate top mask on full model.

## Follow-up: stronger mask A/B (BEAM=4) + notes (2026-02-11)

### Infrastructure notes captured
- `search2.py` now supports general split-mask search over any module exposing:
  - `SPLIT_POINTS` (optional names)
  - `bench_mask(mask, iters, beam) -> ms`
- `bench_block.py` now exports `SPLIT_POINTS` + `bench_mask(...)`.
- Weight/block load behavior for this path:
  - within one Python process: loaded once and reused across mask evaluations (via `_BENCH_MASK_CACHE`)
  - across separate command invocations: each process loads once.

### Stronger A/B command
- Command run:
  - `BEAM=4 .venv2/bin/python - <<'PY' ...`
  - Compared masks: `0x00000` (baseline), `0x04000`, `0x10000`, `0x1c000`
  - Each mask measured 5 times via `bench_mask(mask, iters=6, beam=4)`.

### Results
- `mask=0x00000`: vals `[25.74, 26.94, 27.41, 28.81, 27.45]`, median `27.41 ms` (`36.48 tok/s`)
- `mask=0x04000`: vals `[27.10, 27.03, 26.90, 28.11, 27.23]`, median `27.10 ms` (`36.90 tok/s`)
- `mask=0x10000`: vals `[26.90, 26.86, 27.33, 27.54, 27.77]`, median `27.33 ms` (`36.59 tok/s`)
- `mask=0x1c000`: vals `[28.63, 28.86, 27.82, 32.71, 26.87]`, median `28.63 ms` (`34.92 tok/s`)

### Interpretation
- Best median in this run: `0x04000` (split only `ffn_shexp_gate_up`), small gain vs baseline:
  - `27.41 -> 27.10 ms` (~`+1.1%` tok/s proxy)
- `0x10000` is near-baseline.
- `0x1c000` is unstable in this stronger test (large tail), not a keeper.
- Conclusion: yes, we found a better mask in bench proxy, but it is a **small** win and needs full-model confirmation before keeping.

## Follow-up: same stronger A/B with BEAM=2 (2026-02-11)

### Setup
- Same mask set and methodology as prior stronger run:
  - masks: `0x00000`, `0x04000`, `0x10000`, `0x1c000`
  - each measured 5 times via `bench_mask(mask, iters=6, beam=2)`

### Results
- `mask=0x00000`: vals `[26.62, 26.82, 28.53, 27.29, 27.78]`, median `27.29 ms` (`36.65 tok/s`)
- `mask=0x04000`: vals `[28.13, 27.74, 28.05, 27.97, 28.37]`, median `28.05 ms` (`35.65 tok/s`)
- `mask=0x10000`: vals `[28.02, 26.32, 28.37, 25.33, 28.32]`, median `28.02 ms` (`35.69 tok/s`)
- `mask=0x1c000`: vals `[27.87, 27.97, 27.71, 27.91, 27.94]`, median `27.91 ms` (`35.82 tok/s`)

### Interpretation
- With `BEAM=2`, baseline fused mask `0x00000` is best among these candidates.
- This differs from earlier `BEAM=4` run where split masks (`0x04000`/`0x10000`) looked best.
- Reinforces coupling: split/fuse ranking depends strongly on opt search regime (`BEAM` depth and chosen opts).

## Primitive-pipeline refactor pass (2026-02-11)

### Goal
- Move closer to llama.cpp-style decode structure without handwritten backend kernels:
  - fixed MoE primitive pipeline in `mla.py`
  - primitive `mul_mat_id` path for non-Q4 experts (Q5_K/Q6_K)
  - minimal explicit split points, then co-search with BEAM.

### Code changes
- `tinygrad/apps/mla.py`
  - Simplified `_attention` back to normal fused DSL (removed broad split instrumentation).
  - `_feed_forward` now follows fixed pipeline:
    - router -> `ffn_gate_up_exps(sel, h_norm)` -> `silu*up` -> `ffn_down_exps(sel, gated)` -> weighted combine -> + shared expert.
  - Kept only 3 split points via `MLA_SPLIT_MASK`:
    - bit 0: `ffn_moe_shared_boundary` (default ON via `getenv(..., 1)`)
    - bit 1: `ffn_shexp_gate_up`
    - bit 2: `ffn_shexp_down`
- `tinygrad/apps/quantized.py`
  - Extended decode-specialized opts helper to include `n_sel` in Q4 `mul_mat_id` scheduling.
  - Added `custom_fp16_mul_mat_id` primitive (expert-id indexed gather+dot over cached expert weights).
  - Added Q5_K/Q6_K expert path in `QuantizedExpertWeights.__call__`:
    - build dequantized expert cache once (`_ensure_dequant_expert_cache`)
    - run `Tensor.mul_mat_id(..., fxn=custom_fp16_mul_mat_id)` for selected experts.
  - Keeps one primitive contract for expert paths; avoids materialized gathered intermediates in call-site DSL.
- `bench_block.py`
  - Updated split-point labels to the 3 focused boundaries above.

### Validation
- Syntax checks:
  - `python3 -m py_compile tinygrad/apps/mla.py tinygrad/apps/quantized.py bench_block.py search2.py`
- `bench_block` sanity:
  - `BEAM=0 bench_block.py 12`:
    - full block median `2.07 ms`
    - decode stack median `26.01 ms` (`38.44 tok/s` proxy)
    - feed_forward median `9.62 ms`
  - `BEAM=2 bench_block.py 12`:
    - full block median `1.13 ms`
    - decode stack median `27.10 ms` (`36.90 tok/s` proxy)
    - feed_forward median `11.04 ms`
- Split/BEAM co-search on new 3-bit boundaries:
  - `BEAM=2 search2.py --mode exhaustive --iters 6 --repeat 3`:
    - best mask `0x5` median `25.624 ms`
    - baseline `0x0` median `25.626 ms` (effectively tied)
  - `BEAM=0 search2.py --mode exhaustive --iters 6 --repeat 3`:
    - best mask `0x4` median `28.154 ms`
    - baseline `0x0` median `28.440 ms`

### Notes
- Q5_K/Q6_K expert primitive path correctness smoke:
  - zero-weight structured test matched fallback reference exactly for both types (`max_diff=0`).
- Random-byte Q5_K/Q6_K blocks can produce NaNs in dequant reference itself; structured tests were used for reliable parity checks.

## Q4 MoE raw MSL prototype (llama.cpp-style) (2026-02-12)

### Goal
- Build a standalone, correct, fast Metal kernel for decode MoE Q4_0 `mul_mat_id`
  using llama.cpp-style tricks, then compare against current hot kernels.

### Kernel design
- `ushort` packed reads (2 bytes -> 4 q4 values).
- mask/scale unpack (`1`, `1/16`, `1/256`, `1/4096`) + `-8*scale` bias (llama-style).
- reduction via `simd_sum` + one cross-simdgroup combine, not full tree barriers.
- one threadgroup per `(token_slot, out_feature)` and expert-id row selection in-kernel.

### Important correctness finding
- Initial validation against `custom_q4_0_mul_mat_id` was misleading on synthetic random blocks:
  - with decode-specialized opts active (`N=4, O=3072, I=2048`), custom path can diverge
    heavily from pure math on random packed values.
- Switched validation to pure tensor-math reference:
  - `scale[sel] * ((lo-8)*x_lo + (hi-8)*x_hi)` then reduce over `I`.
- Current MSL prototype matches math within fp16/reduction-order tolerance (`max_abs_diff ~ 0.5`).

### Commands
- MSL microbench:
  - `BEAM=0 .venv2/bin/python tinygrad/apps/bench_q4_moe_msl.py --iters 40 --threads 64`
- Thread sweep:
  - `BEAM=0 .venv2/bin/python - <<'PY' ... run_msl_kernel(...) ... PY`
- Real pipeline kernel timing snapshot:
  - `BEAM=0 DEBUG=2 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`

### Results (representative)
- Raw MSL kernel (math-checked):
  - gate/up shape (`N=4,O=3072,I=2048`): ~`99-106 us` median in repeated runs.
  - down shape (`N=4,O=2048,I=1536`): ~`69 us` median (stable in isolated repeats).
- Current tinygrad decode hot-kernel timings from `bench_block DEBUG=2`:
  - `custom_q4_0_mul_mat_id_4_3072_2048`: ~`144-147 us`.
  - `custom_q4_0_mul_mat_id_4_2048_1536`: ~`72-80 us`.

### Interpretation
- The raw MSL prototype is a real kernel-level win candidate for gate/up
  (roughly `1.4x` faster than current hot kernel in this snapshot).
- Down kernel is near-parity to mild win depending on launch shape.
- Main remaining gap is integration: benchmark script speed does not automatically
  translate to model tok/s until tinygrad can schedule/dispatch this path cleanly.

## Env-var integration trial: `QL_MOE_MSL=1` with `CompiledRunner` pattern (2026-02-12)

### Goal
- Wire raw Q4 MoE MSL path into runtime behind env var, using the old `CompiledRunner`/`ExecItem`
  capture pattern (same style as deleted `metal_q4k.py`) so it can participate in TinyJit capture.

### Code
- Added `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/q4_moe_msl.py`
  - `Q4MoEMSLRunner(CompiledRunner)` + cached source compile/runtime.
  - `q4_moe_mul_mat_id_msl(x, scale, packed, sel)` launches MSL via `ExecItem(..., prg=runner)`.
- Updated `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/quantized.py`
  - Q4 expert path now uses MSL when `QL_MOE_MSL=1` on Metal; otherwise existing `mul_mat_id` path.

### Correctness check
- Command:
  - `BEAM=0 .venv2/bin/python - <<'PY' ... q4_moe_mul_mat_id_msl vs pure tensor math ... PY`
- Result:
  - `max_abs_diff 0.5` (same fp16/reduction-order tolerance as prototype).

### Bench smoke A/B (`bench_block.py`, same command, sequential)
- MSL env:
  - `BEAM=0 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
  - full block median: `2.61 ms`
  - decode stack median: `38.30 ms` (`26.11 tok/s` proxy)
  - feed_forward median: `11.79 ms`
- Baseline:
  - `BEAM=0 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
  - full block median: `2.03 ms`
  - decode stack median: `25.25 ms` (`39.60 tok/s` proxy)
  - feed_forward median: `9.09 ms`

### DEBUG=2 confirmation
- Command:
  - `BEAM=0 QL_MOE_MSL=1 DEBUG=2 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
- New kernel is active (`q4_moe_mul_mat_id_msl_4_3072_2048`, `q4_moe_mul_mat_id_msl_4_2048_1536`), but observed times are high in pipeline:
  - gate/up: ~`209-214 us`
  - down: ~`125-130 us`

### Conclusion
- Keep as opt-in debug path only (env-var off by default).
- Rejected as performance path in current integration form.
- Key insight: `CompiledRunner` wiring alone is not sufficient; the in-pipeline kernel is much slower than the isolated prototype and harms end-to-end decode.

## MSL crash fix + verification pass (`PYTHONPATH=.`) (2026-02-12)

### Goal
- Fix the `QL_MOE_MSL=1` runtime crash (`AssertionError must be BUFFER Ops.CAST`) and re-verify whether MSL is faster.

### Code
- Updated `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/q4_moe_msl.py`:
  - Added explicit buffer-identity checks for `x/scale/packed/sel` with clear `RuntimeError` messages instead of raw internal assert.
- Updated `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/quantized.py`:
  - In Q4 env-var branch, call MSL only when `x_fp16` and `sel_flat` already have buffer identity.
  - Otherwise, clean fallback to existing `custom_q4_0_mul_mat_id` path.

### Validation commands
- `PYTHONPATH=. BEAM=0 .venv2/bin/python tinygrad/apps/bench_q4_moe_msl.py --mode like-for-like --iters 6 --warmup 3 --decode-warmup 1 --model "glm-4.7:flash-unsloth-Q4_0"`
- `PYTHONPATH=. BEAM=0 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
- `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
- `PYTHONPATH=. BEAM=0 DEBUG=2 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0" | rg "q4_moe_mul_mat_id_msl|custom_q4_0_mul_mat_id"`

### Results
- Like-for-like (real model-loaded tensors):
  - gate/up custom: `470.60 us` median
  - gate/up MSL: `303.37 us` median (`1.55x` faster)
  - down custom: `313.85 us` median
  - down MSL: `278.88 us` median (`1.13x` faster)
- `bench_block` smoke A/B (1-iter quick pass):
  - baseline decode proxy: `35.96 tok/s`
  - `QL_MOE_MSL=1` decode proxy: `34.18 tok/s`
  - no crash in either run.
- DEBUG=2 grep confirms env-var run currently executes `custom_q4_0_mul_mat_id_*` kernels (fallback path), not `q4_moe_mul_mat_id_msl_*`.

### Interpretation
- MSL kernel itself is still faster in like-for-like A/B when inputs are already concrete buffers.
- In current full decode graph shape, inputs often arrive as lazy non-buffer-identity UOps; the new guard correctly falls back, so the env-var path is safe but not yet a model-speedup path.
- Next integration work should make MSL callable from lazy graphs (or introduce an intentional boundary that pays for itself), rather than forcing per-call materialization.

## Forced MSL wiring in full decode + full model A/B (2026-02-12)

### Goal
- Make `QL_MOE_MSL=1` truly execute `q4_moe_mul_mat_id_msl` in full decode (not fallback), then measure full model tok/s.

### Code
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/quantized.py`
  - In Q4 env-var branch, materialize dynamic inputs needed by external runner:
    - `x_msl = x_fp16.contiguous().realize()` when `x_fp16` is not buffer-identity.
    - `sel_msl = sel_flat.cast(int)` if needed, then `contiguous().realize()` when not buffer-identity.
  - Then call `q4_moe_mul_mat_id_msl(x_msl, scale, packed, sel_msl)` directly.

### Wiring verification
- Command:
  - `PYTHONPATH=. BEAM=0 DEBUG=2 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0" | rg "q4_moe_mul_mat_id_msl|custom_q4_0_mul_mat_id"`
- Result:
  - `q4_moe_mul_mat_id_msl_4_3072_2048` and `q4_moe_mul_mat_id_msl_4_2048_1536` are now emitted repeatedly in decode path.
  - This confirms env-var path is truly wired in for real decode.

### Full model benchmark (sequential)
- Baseline:
  - `PYTHONPATH=. BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - steady decode lines: `24.98`, `24.17`, `23.96`, `24.95`, `23.41` tok/s.
- Forced MSL:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - steady decode lines: `18.17`, `18.30`, `18.01`, `18.12`, `18.13`, `18.53` tok/s.

### Conclusion
- MSL path is now truly wired in and active in full decode.
- End-to-end full model performance regresses materially (`~24-25 tok/s` -> `~18 tok/s`).
- Likely reason: per-call materialization boundaries needed by the external runner are expensive enough to outweigh kernel-level wins.

## MSL integration v2: remove explicit realize boundaries via CALL lowering (2026-02-12)

### Goal
- Keep MSL expert kernel in graph execution (no manual `.contiguous().realize()` in `quantized.py`) so dependencies are scheduled naturally.

### Code
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/quantized.py`
  - `QL_MOE_MSL=1` now dispatches through `Tensor.mul_mat_id(..., fxn=custom_q4_0_mul_mat_id_msl)`.
  - Removed explicit input materialization from model path.
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/q4_moe_msl.py`
  - Added `custom_q4_0_mul_mat_id_msl(out,x,scale,packed,sel)` that returns a SINK-tagged custom AST with shape metadata.
  - Added `lower_q4_moe_msl_ast(...)` to return cached `Q4MoEMSLRunner` from that AST.
- `/Users/rvd/src/rvd/tinygrad/tinygrad/engine/realize.py`
  - Added targeted lowerer hook for SINK+CUSTOM tag `q4_moe_mul_mat_id_msl` before generic SINK lowering.

### Validation
- Compile checks:
  - `PYTHONPATH=. .venv2/bin/python -m py_compile tinygrad/apps/quantized.py tinygrad/apps/q4_moe_msl.py tinygrad/engine/realize.py`
- Representative bench A/B (sequential):
  - Baseline: `PYTHONPATH=. BEAM=0 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
  - MSL v2: `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"`
- Kernel presence check:
  - `PYTHONPATH=. BEAM=0 DEBUG=2 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0" | rg "q4_moe_mul_mat_id_msl|custom_q4_0_mul_mat_id"`

### Results
- Baseline decode proxy: `36.89 tok/s`.
- MSL v2 decode proxy: `32.88 tok/s` (still slower).
- DEBUG=2 confirms MSL kernels are active in decode (`q4_moe_mul_mat_id_msl_4_3072_2048`, `q4_moe_mul_mat_id_msl_4_2048_1536`).

### Interpretation
- Removing explicit `.realize()` boundaries was necessary and fixed correctness/integration cleanliness.
- But even with in-graph lowering, current MSL kernel schedule still underperforms end-to-end vs tuned DSL/custom path on this workload.
- Next work must target kernel quality/launch strategy itself (or downstream interaction), not just integration plumbing.

## Representativeness gap and improved bench_block mode (2026-02-12)

### Why this was done
- We observed cases where `bench_block` predicted big gains that did not fully appear in full model tok/s.
- Goal: make `bench_block` more representative and quantify remaining gap.

### Code changes
- `/Users/rvd/src/rvd/tinygrad/bench_block.py`
  - Added `--real_blocks` mode (default on): load actual model MoE blocks via `Transformer.from_gguf(..., quantized=True, realize=False)`.
  - Added `--unique_blocks` decode-bank size (default `8`) and cycle through multiple unique blocks in decode stack.
  - Switched attention quant types in synthetic build path to GGUF-matched types from block 1 metadata.

### Measured deltas
- Full model (latest MSL integration, same code):
  - baseline steady decode ~`26.21 tok/s` median
  - `QL_MOE_MSL=1` steady decode ~`25.67 tok/s` median
  - net ~`-2%` (near parity/slight regression)
- Old synthetic-ish bench_block (`--real_blocks 0`, 8 unique synthetic blocks):
  - baseline `32.91 tok/s` proxy
  - MSL `39.89 tok/s` proxy
  - net `+21%` (too optimistic)
- New real-block bench_block (`--real_blocks 1 --unique_blocks 8`):
  - baseline `27.26 tok/s` proxy
  - MSL `29.93 tok/s` proxy
  - net `+9.8%`

### Interpretation
- `bench_block` is now much closer in absolute tok/s and less optimistic than before.
- Remaining gap indicates decode-stack-only proxy still misses full-model effects (non-MoE kernels, token pipeline interactions, cache/ICB dynamics across full graph).
- There is real kernel-level signal in MSL, but current end-to-end integration still leaves gains on the table in full model.

## MoE combine and weighting rewrite (accepted, 2026-02-12)

### Goal
- Reduce non-custom MoE overhead in decode by replacing generic reduction and moving probability weighting earlier where math permits.

### Code
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/mla.py`
  - Replaced `sum(axis=2)` MoE combine with explicit adds when `num_experts_per_tok == 4` (decode default), fallback to sum for other K.
  - Moved expert probability weighting before down projection:
    - old: `expert_out = down(gated)`, then `expert_out * probs`
    - new: `expert_out = down(gated * probs)`, then combine experts
  - Keeps math equivalent because down projection is linear.

### Bench-block validation (sequential)
- Commands:
  - `PYTHONPATH=. BEAM=0 .venv2/bin/python bench_block.py 3 --model "glm-4.7:flash-unsloth-Q4_0"`
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 .venv2/bin/python bench_block.py 3 --model "glm-4.7:flash-unsloth-Q4_0"`
- Results on current revision:
  - without MSL: decode proxy `31.45 tok/s`
  - with MSL: decode proxy `34.56 tok/s`

### Full model validation (sequential)
- Commands:
  - `PYTHONPATH=. BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Steady decode lines:
  - baseline: `25.84`, `26.69`, `26.58`, `28.29`, `27.27` tok/s
  - +MSL: `25.92`, `28.37`, `30.22`, `29.85`, `29.84` tok/s

### Conclusion
- This rewrite is a real end-to-end win.
- Current best observed steady-state is ~`30 tok/s` with `QL_MOE_MSL=1` on this revision.

## DSL primitive boundary A/B for MoE matmul-id (2026-02-12)

### Hypothesis
- `mul_mat_id` as a first-class primitive boundary (CALL to SINK/KernelInfo) should outperform a loose DSL composition that exposes gather+dequant+reduce tensors.

### A/B setup
- A (boundary): existing Q4 expert path in `QuantizedExpertWeights.__call__` using:
  - `Tensor.mul_mat_id(..., fxn=custom_q4_0_mul_mat_id)` (or MSL variant under env).
- B (loose): temporary rewrite to:
  - `scale = _q4_0_scale[sel]`, `packed = _q4_0_packed[sel]`,
  - unpack/dequant in Tensor DSL,
  - reduce with `.sum(axis=-1)`.
- Then restore A.

### Commands
- `PYTHONPATH=. BEAM=0 .venv2/bin/python bench_block.py 3 --model "glm-4.7:flash-unsloth-Q4_0"`
- `PYTHONPATH=. BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 8`

### Results
- `bench_block` decode proxy:
  - A boundary: `33.75 tok/s`
  - B loose: `28.92 tok/s`
  - delta: boundary is ~`+16.7%` faster.
- Full model (`--benchmark 8`) was consistent with boundary being at least not worse and generally slightly better in steady rows, but with much smaller delta than bench proxy.

### Conclusion
- Yes: the SINK/CALL-style primitive boundary translates directly to DSL and is materially faster for the MoE step than loose gather/dequant/reduce composition.
- Keep the primitive boundary path as production default.

## Big trial: llama-style online dequant for shared Q5_K/Q6_K linears (accepted, 2026-02-12)

### Why this was selected
- Recent profile data showed shared-expert linears are a major non-MoE hotspot:
  - `ffn_gate_up_shexp`: Q5_K `(3072, 2048)` in 46 MoE blocks/token.
  - `ffn_down_shexp`: Q6_K `(2048, 1536)` in 46 MoE blocks/token.
- Existing path dequantized full weights to fp16 cache (`x.linear(w.T)`), which is bandwidth-heavy versus online dequant.

### Code
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/qk_linear_msl.py`
  - Added two Metal runners with SINK/CUSTOM lowering tags:
    - `q5_k_linear_msl` for Q5_K shared gate/up.
    - `q6_k_linear_msl` for Q6_K shared down.
  - Kernels are decode-oriented matvecs with in-kernel dequant (llama-style).
- `/Users/rvd/src/rvd/tinygrad/tinygrad/engine/realize.py`
  - Added lowerer dispatch for tags:
    - `"q5_k_linear_msl"`, `"q6_k_linear_msl"`.
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/quantized.py`
  - Added `_blocks_cache` and `_ensure_blocks_cache` to keep raw quant blocks on device.
  - Added `QL_SHEXP_MSL=1` path in `QuantizedLinear.__call__` for:
    - Q5_K `(3072, 2048)`.
    - Q6_K `(2048, 1536)`.
  - Uses `Tensor.custom_kernel(...)` with the new SINK/CUSTOM lowering.
- `/Users/rvd/src/rvd/tinygrad/tinygrad/apps/mla.py`
  - `merge_gate_up_shared_expert` now initializes `merged._blocks_cache = None` for `QuantizedLinear.__new__` merged modules.

### Correctness checks
- Deterministic and random finite-block A/B:
  - `QL_SHEXP_MSL=0` vs `QL_SHEXP_MSL=1` outputs match exactly (`max_abs_diff = 0`) for both Q5_K and Q6_K target shapes.

### Bench-block A/B (sequential)
- Commands:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 3 --model "glm-4.7:flash-unsloth-Q4_0"`
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python bench_block.py 3 --model "glm-4.7:flash-unsloth-Q4_0"`
- Results:
  - baseline decode proxy: `34.04 tok/s`
  - +Q5/Q6 shared MSL decode proxy: `39.81 tok/s`
  - delta: `+16.9%`

### Full model A/B (sequential, expensive confirmation)
- Commands:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Steady decode lines (tokens 6-10):
  - baseline: `28.24`, `28.28`, `27.96`, `28.03`, `28.82` tok/s
  - +Q5/Q6 shared MSL: `30.99`, `31.34`, `30.43`, `30.22`, `30.95` tok/s
- Approx steady average:
  - baseline: `~28.27 tok/s`
  - +Q5/Q6 shared MSL: `~30.79 tok/s`
  - delta: `+8.9%`

### Notes
- DEBUG=2 confirms new kernels are active (`q5_k_linear_msl_*`, `q6_k_linear_msl_*`).
- `q6_k_linear_msl` shows occasional spikes in debug traces; despite that, end-to-end tok/s still improved in both proxy and full model A/B.

## DEBUG=6 kernel-source A/B: DSL vs MSL path (analysis, 2026-02-12)

### Goal
- Inspect generated kernels in `DEBUG=6`, compare DSL and MSL for the same representative run, and explain the performance gap.

### Commands (sequential)
- `PYTHONPATH=. BEAM=0 DEBUG=6 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0" > /tmp/bench_dsl_debug6.log 2>&1`
- `PYTHONPATH=. BEAM=0 DEBUG=6 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0" > /tmp/bench_msl_debug6.log 2>&1`

### High-level result
- DSL (`QL_MOE_MSL=0`, `QL_SHEXP_MSL=0`): decode proxy `12.18 tok/s`.
- MSL (`QL_MOE_MSL=1`, `QL_SHEXP_MSL=1`): decode proxy `37.78 tok/s`.

### Kernel timing evidence
- Q4 dense linear (same shape):
  - DSL `custom_q4_0_linear_1_2048_5120`: median `~2416 us`.
  - MSL `q4_0_linear_msl_1_2048_5120_nr1`: median `~98 us`.
  - ~`24.7x` faster on this kernel.
- Q4 dense linear (other shapes):
  - `1x5120x768`: DSL `~886 us` vs MSL `~52 us` (~`17x`).
  - `1x768x2048`: DSL `~391 us` vs MSL `~25 us` (~`15.7x`).
- Q4 MoE expert kernel:
  - DSL `custom_q4_0_mul_mat_id_4_3072_2048`: median `~294 us`.
  - MSL `q4_moe_mul_mat_id_msl_4_3072_2048_t32_nr1`: median `~206 us` (~`1.43x`).
- MSL-only shared expert kernels:
  - `q5_k_linear_msl_1_3072_2048_nr2`: median `~92 us`.
  - `q6_k_linear_msl_1_2048_1536_nr1`: median `~54 us`.

### Source-level cause (from DEBUG=6)
- DSL `custom_q4_0_linear_1_2048_5120` is effectively serial in-kernel over the full reduce domain:
  - nested loops `for Ridx2_0 in 0..159` then `for Ridx2_1 in 0..15`.
  - no `LOCAL/GROUP` opts applied (`opts_to_apply=()`).
  - scale decode from two bytes to half appears inside the inner loop expression, so scale work is repeated many times.
- MSL `q4_0_linear` uses explicit decode-specialized structure:
  - `THREADS=32`, each lane processes `br = tid; br < BPR; br += THREADS`.
  - per-block constants (`s`, `s16`, `s256`, `s4096`, `md`) are hoisted once per block.
  - packed quant read as `ushort` (`j += 2`) to decode 4 nibbles per load.
  - reduction uses `simd_sum(acc)`; one lane writes output.

### Conclusion
- Main gap is structural execution shape, not a small heuristic miss:
  - DSL Q4 dense path is scalarized/serial.
  - MSL path is SIMD-parallel with hoisted decode math and better packed-load usage.
- If we want DSL to close this gap, it needs the same primitive boundary + parallel reduction shape generation (without handwritten MSL), not minor expression tweaks.

## DSL catch-up from MSL insights: Q4 dense kernel (2026-02-12)

### Objective
- Use MSL findings to improve non-MSL DSL path on the main dense Q4 kernel (`1x2048x5120`) without handwritten backend code.

### Root cause found
- `custom_q4_0_linear` had `KernelInfo(..., opts_to_apply=())`.
- In tinygrad postrange, any non-`None` `opts_to_apply` bypasses heuristic optimization (`apply_opts`), so this effectively hard-disabled auto scheduling.

### Step 1: remove extra dequant work in DSL kernel (accepted)
- Change DSL custom kernel to consume pre-separated tensors:
  - `scale: (O, bpr, 1)` and `packed: (O, bpr, 16)`
  - instead of raw `(O, bpr, 18)` blocks with per-iteration fp16 scale unpack.
- Bench (`QL_MOE_MSL=0 QL_SHEXP_MSL=0`):
  - before: decode proxy `12.88 tok/s`
  - after separation: decode proxy `15.29 tok/s`

### Step 2: try enabling generic heuristic path directly (rejected)
- Removing `opts_to_apply=()` exposed optimizer/codegen issues for this kernel:
  - crash in `gpudims.py` (`ptrdtype` assumption on non-pointer INDEX base after vectorized path).
  - after guard fix, further SPEC failure on invalid vectorized register INDEX shape.
- Conclusion: generic heuristic path is not yet robust for this custom-reduce kernel form.

### Tinygrad core fix landed
- `tinygrad/codegen/gpudims.py`:
  - guarded global-store masking logic to only access `ptrdtype` when INDEX base is actually `PtrDType`.
  - fixes one real crash class discovered by this kernel.

### Step 3: explicit shape-specialized DSL opts (accepted)
- Added `_q4_0_linear_opts` and applied via `KernelInfo.opts_to_apply` for decode shapes.
- Tried `LOCAL+GROUP` first:
  - huge perf win but numerically wrong on microbench (`max_diff` enormous) -> rejected.
- Kept `LOCAL` only (safe):
  - final default: `LOCAL axis=0 arg=16` for `(1,2048,5120)`, `(1,5120,768)`, `(1,768,2048)`.

### Result
- Representative bench (`QL_MOE_MSL=0 QL_SHEXP_MSL=0`):
  - decode proxy improved to `31.71 tok/s` (typical range seen: ~31-32).
  - from baseline `12.88 tok/s` -> ~`2.46x` improvement.
- DEBUG=2 kernel metrics now show major lift for hot dense kernel:
  - `custom_q4_0_linear_1_2048_5120`: ~`130 GFLOPS` (was ~`21 GFLOPS` before this line of work).

### Production-path safety check
- MSL path unaffected by this DSL work:
  - `QL_MOE_MSL=1 QL_SHEXP_MSL=1` decode proxy remains ~`41.94 tok/s`.

### Fast-iteration tooling added
- Added focused microbench:
  - `tinygrad/apps/bench_q4_linear_dsl.py`
  - covers default GLM dense Q4 shapes, runs correctness check, and optional MSL compare.

## Isolated A/B: q4_0_linear LOCAL 32 -> 16 only (2026-02-12)

### Setup
- Before: clean worktree `/tmp/tg_ab_sKEl2y` (detached `e92f28fab`)
- After: current tree
- Only code diff in `tinygrad/apps/quantized.py`:
  - `_q4_0_linear_opts` for `(1,2048,5120)`, `(1,5120,768)`, `(1,768,2048)` changed `LOCAL(axis=0,arg=32)` -> `LOCAL(axis=0,arg=16)`
- Command (both sides, sequential):
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 20 --model "glm-4.7:flash-unsloth-Q4_0"`

### Result
- Before (LOCAL=32):
  - full block median: `1.08 ms`
  - decode stack median: `35.99 ms` (`27.78 tok/s`)
  - feed_forward median: `16.57 ms`
- After (LOCAL=16):
  - full block median: `1.05 ms`
  - decode stack median: `35.31 ms` (`28.32 tok/s`)
  - feed_forward median: `15.35 ms`

### Delta
- decode stack: `-1.89%` latency (`+1.94%` tok/s)
- feed_forward: `-7.36%` latency
- full block: `-2.78%` latency

### Decision
- Keep `LOCAL=16` for these three dense Q4 shapes.

## Retry pass (noise reduction): DSL schedule candidates (2026-02-12)

### Retried with stronger settings
- Command shape: `bench_block.py --model glm-4.7:flash-unsloth-Q4_0`
- Sequential only, with `BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0`.

### A/B 1 (count=20, rounds=2)
- Default (`QL_DSL_MOE_LOCAL_3072_2048=8`):
  - decode stack median: `32.76 ms`
  - decode proxy: `~30.5 tok/s`
- Candidate (`QL_DSL_MOE_LOCAL_3072_2048=4, QL_DSL_MOE_GROUP_3072_2048=16`):
  - decode stack median: `32.61 ms`
  - decode proxy: `~30.7 tok/s`
- Outcome: small decode win (`~0.46%`), but FF/full medians not clearly better.

### A/B 2 (count=10, rounds=2)
- `QL_DSL_Q4_LINEAR_LOCAL=16` + candidate MoE setting above:
  - decode stack median: `31.51 ms`
  - decode proxy: `~31.7 tok/s`
- This looked best in this retry batch, but still below MSL-path decode proxy.

### Core feature added for fast iteration
- `tinygrad/uop/spec.py`: allow `Ops.INDEX` with vectorized pointer source in linearized kernels.
- Purpose: unblock upcasted schedules for this DSL custom kernel family.

### Upcast trial after core feature (count=10, rounds=2)
- `QL_DSL_Q4_LINEAR_UPCAST=2`:
  - decode stack median: `32.53 ms`
  - decode proxy: `~30.7 tok/s`
- Outcome: stable and slightly better than default baseline, but not a step-change.

### Current status
- DSL improved and stabilized, but still materially behind MSL path (recent MSL decode proxy ~`36.6 tok/s` in same bench style).
- Remaining gap is still reduction/codegen shape for dense Q4 and MoE kernels.

## DSL iteration: re-test auto path + reduce-shape rewrite (2026-02-12, follow-up)

### Goal
- Continue DSL-vs-MSL work with the new spec/legalization changes.
- Re-test the previously blocked auto-scheduled DSL path and then improve the hot Q4 dense kernel without MSL.

### 1) Re-test: auto DSL schedule for `custom_q4_0_linear`
- Baseline DSL (safe fixed opts):
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 2 --model "glm-4.7:flash-unsloth-Q4_0"`
  - decode proxy: `30.47 tok/s` (same ballpark as recent)
- Auto attempt:
  - `... QL_DSL_Q4_LINEAR_AUTO=1 ...`
  - still fails, now at Metal compile (not SPEC): invalid emitted source (`type-id cannot have a name`, pointer-indirection errors).
- Conclusion:
  - `Ops.INDEX(VECTORIZE(ptr), idx)` legalization in spec is not sufficient.
  - Remaining blocker is renderer/linearizer output validity for upcasted/vectorized forms.

### 2) Controlled schedule sweep (safe path)
- `QL_DSL_Q4_LINEAR_LOCAL=16` and `=64` were both worse than local=32 for decode stack.
- `QL_DSL_Q4_LINEAR_LOCAL_AXIS=1` is invalid for this kernel family (`KernelOptError: local is for globals`).

### 3) Main DSL kernel rewrite (accepted)
- Changed `custom_q4_0_linear` reduction body from one-byte/2-value steps to two-byte/4-value steps:
  - reduce trip count: `bpr*16` -> `bpr*8`
  - identical math mapping across lanes, correctness preserved (`max_diff` unchanged in microbench).
- Representative bench:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 2 --model "glm-4.7:flash-unsloth-Q4_0"`
  - decode proxy improved: `30.40 -> 31.74 tok/s` (about +4.4%)
  - full-block median improved: `2.43 ms -> 1.47 ms` in this run pair.

### Notes
- Kept GROUPTOP path opt-in only (`QL_DSL_Q4_LINEAR_GROUPTOP_2048_5120=1`) due observed numerical instability on that schedule.
- Added inline comments in code to document why these choices exist and where instability remains.

## DSL follow-up iteration: core blocker triage and safe gains (2026-02-12)

### Retest: exact blocked path
- Command:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 QL_DSL_Q4_LINEAR_AUTO=1 .venv2/bin/python bench_block.py 2 --model "glm-4.7:flash-unsloth-Q4_0"`
- Result:
  - still fails in Metal codegen for upcasted/vectorized custom Q4 linear path.
  - SPEC no longer fails first; now failing stage is renderer output/compile.

### Root-cause inspection
- Dumped pre-compile source for `custom_q4_0_linear_1_2048_5120` with `QL_DSL_Q4_LINEAR_UPCAST=2`.
- Found malformed scalar/vector access in emitted C-style source (`(*(acc0+0)).x/.y` on scalar float and pointer-vector expressions).
- Added cstyle cleanup rewrite to collapse `VECTORIZE(ptr, ptr, ...)` -> scalar ptr lanes (`tinygrad/renderer/cstyle.py`).
- This moved the compile error forward but did not fully resolve upcast compile validity.

### Accepted DSL change in this iteration
- Kept two-byte reduction rewrite for `custom_q4_0_linear` only (dense Q4 path).
- Representative DSL bench stayed improved:
  - decode proxy around `31.3-31.7 tok/s` (vs ~30.4 baseline before this rewrite line).

### Rejected in this iteration
- Two-byte reduction rewrite for `custom_q4_0_mul_mat_id` (MoE Q4 expert):
  - required changing group geometry, then regressed FF and did not improve decode.
  - reverted to byte-wise original reduction + original group defaults.

### Current status
- DSL path improved but still behind MSL path on this machine.
- Main blocker to a bigger DSL jump remains upcast/vectorized custom-kernel codegen legality for the dense Q4 hotspot.

## DSL iteration (2026-02-12, continued): reduction-shape and router cleanup

### Accepted changes

1) Dense Q4 hot shape schedule refinement
- File: `tinygrad/apps/quantized.py`
- Change: `_q4_0_linear_opts` for `(N,O,I)=(1,2048,5120)`
  - from: `LOCAL(0,32) + GROUPTOP(1,4)`
  - to:   `LOCAL(0,16) + GROUPTOP(1,4)`
- Why: kernel-level DEBUG=2 median improved (`~148.5us -> ~141.5us`) for `custom_q4_0_linear_1_2048_5120`.
- End-to-end check:
  - `bench_block.py 3 --rounds 3` aggregate decode stack improved from ~`27.31 ms` (older baseline) to ~`27.21 ms` with dense tweak, and further with next tweak below.

2) MoE gate/up schedule: use GROUPTOP
- File: `tinygrad/apps/quantized.py`
- Change: `_q4_0_mul_mat_id_opts` for `(N,O,I)=(4,3072,2048)`
  - from: `LOCAL(1,8) + GROUP(1,8)`
  - to:   `LOCAL(1,8) + GROUPTOP(1,8)`
- Why: this is the largest DSL kernel by cumulative time; DEBUG=2 microbench median improved strongly (`~143.9us -> ~112.9us`) for `custom_q4_0_mul_mat_id_4_3072_2048`.
- End-to-end check:
  - `bench_block.py 3 --rounds 3` aggregate decode stack reached ~`27.05 ms` (decode proxy ~`36.95-36.99 tok/s` in rounds).

3) Remove unused top-k value gather in MoE routing path
- File: `tinygrad/apps/mla.py`
- Added: `_topk_pairwise_indices(scores, k)`
- Change in `_feed_forward`: use index-only topk helper, avoid computing/returning top-k values that are discarded.
- Why: old path did extra gather work in every MoE layer even though only indices are needed.
- A/B validation (`bench_block.py 3 --rounds 3`):
  - with index-only topk: aggregate decode stack ~`27.08 ms`
  - old value+index topk: aggregate decode stack ~`27.21 ms`
  - net: ~`0.13 ms` improvement on decode stack in this controlled A/B.

### Rejected changes (kept out)

- Dense Q4 hot shape `GROUPTOP(2)` variants (`LOCAL=8/32/64`) looked good in isolated microbench but regressed `bench_block` decode proxy.
- MoE down shape `(4,2048,1536)` switched to `GROUP(1,8)` looked faster in per-kernel timing but regressed aggregate decode in rounds A/B; reverted to `GROUPTOP(1,8)`.
- Routing Q5/Q6 dequant-cache linears through `custom_fp16_linear` increased kernel count (`1700`) and regressed decode proxy; reverted.

### Current retained DSL defaults after this iteration

- Dense Q4 `(1,2048,5120)`: `LOCAL(0,16) + GROUPTOP(1,4)`
- MoE gate/up `(4,3072,2048)`: `LOCAL(1,8) + GROUPTOP(1,8)`
- MoE down `(4,2048,1536)`: `LOCAL(1,8) + GROUPTOP(1,8)`
- MoE routing uses `_topk_pairwise_indices` in `_feed_forward`.

### Full-model check after retained DSL changes
- Command:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Steady decode lines from latest run:
  - `30.12, 30.54, 30.93, 30.64, 31.09 tok/s`
- Notes:
  - Bench-block proxy improved and stabilized in the `~36.9 tok/s` band.
  - Full-model remains around low-31 tok/s range with expected run-to-run variance.

## 2026-02-12: ideas2/ideas3 next-step validation

- Reviewed `/glm_context/ideas2.md` and `/glm_context/ideas3.md` against current DSL path.
- Confirmed highest remaining DSL gap is still dense Q4 shape `custom_q4_0_linear_1_2048_5120`.

### Fast probes

- `bench_q4_linear_dsl.py --case ffn_in --opts_mode auto`:
  - median ~619.67us, max_diff=0.5
- `bench_q4_linear_dsl.py --case ffn_in --opts_mode default`:
  - median ~401.96us, max_diff=0.5
- MoE primitive check (`bench_q4_moe_dsl.py --all --compare_msl`):
  - gate_up: DSL faster than MSL in this sample (~328us vs ~401us)
  - down: DSL slower than MSL (~316us vs ~282us)

### Accepted change

- File: `tinygrad/apps/quantized.py`
- Shape-tuned dense Q4 schedule update for `(1,2048,5120)`:
  - from `LOCAL(0,16)+GROUPTOP(1,4)`
  - to   `LOCAL(0,8)+GROUPTOP(1,4)`
- Rationale: microbench sweep found lower median on this hotspot while retaining numerical parity behavior.

### End-to-end check (DSL path)

- Command:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 2 --model "glm-4.7:flash-unsloth-Q4_0"`
- Before (recent anchor): decode proxy ~34.79 tok/s
- After: decode proxy ~35.31 tok/s
- Net: +0.52 tok/s on representative bench_block run.

### Next high-impact idea (from ideas3 Tier 1)

- Unblock fully legal auto-scheduled dense Q4 path for this kernel family by fixing vectorized reduction/legalization in core (`spec.py`, `linearizer.py`, `cstyle.py`) so we can safely beat fixed schedule defaults.

## 2026-02-13 — GLM DSL + DeepSeek generic pass

### Accepted

1) **Generic heuristic fallback for non-GLM Q4 shapes** in `quantized.py`:
- `_q4_0_linear_opts`: fallback changed from `()` to `None`
- `_q4_0_mul_mat_id_opts`: fallback changed from `()` to `None`

Why: `()` disables postrange heuristics entirely; `None` allows generic LOCAL/GROUP(TOP) scheduling.

DeepSeek full-model result:
- Before (steady): ~8.5 tok/s
- After (steady): ~50-52 tok/s

Command:
- `PYTHONPATH=. BEAM=0 .venv2/bin/python tinygrad/apps/llm.py --model "deepseek-v2-lite-Q4_0" --benchmark 10`

2) **GLM MoE shape schedule update** in `quantized.py`:
- `custom_q4_0_mul_mat_id_4_3072_2048`: `LOCAL(8), GROUPTOP(4)`
- `custom_q4_0_mul_mat_id_4_2048_1536`: `LOCAL(16), GROUPTOP(4)`

Microbench evidence (representative):
- `4x3072x2048`: ~338us -> ~299us
- `4x2048x1536`: ~331us -> ~302us

`bench_block.py` steady decode proxy in this window hovered around ~33.7-35.8 depending run noise; current stable sanity is ~33.8-34.1.

### Rejected / Reverted

1) Dense Q4 schedule change `2048x5120 -> GROUPTOP(8)`:
- Looked good in microbench, but full-model GLM DSL steady regressed to ~28.1-28.3 tok/s.
- Reverted to `GROUPTOP(4)`.

2) `custom_q4_0_linear_vec2` SIMD primitive attempt:
- Tried to legalize vector store path and reduce renderer address-space errors.
- Added Metal register pointer prefix support in `cstyle.py` (`reg_prefix_for_cast = "thread "`).
- Fixed pointer target to `out.index(..., ptr=True)` for vec2 store.
- Kernel now compiles, but output is numerically wrong (`max_diff` huge) and slower than scalar path.
- Not promoted into runtime path.

### Current status

- Active DSL path remains scalar `custom_q4_0_linear` + tuned opts.
- GLM DSL representative sanity (`bench_block.py 3`): decode proxy ~33.8 tok/s.
- DeepSeek major gain from generic fallback remains intact (~50 tok/s steady full model).


## 2026-02-13: DSL Q4 dense SIMD primitive iteration (sequential)

### Goal
- Close part of the DSL vs MSL gap on the largest dense Q4 kernel (`custom_q4_0_linear_1_2048_5120`) with a structural change, not tiny tuning.

### Key findings
- `custom_q4_0_linear_vec2` had latent speed but was numerically wrong.
- Root cause from `DEBUG=6`: grouped lanes (`GROUPTOP`) were writing partial sums because vec2 path used manual `acc.after(...)` accumulation instead of explicit `REDUCE`.

### Code changes (accepted)
- `tinygrad/apps/quantized.py`
  - Rewrote `custom_q4_0_linear_vec2` accumulation to explicit `REDUCE` over `(r_br, r_pair)`.
  - Added detailed in-code comment explaining why manual accumulator is wrong with grouped reductions.
  - Added `_q4_0_linear_vec2_opts` and specialized vec2 scheduling for `(1,2048,5120)` -> `LOCAL(8), GROUPTOP(8)`.
  - Routed dense Q4 DSL path to vec2 kernel only for exact decode-hot shape `(N=1, O=2048, I=5120)`.
  - Kept scalar path for all other shapes.
- `tinygrad/apps/quantized.py`
  - Mixed scheduling policy for scalar dense Q4 remains:
    - auto (`None`) for `(1,2048,5120)` and `(1,768,2048)`
    - fixed `(LOCAL(16), GROUP(1,8))` for `(1,5120,768)`

### Code changes (rejected / reverted)
- `tinygrad/renderer/cstyle.py` temporary addrspace cast hook (`reg_prefix_for_cast`) was reverted.
  - No longer needed after explicit-REDUCE vec2 fix.
  - Avoids global renderer risk.

### Microbench evidence
Command:
- `PYTHONPATH=. BEAM=0 .venv2/bin/python - <<'PY' ... custom_q4_0_linear vs custom_q4_0_linear_vec2 ... PY`

Results on `N=1,O=2048,I=5120` (same input, 100 iters):
- scalar-auto: median `390.13us` (`53.8 GFLOPS`)
- vec2-l8g8: median `374.31us` (`56.0 GFLOPS`)
- correctness: vec2 `max_diff` matched scalar tolerance (`<= 1.0` in fp16 compare runs)

### bench_block evidence (DSL mode)
Command:
- `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 5 --model "glm-4.7:flash-unsloth-Q4_0"`

Before this vec2 routing (earlier in this session, same mixed scalar policy):
- decode proxy around `33.69 - 33.73 tok/s`

After vec2 routing for `(1,2048,5120)`:
- run A: `34.15 tok/s`
- run B: `34.13 tok/s`
- later validation run: `33.99 tok/s`

Interpretation:
- Modest but real uplift vs pre-vec2 mixed baseline (roughly `+0.3` to `+0.4 tok/s` on repeat runs).
- Not a step change; main DSL gap to MSL still remains in reduction/codegen shape quality.

### MSL sanity (non-regression check)
Command:
- `PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python bench_block.py 2 --model "glm-4.7:flash-unsloth-Q4_0"`
- decode proxy observed: `41.18 tok/s` (within expected MSL band/no obvious breakage from DSL-only changes).

### Full model (DSL)
Command:
- `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`

Steady tail from this run:
- `27.22, 27.99, 28.01, 28.09, 28.20 tok/s`

Note:
- Full-model value remains below MSL and still volatile; bench-block continues to be the fast directional loop.

## Trial: Force materialization of weighted_gated (.contiguous()) — ACCEPTED

Date: 2026-02-13

### Problem
Down expert kernel fused with silu/swiglu/probs chain → `r_4_256_8_8_48_2_2` at 480us.
Root cause: `weighted_gated` had only ONE consumer (down expert), so scheduler fused everything.

### Change
`mla.py:153`: `weighted_gated = (gated * probs.unsqueeze(-1).cast(gated.dtype)).contiguous()`

### Results
- bench_block DEBUG=2: `r_4_256_8_8_48_2_2` gone, `custom_q4_0_mul_mat_id_4_2048_1536` appears (44 occurrences)
- bench_block: decode proxy 34→36.38 tok/s (+7%)
- **Full model: 27-28 tok/s → 30.3-30.5 tok/s (+11%)**
- DeepSeek non-regression: 54.5 tok/s (unchanged)

### Cleanup
Removed dead split-K code from quantized.py:
- `custom_q4_0_linear_split`, `_q4_0_linear_split_opts`, `_q4_0_linear_split_groups`
- `custom_q4_0_mul_mat_id_split`, `_q4_0_mul_mat_id_split_opts`, `_q4_0_mul_mat_id_split_groups`
- Split-K routing in `QuantizedLinear.__call__` and `QuantizedExpertWeights.__call__`
