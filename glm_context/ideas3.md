# DSL Path: Highest ROI Changes (30 -> 40 tok/s)

## Current anchor
- Full model (DSL path): low-31 tok/s band.
- Recent steady lines: `30.12, 30.54, 30.93, 30.64, 31.09 tok/s`.
- Decode proxy is better (~37 tok/s), so there is still a model-integration gap.

## Scope
- DSL path only (no handwritten MSL required for the core wins below).
- Goal: prioritize changes that can realistically stack into +9 tok/s.

## Tier 1: highest ROI
1. Unblock auto-scheduled dense Q4 path (`custom_q4_0_linear`).
  - Why: this is the largest remaining DSL blocker; current journal calls out upcast/vectorized codegen legality as the main blocker.
  - Where: `tinygrad/renderer/cstyle.py`, `tinygrad/codegen/linearize.py`, `tinygrad/codegen/gpudims.py`, `tinygrad/uop/spec.py`.
  - Target: make `QL_DSL_Q4_LINEAR_AUTO=1` compile and produce legal kernels for hot shapes.
  - Expected gain: +2 to +4 tok/s if we can move dense Q4 kernels from fixed-safe schedules to valid vectorized schedules.

2. Keep `mul_mat_id` as a strict primitive boundary and optimize it, not loose dequant graphs.
  - Why: A/B already showed primitive boundary is ~16.7% faster than loose gather+dequant+reduce composition.
  - Where: `tinygrad/apps/quantized.py` (`custom_q4_0_mul_mat_id*` family), legalization/scheduling around CALL/SINK path.
  - Target: improve hot shapes `(4,3072,2048)` and `(4,2048,1536)` without breaking correctness.
  - Expected gain: +1.5 to +3 tok/s from better MoE expert kernel efficiency.

3. Add a robust DSL split-K/lane-parallel reduction path for hot Q4 kernels.
  - Why: MSL analysis showed structural parallel reduction shape is the big win; DSL should emulate that shape safely.
  - Where: `tinygrad/apps/quantized.py` (`custom_q4_0_linear_split`, `custom_q4_0_mul_mat_id_split`) plus scheduler/codegen legality fixes.
  - Target: default this only when end-to-end beats single-kernel grouped reduction on full-model runs.
  - Expected gain: +1 to +2 tok/s if split path can be made stable and legal.

## Tier 2: medium-high ROI
4. DSL online-dequant path for shared Q5_K/Q6_K (without relying on MSL).
  - Why: shared expert Q5/Q6 linears are still large buckets; fp16 dequant-cache path is bandwidth-heavy.
  - Where: `tinygrad/apps/quantized.py` (Q5_K/Q6_K `QuantizedLinear` path), possibly new DSL custom kernels.
  - Target: avoid full fp16 weight materialization in decode-hot shared expert shapes.
  - Expected gain: +1.5 to +3 tok/s if we can land a stable DSL path.

5. Reduce non-MoE kernel overhead via scheduler fusion that is already proven safe.
  - Why: full model still pays heavy dispatch overhead; proxy gains do not fully carry to model tok/s.
  - Where: `tinygrad/schedule/indexing.py`, related reduce+broadcast fusion paths.
  - Target: cut kernel count in full decode graph without local-memory overflow regressions.
  - Expected gain: +1 to +2 tok/s from dispatch overhead reduction.

## Tier 3: lower but cheap
6. Fold per-token attention algebra at load time.
  - Why: removes repeated per-token matmul work in MLA path.
  - Candidates:
    - fold `q_nope @ attn_k_b^T` into Q projection weights.
    - fold `(attn @ v_b^T) -> attn_output` chain where algebraically valid.
  - Where: `tinygrad/apps/mla.py` (+ weight loading wiring).
  - Expected gain: +0.5 to +1.5 tok/s.

## What not to spend cycles on now
- Global GROUPTOP threshold raises (already caused severe regressions).
- Broad BEAM-based schedule search as default policy (high variance, weak carryover).
- Reverting to loose gather/dequant composition for MoE experts.

## Execution order
1. Fix dense Q4 auto-path legality first (biggest blocker, highest ceiling).
2. Improve Q4 `mul_mat_id` hot-shape schedules (primitive path only).
3. Land split-K DSL path only if full-model A/B confirms gain.
4. Tackle Q5/Q6 shared DSL online-dequant.
5. Finish with scheduler kernel-count reductions and attention algebra folding.

## Acceptance gates (must pass)
- `bench_block.py` (real blocks mode), sequential runs only.
- Full-model check:
  - `PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10`
- Keep only changes that improve full-model steady-state, not microbench-only wins.
