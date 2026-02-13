# tinygrad LLM Optimization How-To

This document is a handover playbook for pushing token throughput up without losing correctness.

## 1) Goal and non-goals

Goal:
- Increase end-to-end decode speed (tok/s), not just individual kernel speed.

Non-goals:
- Local microbench wins that do not translate to model-level wins.
- Adding many runtime knobs/env vars for temporary tuning.

## 2) Operating rules

- Run benchmarks sequentially only. Never run benchmark commands in parallel.
- Change one variable at a time.
- Keep one production path. Avoid optional experimental branches in runtime logic.
- Keep GLM performance while improving generic behavior where possible.
- If a change is not a clear end-to-end win, revert it.
- Any time a change fails end-to-end, write down the missing piece before moving on.

## 3) Benchmark ladder (fast -> expensive)

Always validate in this order:

1. Microbench for the changed kernel
2. `bench_block.py` (representative decode proxy)
3. Full model `llm.py --benchmark`

Use full model only when bench-block evidence is positive.

## 4) Canonical commands

Representative DSL run:
```bash
PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 5 --model "glm-4.7:flash-unsloth-Q4_0"
```

Representative MSL sanity:
```bash
PYTHONPATH=. BEAM=0 QL_MOE_MSL=1 QL_SHEXP_MSL=1 .venv2/bin/python bench_block.py 2 --model "glm-4.7:flash-unsloth-Q4_0"
```

Kernel source/timing inspection:
```bash
PYTHONPATH=. BEAM=0 DEBUG=6 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python bench_block.py 1 --model "glm-4.7:flash-unsloth-Q4_0"
```

Full model confirmation:
```bash
PYTHONPATH=. BEAM=0 QL_MOE_MSL=0 QL_SHEXP_MSL=0 .venv2/bin/python tinygrad/apps/llm.py --model "glm-4.7:flash-unsloth-Q4_0" --benchmark 10
```

## 5) Decision policy

Keep a change only if all are true:
- Correctness remains within expected tolerance.
- `bench_block.py` improves with repeat runs (not a one-off spike).
- No clear regression in the non-target path (for us: MSL sanity remains in expected band).

Reject/revert when:
- Microbench wins but bench-block does not move.
- Kernel-level GFLOPS rises but decode tok/s does not rise.
- Correctness becomes shape- or schedule-dependent.

## 6) Failure diagnosis: "what is the missing piece?"

When end-to-end does not improve, classify the miss:

1. Wrong bottleneck
- You optimized a non-dominant kernel.
- Fix: re-rank kernels by cumulative wall time from DEBUG output.

2. Local win, system loss
- A faster kernel increases launches, barriers, or sync overhead.
- Fix: check kernel count and schedule shape; reduce fragmentation.

3. Legality/correctness blocker
- Auto schedule is fast but numerically wrong.
- Fix: legalize transform in core/spec/renderer instead of pinning weak manual opts.

4. Cache/shape instability
- Too many variant shapes cause cache churn.
- Fix: stabilize decode shapes and reduce shape-special-case explosion.

5. Memory path mismatch
- Better ALU schedule but poor memory access pattern.
- Fix: target packed loads, contiguous reads, and hoisted per-block constants.

6. Fusion boundary mismatch
- Extra intermediates/materialization destroy expected wins.
- Fix: inspect graph/schedule; move primitive boundary to where data reuse is highest.

## 7) Kernel inspection workflow

For a candidate kernel:
- Compare DSL vs MSL generated source in DEBUG=6.
- Check whether reductions are explicit and legally grouped.
- Confirm lane-level parallel structure exists where expected.
- Check if hot-loop index math/div-mod and scale unpack are hoisted.
- Verify throughput metrics (GFLOPS/GB/s), then decide by tok/s.

## 8) Refactoring priorities (not tuning)

Prefer structural improvements over opt fiddling:
- Introduce/adjust primitive boundaries for dominant operations.
- Make reduction shape legal and parallel in core pipeline.
- Reduce materialization and launch count in decode path.
- Replace ad-hoc env tuning with default-fast behavior.

## 9) Documentation discipline

For every trial, append to `glm_context/optimization_journal.md`:
- Exact command
- Before/after numbers
- Keep/revert decision
- Missing-piece diagnosis when rejected

Also add short, explicit comments in code for accepted non-obvious choices.

## 10) Minimal handover checklist

Before handing over a branch:
- Current best command and measured tok/s band
- Top 3 kernel hotspots by cumulative time
- Accepted changes and why
- Rejected changes and missing-piece diagnosis
- Next highest-impact hypothesis with one concrete test command
