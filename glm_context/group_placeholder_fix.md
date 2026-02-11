# GROUP Optimization Fix for UOp.placeholder (Register Accumulators)

## Problem

GROUP/GROUPTOP optimization produces **wrong results** when used with custom kernels
that use `UOp.placeholder` (register accumulators) via the `set()/end()` pattern.

Normal tensor reductions use `Ops.REDUCE`, which `fix_group_for_reduce` handles.
Custom kernels use `STORE + END` patterns instead — GROUP_REDUCE ranges end up on
the END node, but nothing rewrites them into the cooperative shared-memory reduction.

### Root Cause (three bugs)

**Bug 1: Missing pattern match** (`expander.py`) — `fix_group_for_reduce` only matches
`Ops.REDUCE`. Custom kernels produce `END(STORE(...), GROUP_REDUCE_range)`. No pattern
matcher handled this, so GROUP_REDUCE ranges were silently ignored — each thread did a
partial reduction and only one thread's result was kept.

**Bug 2: DEFINE_REG slot collision** (`__init__.py`) — `ReduceContext` starts `acc_num`
at 0. The placeholder's `DEFINE_REG` also uses slot 0. Due to UOp caching (same
op/dtype/arg = same object), the reduce accumulator and the placeholder register become
the **same UOp**. The final reduce adds on top of the partial sum instead of starting fresh.

**Bug 3: gpudims assumes all STOREs have PtrDType** (`gpudims.py`) — After GROUP rewrite,
some STOREs have non-pointer INDEX sources (e.g. `dtypes.float.vec(3)` from vectorized
register accumulators). `add_gpudims` calls `.ptrdtype` unconditionally → crash.

## Fixes

### 1. `fix_group_for_end_store` in `tinygrad/codegen/late/expander.py`

New pattern in `pm_group_for_reduce` matching `END(STORE(...), GROUP_REDUCE_range)`:

1. **Partial END** — each thread accumulates over non-GROUP_REDUCE ranges only
2. **Bufferize to LOCAL** — write per-thread partial results to shared memory
3. **REDUCE from LOCAL** — cooperative reduction across GROUP_REDUCE threads
4. **Store back to register** — write final result

Only fires for `AddrSpace.REG` stores (placeholder pattern).

### 2. ReduceContext initialization in `tinygrad/codegen/__init__.py`

```python
# Before: acc_num=0, collides with placeholder DEFINE_REG(arg=0)
ctx=ReduceContext()

# After: start from max existing slot + 1
existing_reg_slots = [u.arg for u in sink.toposort() if u.op is Ops.DEFINE_REG]
ctx=ReduceContext(acc_num=max(existing_reg_slots, default=-1)+1)
```

### 3. PtrDType guard in `tinygrad/codegen/gpudims.py`

```python
# Before: crashes on non-pointer STOREs
if r.op is Ops.STORE and (idx := r.src[0]).src[0].ptrdtype.addrspace == AddrSpace.GLOBAL:

# After: guard with isinstance check
if r.op is Ops.STORE and isinstance((idx := r.src[0]).src[0].dtype, PtrDType) and idx.src[0].dtype.addrspace == AddrSpace.GLOBAL:
```

## Current Status

- **Custom expert kernel** (`custom_q4_0_expert_matmul`): works with `GROUPTOP(0, 16)`,
  runs at **3.2 tok/s** (down from 20 tok/s baseline with tensor-ops path).
  The bottleneck is that GROUPTOP=16 only uses 16 threads for reduction — needs tuning.
- **Heuristic opts** (`opts_to_apply=None`): still broken — heuristic applies
  UPCAST/UNROLL to register accumulators creating `vec(48)` pointer types that fail
  spec verification. Heuristic needs to learn not to UPCAST register placeholders.

## Test Results

| Test | Status |
|------|--------|
| placeholder + no opts | PASS |
| placeholder + GROUP | PASS (was FAIL) |
| buffer-style + no opts | PASS |
| buffer-style + GROUP | FAIL (not in scope) |
| test_custom_kernel.py (18 tests) | ALL PASS |
| GLM baseline (no custom kernel) | 20 tok/s, PASS |
| GLM + custom expert + GROUPTOP(0,16) | 3.2 tok/s, PASS |

## Files Modified

- `tinygrad/codegen/late/expander.py` — `fix_group_for_end_store` + pattern in `pm_group_for_reduce`
- `tinygrad/codegen/__init__.py` — ReduceContext avoids DEFINE_REG slot collision
- `tinygrad/codegen/gpudims.py` — PtrDType guard on STORE iteration
- `tinygrad/apps/quantized.py` — `custom_q4_0_expert_matmul` kernel + use in `QuantizedExpertWeights`
- `test_group_placeholder.py` — 4 test cases

## Open Issues

1. **Performance**: Custom expert kernel at 3.2 tok/s vs 20 tok/s tensor-ops. GROUPTOP=16
   is too small (only 16 threads cooperate on reduction of 64 blocks). Need larger GROUP
   or different kernel structure.
2. **Heuristic + placeholder**: `opts_to_apply=None` crashes because heuristic applies
   UPCAST to register pointers → `vec(48)` spec failure. The heuristic doesn't know about
   `AddrSpace.REG` and tries to vectorize it.
3. **Reduce range splitting**: Scheduler splits `r = range(bpr*16)` into `br = range(bpr)`
   and `j = range(16)` via `r//16, r%16`. This means GROUPTOP targets `br` (size 64),
   not the full 1024. Amounts must divide 64 (not 1024).
