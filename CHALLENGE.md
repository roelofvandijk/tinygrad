# The GLM 2x Performance Challenge

**Goal**: Achieve 40+ tok/s on GLM-4.7-Flash (2x the 22 tok/s baseline)
**Current**: 21 tok/s, **1450 kernels/token**
**Status**: Dispatch-bound at theoretical ceiling

---

## The Problem

GLM generates **1450 kernels/token** vs llama.cpp's ~200. Each kernel has ~34us dispatch overhead:

```
1450 kernels × 34us = 49ms dispatch overhead alone
Theoretical ceiling: 1000ms / 49ms = 20.4 tok/s
```

**To reach 40 tok/s (25ms/token):**
- Need ~500 kernels/token (3x reduction)
- Target: 500 × 34us = 17ms dispatch + ~8ms GPU work = 25ms total

---

## What Was Accomplished (Feb 9, 2026)

### 1. MV Heuristic Dequant Chain Walker ✅

**Problem**: MV heuristic required `MUL(INDEX, INDEX)` pattern but quantized experts produced `MUL(dequant_chain(INDEX), INDEX)`, causing GROUP optimization to fail.

**Fix** (`tinygrad/codegen/opt/heuristic.py:63-76`):
```python
def find_index_through_chain(u: UOp, max_depth=5) -> UOp | None:
  """Walk backward through dequant chain (bitcast/bitwise ops) to find underlying INDEX."""
  if u.op is Ops.INDEX: return u
  if max_depth == 0: return None
  dequant_ops = {Ops.BITCAST, Ops.CAST, Ops.AND, Ops.OR, Ops.SHR, Ops.SUB, Ops.ADD, Ops.MUL}
  if u.op in dequant_ops and len(u.src) > 0:
    if (idx := find_index_through_chain(u.src[0], max_depth-1)) is not None:
      return idx
  return None
```

**Modified MV heuristic** (lines 87-95) to use the walker:
```python
wt = mulop.src[1 - act_i]
if wt.op is Ops.INDEX:
  weight_index = wt
  if not all(r in weight_index.src[1].get_idx().ranges for r in act_idx.ranges): break
elif (weight_index := find_index_through_chain(wt)) is not None:
  # Found INDEX through dequant chain - MV pattern applies
  if DEBUG >= 3: print(f"MATVEC: found INDEX through dequant chain")
  pass
else:
  continue  # Not an INDEX pattern we can optimize
```

**Result**:
- ✅ **Verified working**: `DEBUG=3` shows "MATVEC: found INDEX through dequant chain" firing on GLM
- ✅ GROUP=16 now applies to quantized expert matmuls
- ❌ **But**: Still hits ~50 GB/s ceiling from scattered byte reads (llama.cpp custom MSL gets 100+ GB/s)
- ❌ **Performance**: No net improvement (21 tok/s) - improved kernel quality offset by dispatch overhead

### 2. Buffer Fusion Limits Relaxed ✅

**Problem**: MoE expert aggregation (8 inputs → 1 output) was blocked by:
- >3 buffer limit (line 205: `if len(accessed_buffers) > 3`)
- 10x compression ratio requirement (line 218: `if out_in_ratio < 10`)

**Fix** (`tinygrad/schedule/rangeify.py:205-209, 220-222`):
```python
# Allow more buffers for reduce ops (MoE expert aggregation)
is_reduce_heavy = any(r.op is Ops.REDUCE for r in reduces)
max_bufs = 8 if is_reduce_heavy else 3
if len(accessed_buffers) > max_bufs and not (PCONTIG > 2): return None

# Lower threshold for multi-input reductions (MoE weighted sum)
threshold = 3 if len(accessed_buffers) > 2 else 10
if out_in_ratio < threshold: return None
```

**Result**:
- ✅ Allows 8-buffer MoE expert aggregation fusion
- ❌ **Performance**: Minimal impact - other barriers dominate

### 3. Performance Results

| Model | MUL_MAT_ID | tok/s | Kernels/tok | Notes |
|-------|------------|-------|-------------|-------|
| **DeepSeek-V2-Lite** | 0 (tinygrad) | 34 | ~806 | MV heuristic fix helped (was ~20) |
| **DeepSeek-V2-Lite** | 1 (custom MSL) | 44 | ~806 | Near 50 baseline, minimal headroom |
| **GLM-4.7-Flash** | 0 (tinygrad) | 21 | 1450 | MV heuristic working but dispatch-bound |
| **GLM-4.7-Flash** | 1 (custom MSL) | 21 | 1450 | Same as tinygrad path |

---

## The Fundamental Bottleneck: Scheduler Partitioning

From `glm_context/START.md` and `glm_context/bottlenecks.md`:

### The Catch-22

**Without `.contiguous()` barriers**: Scheduler sees entire 47-layer graph → explodes to **3600+ kernels** with terrible partitioning

**With `.contiguous()` barriers**: Scheduler partitions properly but creates **1450 kernels** from forced boundaries

**Why barriers exist** (from `tinygrad/apps/mla.py:139-148`):
```python
gate = self.ffn_gate_exps(sel, h_norm).silu().contiguous()  # break gate·up multireduce
up = self.ffn_up_exps(sel, h_norm).contiguous()             # lets MV heuristic match
expert_out = expert_out.contiguous()                        # separate kernel gets GROUP
shexp_gate = self.ffn_gate_shexp(h_norm).silu().contiguous() # Q5_K 89 GB/s vs 34 GB/s fused
```

Each barrier was **empirically tested** and provides:
- MoE down proj barrier: **+41%** (18→25.5 tok/s ds2)
- Shared expert barrier: **+17%** (18→20.9 tok/s GLM)
- Selective Q5K+Q6K split: **+27%** (11→14 tok/s GLM)

**Removing any barrier causes regression** - they're a local optimum.

### Why the Scheduler Can't Partition Large Graphs

The scheduler uses simple barrier-based partitioning:
1. `.contiguous()` and `.realize()` mark hard boundaries (ALWAYS_RUN_OPS)
2. Within each partition, schedule greedily
3. **No global cost model** - can't estimate if fusion will help or hurt

With 47 layers × 20 ops/layer = ~940 operations:
- Scheduler tries to fuse everything into mega-kernels
- Runs out of registers/local memory
- Falls back to naive splitting → 3600+ kernels

**Current 1450 kernels is the best the scheduler can do** with current heuristics.

---

## Failed Attempts at CONTIGUOUS Fusion

### Attempt 1: Remove CONTIGUOUS from ALWAYS_RUN_OPS

**Tried** (`tinygrad/schedule/rangeify.py:145`):
```python
# OLD:
ALWAYS_RUN_OPS = {Ops.CONTIGUOUS, Ops.COPY, Ops.ASSIGN, Ops.ENCDEC}

# NEW:
ALWAYS_RUN_OPS = {Ops.COPY, Ops.ASSIGN, Ops.ENCDEC}
CONTIGUOUS_FUSION_OPS = {Ops.REDUCE_AXIS, Ops.MUL, Ops.ADD, Ops.SUB, Ops.CAST}
```

**Modified** `cleanup_dead_axes` and `remove_bufferize` to check if source ops are fusible.

**Result**:
- ❌ **Test showed 7 kernels** for `(x + y).contiguous() * 2.0` (expected ≤2)
- ❌ Logic didn't work - CONTIGUOUS still acting as barrier
- ❌ **Reverted** all CONTIGUOUS fusion changes

**Why it failed**: `.contiguous()` marks buffer as `removable=False` at creation time, not just via ALWAYS_RUN_OPS check. Multiple enforcement points in the scheduler.

---

## How to Investigate Further

### Key Tools & Commands

#### 1. Profile Kernel Count
```bash
MUL_MAT_ID=0 timeout 60 .venv2/bin/python profile_model.py "glm-4.7:flash-Q4_0" 10
```
Shows:
- Kernels/token breakdown
- ICB batching structure
- Per-kernel categories (reduce, elementwise, etc.)
- Scheduling cache hits/misses

#### 2. Verify MV Heuristic
```bash
DEBUG=3 MUL_MAT_ID=0 timeout 60 .venv2/bin/python tinygrad/apps/llm.py \
  --model "glm-4.7:flash-Q4_0" --prompt "Hi" --count 2 2>&1 | grep "MATVEC"
```
Look for:
- `MATVEC: found INDEX through dequant chain` (dequant chain walker working)
- `MATVEC: k.full_shape=...` (MV optimization applied with GROUP/LOCAL/UPCAST)

#### 3. Test Fusion with Small Scripts
```bash
timeout 60 .venv2/bin/python test_fusion.py
```
Create minimal test cases for:
- Dequant chain → matmul (does MV heuristic fire?)
- Multi-input reduce (does 8-buffer fusion work?)
- CONTIGUOUS fusion (can we fuse through barriers?)

#### 4. Visualize Schedule
```bash
VIZ=1 timeout 60 .venv2/bin/python tinygrad/apps/llm.py \
  --model "glm-4.7:flash-Q4_0" --count 1
```
Generates graph visualization showing:
- UOp structure
- Fusion boundaries
- Where barriers are inserted

#### 5. Deep Dive with SPEC Validation
```bash
SPEC=2 MUL_MAT_ID=0 timeout 60 .venv2/bin/python tinygrad/apps/llm.py \
  --model "glm-4.7:flash-Q4_0" --count 3
```
Catches UOp spec violations that might indicate scheduler bugs.

### Key Files to Understand

| File | Purpose | Key Functions |
|------|---------|--------------|
| `tinygrad/schedule/rangeify.py` | Kernel fusion & partitioning | `remove_bufferize()` (fusion decisions), `cleanup_dead_axes()` |
| `tinygrad/engine/schedule.py` | Graph → kernel scheduling | `schedule_uops()` (main entry point) |
| `tinygrad/codegen/opt/heuristic.py` | Kernel optimization | `hand_coded_optimizations()` (MV, TC, GROUPTOP) |
| `tinygrad/apps/mla.py` | MLA+MoE implementation | `_feed_forward()` (lines 127-149, the hot loop) |
| `tinygrad/apps/quantized.py` | Quantized matmuls | `QuantizedExpertWeights.__call__()` (MoE dispatch) |

### Understanding the Scheduler Flow

1. **User code** → Tensor operations (lazy evaluation)
2. **`.schedule()`** → Convert tensor graph to UOp graph
3. **`rangeify.py`** → Insert BUFFERIZE nodes, try to remove them (fusion)
4. **`schedule.py`** → Linearize remaining kernels, build dependency graph
5. **`heuristic.py`** → Apply optimization passes (MV, TC, GROUPTOP)
6. **`codegen/`** → Generate device code (Metal MSL)
7. **`runtime/`** → Execute kernels via ICB batching

**Fusion happens in step 3** - `remove_bufferize()` decides if intermediate buffers can be eliminated.

### Debugging Fusion Decisions

Add debug prints to `rangeify.py:remove_bufferize()`:
```python
def remove_bufferize(src:UOp, buf:UOp, idx:UOp):
  if src.op in ALWAYS_RUN_OPS or not buf.arg.removable:
    print(f"KEEP buffer: src.op={src.op}, removable={buf.arg.removable}")
    return None

  # ... fusion logic ...

  if len(accessed_buffers) > max_bufs:
    print(f"KEEP buffer: {len(accessed_buffers)} buffers > {max_bufs} limit")
    return None
```

Run with small graph to see why specific buffers aren't fusing.

---

## Potential Solutions (Ranked by Feasibility)

### 1. **Accept Current Performance** (Easy)
- DeepSeek: 44 tok/s (88% of 50 baseline)
- GLM: 21 tok/s (95% of 22 baseline)
- **MV heuristic fix achieved its goal** (enable GROUP on quantized matmuls)
- Focus optimization elsewhere (other models, features)

### 2. **Custom MSL Kernels for GLM MoE** (Medium, Concrete)
Following llama.cpp's `mul_mat_id` approach:
- Fuse expert selection + dequant + matmul into single MSL kernel
- Coalesced memory access patterns (ushort4 reads)
- Hand-tuned SIMD reduction
- **Expected**: 100+ GB/s (vs current 50 GB/s ceiling)
- **Impact**: ~20% speedup on MoE kernels → 21→25 tok/s GLM
- **Still dispatch-bound**: 1450 kernels × 34us = won't reach 40 tok/s

### 3. **Scheduler Cost Model** (Hard, Research)
Replace barrier-based partitioning with cost-based:
```python
def should_fuse(op1: UOp, op2: UOp) -> bool:
  # Estimate kernel count with vs without fusion
  fused_kernels = estimate_schedule(fuse(op1, op2))
  split_kernels = estimate_schedule(op1) + estimate_schedule(op2)

  # Estimate GPU time with vs without fusion
  fused_time = estimate_gpu_time(fused_kernels)
  split_time = estimate_gpu_time(split_kernels) + DISPATCH_OVERHEAD

  return fused_time < split_time
```

Challenges:
- Requires predictive model of scheduler behavior
- Risk of infinite recursion (estimating schedule requires scheduling)
- Needs extensive benchmarking to calibrate

### 4. **Kernel Reordering for Concurrency** (Hard, Marginal)
llama.cpp uses N_FORWARD=8 lookahead to reorder independent kernels.

**Tested Feb 8**: ICB barrier removal with RAW/WAR/WAW conflict detection.
**Result**: **0% speedup** - nearly all kernels are data-dependent.

Only ~2 kernels/layer are independent (q_a and kv_a both read x_norm) = ~90 pairs total = ~3ms theoretical savings.

### 5. **Graph-Level Fusion Passes** (Very Hard, Research)
Pre-schedule optimization passes to reduce op count:
- Fuse norm+matmul (RMSNorm → mul in single kernel)
- Fuse QK+softmax (online softmax pattern)
- Fuse gather+matmul (expert routing + computation)

Requires:
- Pattern matching at tensor graph level
- Proving correctness of transformations
- Implementing fused primitives in codegen

**This is the "right" solution but requires months of work.**

---

## The Core Insight

**Dispatch overhead (49ms) dominates GPU compute (~8ms).**

Even with infinitely fast kernels:
```
Best case: 1450 kernels × 0us GPU + 49ms dispatch = 20.4 tok/s
```

**The only path to 40+ tok/s is reducing kernel count to ~500.**

But current scheduler can't partition large graphs without exploding to 3600+ kernels.

**The challenge**: Design a scheduler that can handle 47-layer graphs with ~940 operations and produce ~500 well-formed kernels instead of either 1450 (with barriers) or 3600+ (without barriers).

This is fundamentally a **graph partitioning problem** that needs a global cost model, not local heuristics.

---

## References

- `glm_context/START.md` - Performance baselines, what worked/failed
- `glm_context/bottlenecks.md` - Detailed gap analysis vs llama.cpp
- `glm_context/architecture.md` - MLA/MoE formulations, quantization
- `glm_context/tools.md` - Profiling commands, benchmarking rules
- `CLAUDE.md` - tinygrad architecture overview, testing workflows
- `MEMORY.md` - Project-specific learnings, JIT buffer view bug

---

## Test Cases for Validation

```bash
# Verify MV heuristic dequant chain walker
DEBUG=3 MUL_MAT_ID=0 timeout 60 .venv2/bin/python test_fusion.py

# Verify buffer fusion limits
# (should schedule fewer kernels for 8-input reduce)
timeout 60 .venv2/bin/python -c "
from tinygrad import Tensor
x = Tensor.randn(1, 1, 8, 128)
p = Tensor.randn(1, 1, 8, 1).softmax(2)
out = (x * p).sum(axis=2)
print(f'Kernels: {len(out.schedule())}')
out.realize()
"

# Benchmark full models
MUL_MAT_ID=0 timeout 60 .venv2/bin/python profile_model.py deepseek-v2-lite-Q4_0 20
MUL_MAT_ID=0 timeout 60 .venv2/bin/python profile_model.py "glm-4.7:flash-Q4_0" 10

# Smoke test correctness
SPEC=2 MUL_MAT_ID=0 timeout 60 .venv2/bin/python tinygrad/apps/llm.py \
  --model "glm-4.7:flash-Q4_0" --prompt "Hello" --count 5
```

---

## What Actually Matters

From 150+ hours of optimization work on GLM/DeepSeek/Youtu models:

1. **Kernel count >> kernel quality** - Reducing 1450→500 kernels (3x) matters more than doubling any individual kernel's bandwidth
2. **Dispatch overhead >> GPU compute** - 49ms dispatch vs 8ms GPU means 86% of time is wasted
3. **Barriers are necessary evils** - Every `.contiguous()` in mla.py was empirically tested and provides measurable benefit
4. **The scheduler is the bottleneck** - Not the heuristics, not the codegen, not Metal - the graph partitioning logic
5. **Step changes require fundamental work** - 1-2 tok/s noise is easy, 2x speedup requires rethinking core architecture

**The MV heuristic fix was valuable** (enables future work, helps DeepSeek) but **can't overcome dispatch overhead** in GLM.

To reach 40 tok/s on GLM: **Fix the scheduler's graph partitioning**, or accept that 21 tok/s is the current limit of the architecture.
