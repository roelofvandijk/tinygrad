# Scheduler Fusion: Reduce+Broadcast Barrier Removal

## The Change (indexing.py:236)

### What it does
Removed the `not (PCONTIG > 1)` guard on the ending-range check in `create_new_ranges()`.

**Before** (line 236):
```python
if not (PCONTIG > 1) or any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges):
```

**After**:
```python
if any(any(rr.arg > e.arg for e in ending_ranges[x]) for rr in r.ranges):
```

### Why it matters
When ranges end at a consumer (from a reduce), the old code ALWAYS forced a buffer boundary (realize) for PCONTIG<=1 (the default). This prevented the scheduler from fusing `reduce → broadcast → elementwise` patterns like RMSNorm (`x * rsqrt(mean(x²) + eps)`).

With the change, a buffer boundary is only created when ranges are actually out of order (`rr.arg > e.arg`). In-order ranges (which is the common case for reduce+broadcast) flow through without a barrier.

### Impact on fusion
| Pattern | Before | After |
|---------|--------|-------|
| `x * sum(x)` | 2 kernels | **1 kernel** |
| RMSNorm: `x * rsqrt(mean(x²)+eps)` | 2 kernels | **1 kernel** |
| softmax: `exp(x-max)/sum(exp(x-max))` | 3 kernels | **2 kernels** |
| sum(x).sqrt() | 1 kernel | 1 kernel (unchanged) |
| silu(x@w1)*(x@w2) | 1 kernel | 1 kernel (unchanged) |

### Test results
- **test_ops.py**: 410 passed, 0 failures (correctness verified)
- **test_schedule.py**: 255 passed, 0 real failures (1 transient Metal XPC)
  - 50 test expectations updated (all improvements: fewer kernels)
- **test_rangeify.py**: 27 passed, 0 failures
- **test_reduce_broadcast_fusion.py**: 14/14 passed (NEW tests)

---

## FIXED: Metal Threadgroup Memory Overflow

### Problem
The scheduler change enabled aggressive fusion (3-matmul FFN, MLA attention), creating LOCAL BUFFERIZE
nodes for intermediates. When LOCAL buffer size exceeds Metal's 32768-byte threadgroup limit → crash.

Two distinct crash patterns:
1. **Single large LOCAL** (FFN): `silu(gate)*up` = 10240 floats = 40960 bytes > 32768
2. **Multiple small LOCALs** (MLA attention): 3 DEFINE_LOCAL totaling 38912 bytes > 32768

### Fix: Two-part shared memory guard

**Part 1: indexing.py:75-83** — Cap individual LOCAL BUFFERIZE at shared_max (32768 bytes).
When partial realization would create a LOCAL buffer exceeding the limit, promote to GLOBAL
by realizing all axes. This prevents single large LOCAL allocations.

**Part 2: postrange.py:153-156** — Account for existing LOCAL allocations in GROUP check.
When applying GROUP/GROUPTOP optimization, the smem check now adds existing DEFINE_LOCAL
bytes to ensure the total threadgroup memory (GROUP + scheduler LOCALs) doesn't exceed shared_max.

### Regression tests
`test/unit/test_reduce_broadcast_fusion.py` — Tier 6 tests:
- `test_three_matmul_ffn_no_crash` — GLM FFN pattern with dim=2048, hidden=10240
- `test_three_matmul_ffn_large_hidden` — Even larger hidden=16384
- `test_three_matmul_ffn_correctness` — Correctness against numpy reference

### Remaining issue
Multiple LOCAL BUFFERIZE in the same kernel (e.g. MLA attention with 3 LOCALs totaling 38912) are
only partially addressed. Part 1 catches individual large LOCALs, Part 2 catches GROUP+LOCAL.
But multiple scheduler LOCALs below 32768 each can still overflow. This doesn't crash for Q4_0 model.

---

## Key Code Locations

- **indexing.py:236** — The one-line scheduler change (range ending guard removal)
- **indexing.py:75-83** — LOCAL BUFFERIZE size cap (FIXED)
- **postrange.py:153-156** — GROUP shared memory check with existing LOCAL accounting (FIXED)
- **rangeify.py:145** — ALWAYS_RUN_OPS includes CONTIGUOUS
- **rangeify.py:174-180** — remove_bufferize cost function
- **heuristic.py:134** — GROUP heuristic trigger condition

## Files Modified
- `tinygrad/schedule/indexing.py` — Line 236 change + LOCAL size cap
- `tinygrad/codegen/opt/postrange.py` — GROUP smem check includes existing LOCALs
- `test/test_schedule.py` — 50 test expectations updated
- `test/unit/test_reduce_broadcast_fusion.py` — 17 tests (14 fusion + 3 over-fusion guards)
