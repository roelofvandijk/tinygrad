# Sign-Extend Shift Optimization Context

## Branch: sign-shift-opt
## Commit: 3f93c3f43

## Optimization Summary
Pattern: `(x<0).where(c, 0)` → `(x >> 31) & c` for signed ints

### Rationale
- Replaces conditional/branch with pure arithmetic
- `x >> 31` for signed int32 gives -1 (all 1s) when x<0, 0 otherwise
- `-1 & c = c`, `0 & c = 0`
- Result: branchless sign-conditional masking

## Files Changed

### tinygrad/uop/ops.py (lines 755-762)
```python
if self.op is Ops.AND and dtypes.is_int(self.dtype) and s1_vmin == s1_vmax >= 0:
  if s0_vmin >= 0: return min(0, s0_vmin), min(s0_vmax, s1_vmax)
  if s0_vmin == -1 and s0_vmax == 0: return 0, s1_vmax  # sign-mask: -1&c=c, 0&c=0
```
Added bounds handling for sign-mask pattern where operand is -1 or 0.

### tinygrad/uop/symbolic.py (lines 65-68)
```python
# (x<0).where(c, 0) -> (x >> 31) & c for signed ints, avoids branch
((UPat.var("x", dtype=dtypes.sints)<0).where(UPat.cvar("c", vec=False), UPat.const(None, 0)).named("w"),
 lambda x,c,w: (x >> (x.dtype.itemsize*8-1)) & c.arg if c.arg > 0 else None),
```

### test/unit/test_uop_symbolic.py
```python
def test_sign_extend_shift(self):
  a = Variable("a", -8, 8, dtypes.int)
  self.helper_test_variable((a<0).where(UOp.const(dtypes.int, 255), UOp.const(dtypes.int, 0)), 0, 255, "((a>>31)&255)", test_z3=False)
  self.helper_test_variable((a<0).where(UOp.const(dtypes.int, 15), UOp.const(dtypes.int, 0)), 0, 15, "((a>>31)&15)", test_z3=False)
```

## Benchmark Results (METAL)
| Model | Before | After | Improvement |
|-------|--------|-------|-------------|
| resnet50 | 290.74ms | 290.46ms | -0.1% |
| openpilot | 908.74ms | 889.82ms | -2.1% |
| efficientnet | 296.16ms | 278.31ms | -6.0% |
| shufflenet | 268.98ms | 255.90ms | -4.9% |
| dm | 446.38ms | 416.75ms | -6.6% |

## Key Technical Details
- Shift amount: `x.dtype.itemsize*8-1` (31 for int32, 63 for int64, etc.)
- Only applies when c.arg > 0
- Requires bounds fix: AND with sign-mask operand (vmin=-1, vmax=0) yields [0, c]
- Test with `test_z3=False` (z3 doesn't handle this transformation)

## Kernel Output Example
Before: `((val0<0)?255:0)`
After: `((val0>>31)&255)`

## Commands
```bash
# Run test
python -m pytest test/unit/test_uop_symbolic.py::TestSymbolic::test_sign_extend_shift -xvs

# Benchmark
PYTHONPATH=. METAL=1 CAPTURE_PROCESS_REPLAY=0 python3 test/external/external_model_benchmark.py

# See generated kernel
METAL=1 DEBUG=4 python -c "from tinygrad import Tensor, dtypes; x=Tensor([-5,-1,0,1,5],dtype=dtypes.int); y=(x<0).where(255,0); y.tolist()"
```
