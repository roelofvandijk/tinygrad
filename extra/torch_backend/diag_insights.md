# torch.diag Test Failure Investigation

## Problem Statement
`test_diag_1d` fails with all zeros output instead of the expected diagonal matrix:
- Input: `[1, 2, 3]`
- Expected: `[[1,0,0],[0,2,0],[0,0,3]]`
- Actual: `[[0,0,0],[0,0,0],[0,0,3]]`

## Root Cause Analysis

### What torch.diag Does
When called with a 1D tensor, `torch.diag(a)` creates a 2D diagonal matrix. PyTorch decomposes this into:
1. `aten.diag_embed.default` - creates diagonal matrix
2. Which further decomposes to operations like `empty.memory_format`, `zero_`, masking/where operations

### The Issue
The problem is in the view realization system introduced in PR #9642. Here's the flow:

1. **torch.diag decomposition creates empty tensor**: `empty_strided([3,3])` returns all zeros
2. **Decomposition should fill diagonal**: Uses masking/where operations to set diagonal values
3. **View tracking breaks data flow**: At some point a view relationship is created with incorrect base
4. **`_copy_from` was calling `maybe_realize_storage`**: This tried to realize views BEFORE data assignment
5. **View realization fails**: `realize_with_views` tries to replay movement operations on realized tensors, but shapes don't match

### Key Discovery: realize_with_views Limitation

The `realize_with_views` function has a fundamental flaw for complex operations like diag:

```python
def realize_with_views(self: Tensor, views: list[Tensor]):
  self.replace(self.realize())  # Base becomes (3,3) realized tensor
  base_uop_set = set(self.uop.toposort())
  for v in views:
    ret = self
    for u in v.uop.toposort():
      if u in base_uop_set: continue
      if u.op in MOVEMENT_OPS: ret = MOVEMENT_OPS[u.op](ret, u)
    v.replace(ret)
```

**The problem**: After `self.replace(self.realize())`, the base tensor is realized with shape (3,3). But the UOps in the view's toposort contain operations like `RESHAPE (3,3) -> (1,)` which were designed for the LAZY computation graph, not for replaying on realized tensors.

Example from debug output:
```
[REALIZE_VIEWS] Base after realize: shape=(3, 3)
[REALIZE_VIEWS] Processing view 0/1, shape=(3,)
[REALIZE_VIEWS] Ops breakdown: 6 in base, 50 movement, 62 other
[REALIZE_VIEWS]   uop[4] RESHAPE shape=(1,) (MOVEMENT)
[REALIZE_VIEWS]     ret.shape before=(3, 3)
[REALIZE_VIEWS]     u.marg=(1,)
[REALIZE_VIEWS]     EXCEPTION: size mismatch, can't reshape ((3, 3)) -> ((1,))
```

You can't reshape 9 elements (3,3) into 1 element (1,). The UOp was designed to work in the lazy graph context, not on realized tensors.

## Fix Attempt #1: Remove maybe_realize_storage from _copy_from

**Change Made**:
```python
# OLD:
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  realize = dest.is_tiny and maybe_realize_storage(unwrap(dest))
  # ... later ...
  if realize: Tensor.realize(dest)

# NEW:
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  # Removed maybe_realize_storage call completely
  # Removed realize flag and Tensor.realize calls
```

**Result**: test_pad_circular_backward PASSES, but test_diag_1d still returns all zeros (no crash though).

**Why it still fails**: The decomposition creates an empty tensor and never properly fills it. Without forcing realization, the assign operations might not be working correctly, OR the decomposition itself isn't being executed properly.

## Current Status

### Working
- ✅ `test_pad_circular_backward` - passes after removing maybe_realize_storage
- ✅ No crashes in diag tests - exception handling prevents reshape errors
- ✅ View tracking system works for simple cases

### Broken
- ❌ `test_diag_1d` - returns all zeros instead of diagonal matrix
- ❌ View realization for complex decompositions

## Debug Output Analysis

With extensive logging (`/tmp/debug_diag.py`), we can see:

1. `torch.diag(a)` is called
2. Decomposition calls `aten.diag_embed.default`
3. `empty.memory_format` creates (3,3) zeros
4. `zero_` is called (redundant since already zeros)
5. **NO `_copy_from` calls are logged** - this is suspicious
6. Result is returned as all zeros

The missing `_copy_from` calls suggest the decomposition might be using a different mechanism to fill the diagonal (like in-place operations, where, or masking).

## Tinygrad's diag Implementation

For comparison, tinygrad's native `diag` does:
```python
def diag(self) -> Tensor:
  return self.unsqueeze(-1).pad((None,(0,n:=self.shape[0]))).flatten().shrink(((0,n*n),)).reshape(n,n)
```

This creates views through movement operations, which is why view tracking gets involved.

## Next Steps for Investigation

1. **Trace what operations diag_embed decomposition actually calls**
   - Enable TORCH_DEBUG=3 with proper backend loaded
   - Look for where/masking operations
   - Check if assign is being called

2. **Understand why _copy_from isn't called**
   - The decomposition must be using a different mechanism
   - Possibly in-place operations or direct assignment
   - Check torch decomposition source for diag_embed

3. **Verify assign() works correctly without maybe_realize_storage**
   - Create minimal test: empty tensor → assign values → read back
   - Check if view relationships interfere with assign

4. **Consider alternative approaches**:
   - Option A: Fix `realize_with_views` to handle shape-changing bases (hard!)
   - Option B: Prevent view tracking for certain operations (empty_strided?)
   - Option C: Implement diag_embed directly instead of using decomposition
   - Option D: Fix the assign flow to work with views correctly

## Files Modified

- `/Users/rvd/src/rvd/tinygrad/extra/torch_backend/backend.py`
  - Line 107: `maybe_realize_storage` - removed exception handling wrapper
  - Lines 359-376: `_copy_from` - removed `maybe_realize_storage` call and realize flags

## Key Code Locations

- `realize_with_views`: Line 96-105 in backend.py
- `maybe_realize_storage`: Line 106-108 in backend.py  
- `_copy_from`: Line 359-376 in backend.py
- `MOVEMENT_OPS`: Line 85-92 in backend.py
- View tracking: Lines 56-70 in backend.py
- Debug script: `/tmp/debug_diag.py`

## Testing Commands

```bash
# Run failing tests
PYTHONPATH=. .venv/bin/python -m pytest extra/torch_backend/test.py::TestTorchBackend::test_diag_1d -xvs

# Run with debug script
.venv/bin/python /tmp/debug_diag.py

# Run with torch debug (need to fix backend loading first)
TORCH_DEBUG=3 .venv/bin/python -c "import torch; import os; os.environ['TINY_BACKEND']='1'; ..."
```

## Related PR

PR #9642 introduced the view tracking and `realize_with_views` mechanism for strided tensor support. The mechanism works for simple view operations but breaks for complex decompositions like `torch.diag` where:
- Multiple views are created during decomposition
- View operations were designed for lazy graph, not realized tensors
- Shape transformations don't make sense when replayed on realized base

## Hypothesis for Next Investigator

The core issue is likely that **torch.diag's decomposition creates intermediate views that shouldn't be tracked as tinygrad views**. The `empty_strided` operation marks the output as a view (via wrap_view_op or similar), but it shouldn't be - it's a fresh allocation that will be filled with data.

Look into:
1. Why does empty_strided result in view tracking?
2. Should certain operations be excluded from view tracking?
3. Is there a way to "break" the view relationship after data is assigned?
