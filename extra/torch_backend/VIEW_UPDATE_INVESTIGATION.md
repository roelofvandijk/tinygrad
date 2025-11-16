# PyTorch Backend View Update Investigation

## Problem Summary

When implementing in-place operations for the PyTorch backend using tinygrad, we encountered infinite loops when trying to update views after modifying the base tensor.

## Root Cause

**The infinite loop has TWO separate causes at different stages:**

### Cause 1: Using `__setitem__` in `_view_write` 
Original code: `flat_base[idx_view] = flat_val`

This caused infinite loops because:
1. `flat_base` is a tinygrad tensor (a view created by `.reshape()`)
2. Tinygrad's `__setitem__` calls `realize()` internally
3. When `realize()` is called, it can trigger tensor operations
4. These operations go back through PyTorch's dispatch (because tensors are wrapped)
5. PyTorch dispatch → tinygrad operation → realize → PyTorch dispatch → infinite recursion

**Solution:** Use `scatter()` instead, which is a pure tinygrad operation that doesn't go through `__setitem__`

### Cause 2: Modifying `.uop` on wrapped tensors
After fixing Cause 1, we tried: `v.uop = updated.uop` to update derived views

This caused infinite loops on the NEXT operation:
1. We modify the `.uop` attribute of tinygrad tensors wrapped by PyTorch
2. PyTorch's C++ wrapper has cached state that becomes stale
3. On the next operation, PyTorch's dispatch gets confused by the modified state
4. This causes infinite recursion in PyTorch's internal dispatch mechanism

**Solution:** Don't update derived views at all - accept the limitation that views become stale

## Investigation Timeline

### 1. Initial Approach: Using `replace()`
- Called `tensor.replace(new_value)` to update tensors in-place
- This worked initially but caused hangs when views were involved
- **Issue**: `replace()` changes the UOp ID, but PyTorch tensor wrappers still reference the old UOp
- When PyTorch tries to dispatch operations on the stale wrapper, it causes issues

### 2. Attempt: Direct `__setitem__`
```python
flat_base[idx_view] = flat_val
```
- **Issue**: This is the ORIGINAL CODE that caused infinite loops
- Tinygrad's `__setitem__` calls `realize()`, which triggers tensor operations
- These operations are dispatched through PyTorch (because tensors are wrapped)
- Creates circular dispatch: PyTorch → tinygrad → realize → PyTorch → ...
- Result: Infinite loop/hang in `b += 1` after `a += 1` when views exist

### 3. Solution for Write: Using `scatter()`
```python
result = flat_base.scatter(0, idx_view, flat_val)
base.assign(result.reshape(base.shape))
```
- **Success**: Avoids `__setitem__` and works without hanging
- Uses tinygrad's `scatter()` operation to build the updated tensor
- Then uses `assign()` to update the base tensor

### 4. Critical Discovery: View Update Problem

After updating the base tensor, we need to update derived views. We tried:

```python
for v in derived_views(base):
    v.uop = _replay_view_from_base(base, v).uop
```

**This causes infinite loops in PyTorch's dispatch!**

#### Why it breaks:
1. PyTorch wraps tinygrad tensors in C++ storage objects
2. When we do `v.uop = new_uop`, we modify the tinygrad tensor
3. But the PyTorch wrapper still points to the same C++ object
4. The C++ object's internal state becomes inconsistent
5. Next operation on that PyTorch tensor causes dispatch to loop infinitely

### 5. Testing Results

**Without view updates (`v.uop = ...` commented out):**
- ✅ No hangs
- ✅ Basic operations work
- ❌ Views don't reflect changes to base tensor

**With view updates (`v.uop = ...` enabled):**
- ❌ Infinite loop in PyTorch dispatch
- Hangs occur AFTER the operation returns, during next dispatch

## Current Solution

```python
def _apply_inplace(target: Tensor, value: Tensor) -> None:
  base = canonical_base(target)
  _view_write(base, target, value)
  # NOTE: We cannot update derived views by modifying their .uop because it breaks PyTorch's wrapper
  # Views will be stale after in-place ops on the base - this is a known limitation
```

### What works:
- ✅ In-place operations on base tensors
- ✅ In-place operations on views update the base tensor correctly via `scatter()`
- ✅ No hangs or infinite loops

### What doesn't work:
- ❌ Other views of the same base tensor don't automatically update
- ❌ Example: After `a[1:].zero_()`, existing view `b = a.view((2,2))` won't show zeros

## Why Direct UOp Modification Fails

The C++ wrapping layer in `wrapped_tensor.cpp` creates a mapping between PyTorch tensors and tinygrad tensors. When we modify `.uop` directly:

1. The tinygrad Tensor object's `uop` attribute changes
2. But the C++ wrapper's pointer/reference doesn't know about this change
3. PyTorch's autograd/dispatch system gets confused
4. Subsequent operations enter infinite dispatch loops

## Proper Solution (Not Implemented)

To properly support view updates, we would need to:

1. **Share storage at C++ level**: All views should point to the same C++ wrapped storage
2. **Update wrapper on assign**: When base tensor is updated, notify all wrappers
3. **Lazy view updates**: Don't update view UOps immediately, let them be recomputed on access

This requires significant changes to the C++ wrapper layer.

## Lessons Learned

1. **Never use `__setitem__` in backend code** - it triggers `realize()` which can cause circular dispatch
2. **Never modify `.uop` on tensors wrapped by PyTorch** - it breaks PyTorch's wrapper state causing infinite dispatch loops
3. **Avoid `replace()` on wrapped tensors** - changes UOp ID but wrapper still references old state
4. **Use `scatter()` for index-based writes** - it's a pure tinygrad operation that doesn't trigger dispatch  
5. **View tracking with `_view_base` works correctly** - the infrastructure is sound
6. **The hang happens in PyTorch's dispatch, not our code** - it's a wrapper state consistency issue
7. **Two different root causes** - one during the write operation (`__setitem__`), one after modifying views (`.uop` assignment)

## Code Pattern That Works

```python
# For in-place write to a view:
def _view_write(base: Tensor, view: Tensor, value: Tensor) -> None:
  # Use scatter to avoid __setitem__
  idx_base = Tensor.arange(base.numel(), ...).reshape(base.shape)
  idx_view = _replay_view_from_base(idx_base, view).reshape(-1)
  flat_base = base.reshape(base.numel())
  flat_val = value.reshape(-1)
  result = flat_base.scatter(0, idx_view, flat_val)
  base.assign(result.reshape(base.shape))
```

## Test Results

With current implementation (no view updates):
- 6/10 tests pass
- 4/10 tests fail because views don't update

The failing tests expect PyTorch's standard view semantics where all views of a tensor share storage and automatically reflect changes.

## Recommendation

Accept the limitation for now and document it. Proper view support requires architectural changes to the C++ wrapper layer to implement shared storage between views and their base tensors.
