# Kernel Fusion Optimization in Torch Backend

## Problem: Inplace Operations Breaking Fusion

The torch backend was preventing kernel fusion by forcing premature execution of operations through `.realize()` and `.assign()` calls. This caused operations that should fuse together (like batchnorm + ReLU, or residual add + ReLU) to execute as separate kernels.

### Root Cause

PyTorch uses inplace operations extensively (e.g., `+=`, `*=`, `.zero_()`, `.fill_()`). The naive implementation of these operations used `.assign()`, which creates `UOp.ASSIGN` (now `Ops.AFTER`) nodes. These ASSIGN operations become kernel boundaries in tinygrad's scheduler, preventing fusion with subsequent operations.

**Key insight**: ASSIGN operations force synchronization points because they write to a buffer that subsequent operations read from. The scheduler must execute the ASSIGN kernel before any kernel that reads its output.

### Example: ResNet Residual Block

```python
# Before optimization - 3 kernels:
out = conv_bn(x)      # Kernel 1: conv+bn fused
out += identity       # Kernel 2: add (uses .assign())
out = relu(out)       # Kernel 3: relu (separate!)

# After optimization - 2 kernels:
out = conv_bn(x)      # Kernel 1: conv+bn fused
out += identity       # Uses .replace() - stays lazy!
out = relu(out)       # Kernel 2: add+relu fused!
```

## Solution: Use `.replace()` for Non-View Tensors

The fix is to use `.replace()` instead of `.assign()` for **non-view tensors** in inplace operations:

```python
# Before (creates fusion barrier):
def add_tensor_(self, other, alpha=1.0):
  self_t = unwrap(self)
  _apply_inplace(self_t, self_t + other * alpha)  # Uses .assign()
  return wrap(self_t)

# After (allows fusion):
def add_tensor_(self, other, alpha=1.0):
  self_t = unwrap(self)
  if not is_view(self_t):
    self_t.replace(self_t + other * alpha)  # Replaces UOp directly
  else:
    _apply_inplace(self_t, self_t + other * alpha)  # Still use assign for views
  return wrap(self_t)
```

### Why This Works

- **`.replace(new_uop)`**: Updates the tensor's UOp to point to the new computation graph. The computation stays **lazy** and can fuse with subsequent operations.
- **`.assign(value)`**: Creates an explicit write operation that must execute before anything reads from it. This is necessary for views but creates barriers for regular tensors.

## Implementation

### Helper Function

```python
def _inplace_op(t, new_value):
  # t is already unwrapped (tinygrad Tensor) when called from wrap_fxn lambdas
  if not is_view(t): t.replace(new_value)  # Fast path: allows fusion
  else: _apply_inplace(t, new_value)        # Slow path: for views
  return t
```

### Operations Fixed

Applied to these inplace operations in `tiny_backend` dict:
- `aten.zero_`: `lambda x: _inplace_op(x, x.zeros_like())`
- `aten.fill_.Scalar`: `lambda x, y: _inplace_op(x, x.full_like(y))`
- `aten.add_.Tensor`: `lambda self, other, alpha=1.0: _inplace_op(self, self + other * alpha)`
- `aten.add_.Scalar`: `lambda self, other, alpha=1.0: _inplace_op(self, self + other * alpha)`
- `aten.mul_.Tensor`: `lambda self, other: _inplace_op(self, self * other)`
- `aten.mul_.Scalar`: `lambda self, other: _inplace_op(self, self * other)`

## Results

### Before Optimization
```
*** METAL 231 [...] ['add', 'rsqrt', 'conv2d', 'batchnorm', 'relu']
*** METAL 232 [...] ['add', 'rsqrt', 'conv2d', 'batchnorm', '__add__']
*** METAL 233 [...] ['relu']  # ❌ ReLU executing separately
```

### After Optimization
```
*** METAL 224 [...] ['add', 'rsqrt', 'conv2d', 'batchnorm', 'relu']
*** METAL 225 [...] ['add', 'rsqrt', 'conv2d', 'batchnorm', '__add__', 'relu']  # ✅ ReLU fused!
```

ReLU now fuses with the residual addition, reducing kernel count and improving performance.

## When NOT to Apply This Optimization

### 1. Views Require `.assign()`

Views share underlying storage with base tensors. Using `.replace()` on a view would break the connection:

```python
x = torch.randn(10)
y = x[5:]  # y is a view of x
y += 1     # Must update the shared storage via .assign()
```

### 2. Running Statistics in BatchNorm

```python
# These MUST use .assign() - they're intentional side effects
running_mean_t.assign((1 - momentum) * running_mean_t + momentum * batch_mean.detach())
running_var_t.assign((1 - momentum) * running_var_t + momentum * batch_var.detach())
```

These are **intended** synchronization points. The running statistics need to be materialized and stored.

### 3. Output Form Operations

Operations like `cat.out`, `topk.values`, `sort.values` that write to pre-allocated output tensors should continue using `_apply_inplace` which internally uses `.assign()`. These are explicit user requests for in-place updates.

## Understanding the Trade-offs

### The Fundamental Tension

- **PyTorch semantics**: Inplace operations must modify existing tensors
- **Kernel fusion**: Requires keeping operations lazy in the computation graph

### How `.replace()` Resolves This

`.replace()` maintains PyTorch semantics (the tensor object still refers to the result) while keeping the computation lazy (the UOp is updated but not executed).

### Scheduler Behavior

In tinygrad's scheduler:
1. `Ops.AFTER` (ASSIGN) creates edges in the kernel dependency graph
2. Each ASSIGN becomes a potential kernel boundary
3. Operations can only fuse if they're in the same kernel
4. By avoiding unnecessary ASSIGN ops, we allow the scheduler to fuse more operations

## Performance Impact

Kernel fusion provides:
- **Reduced kernel launches**: Lower overhead from GPU kernel invocations
- **Better memory locality**: Intermediate results stay in registers/cache
- **Increased arithmetic intensity**: More compute per memory access

For ResNet-style architectures, the critical fusions are:
1. Conv → BatchNorm → ReLU
2. Residual Add → ReLU

Both of these now work correctly with the optimization.

## Future Improvements

To get even more fusion would require:
1. **Lazy ASSIGN scheduling**: Delay materializing ASSIGN operations until necessary
2. **Cross-ASSIGN fusion**: Allow scheduler to fuse operations across ASSIGN boundaries when safe
3. **Smart barrier insertion**: Only create barriers when there are actual data dependencies

These would require changes to tinygrad's core scheduler, not just the torch backend.

---

# Deeper Architectural Changes for Better Kernel Fusion in Torch Backend

## The Core Problem

When a tinygrad tensor is wrapped for torch and the same weight is used multiple times:
```python
conv1 = conv2d(x, w, b)  # w is used here
relu1 = conv1.relu()
conv2 = conv2d(relu1, w, b)  # w is reused here
```

Without `.realize()`, torch's autograd builds this graph:
```
conv1_output (unrealized UOP) → relu → conv2_output (unrealized UOP)
                ↓                              ↓
             weight_w                       weight_w
```

The problem: When torch computes gradients, it doesn't understand that the **same tinygrad tensor** `w` appears in **two different UOP graphs** because they haven't been realized yet.

## Why Current `.realize()` Fixes It

With `.realize()`:
```
conv1_output (BUFFER) → relu → conv2_output (BUFFER)
        ↓                              ↓
    weight_w                       weight_w
```

Now each conv output is a realized buffer, and torch can properly track that `w` contributes to both paths.

## Proposed Architectural Solutions

### Option 1: Custom Autograd Function with Explicit Dependency Tracking

**What**: Implement a custom `torch.autograd.Function` that explicitly tracks all tensor dependencies.

**How**:
```python
class TinygradConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
    input_t, weight_t, bias_t = unwrap(input), unwrap(weight), unwrap(bias)
    
    # Perform lazy conv2d
    result = input_t.conv2d(weight_t, bias_t, ...)
    
    # Track ALL tensors in the UOP graph for this result
    ctx.save_for_backward(input, weight, bias)
    ctx.uop_inputs = _extract_all_tensors_from_uop(result.uop)  # NEW
    ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
    
    return wrap(result)  # NO realize needed
  
  @staticmethod
  def backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors
    
    # Build fresh forward graph for gradient computation
    input_t, weight_t, bias_t = unwrap(input), unwrap(weight), unwrap(bias)
    out = input_t.conv2d(weight_t, bias_t, ...)
    
    # Compute gradients - this correctly handles reused weights
    grads = out.gradient(input_t, weight_t, bias_t, gradient=unwrap(grad_output))
    
    return wrap(grads[0]), wrap(grads[1]), wrap(grads[2]), None, None, None, None

def _extract_all_tensors_from_uop(uop):
  """Extract all BUFFER UOps that represent actual tensors"""
  buffers = set()
  for u in uop.toposort():
    if u.op == Ops.BUFFER:
      buffers.add(u)
  return buffers
```

**Benefits**:
- No `.realize()` needed in forward pass
- Proper gradient tracking for reused tensors
- Laziness preserved

**Challenges**:
- Complex to implement correctly
- Need to handle all edge cases (views, in-place ops, etc.)
- Performance overhead of tracking all dependencies

### Option 2: Lazy Realization Scheduler

**What**: Implement a smart scheduler that realizes tensors only when necessary for gradient correctness.

**How**:
```python
class LazyRealizationManager:
  def __init__(self):
    self.tensor_usage_count = {}  # Maps tensor id -> usage count
    self.pending_realizes = []
  
  def mark_usage(self, tensor):
    tid = id(unwrap(tensor))
    self.tensor_usage_count[tid] = self.tensor_usage_count.get(tid, 0) + 1
    
    # If tensor is used more than once in forward pass, schedule realize
    if self.tensor_usage_count[tid] > 1:
      self.pending_realizes.append(tensor)
  
  def realize_if_needed(self):
    # Realize all multi-use tensors before backward
    for t in self.pending_realizes:
      unwrap(t).realize()
    self.pending_realizes.clear()

# Usage:
lazy_manager = LazyRealizationManager()

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, ...):
  lazy_manager.mark_usage(weight)  # Track weight usage
  lazy_manager.mark_usage(input)
  
  result = input.conv2d(weight, bias, ...)
  
  # Don't realize immediately
  return wrap(result)

# Hook into backward pass start
def pre_backward_hook():
  lazy_manager.realize_if_needed()
```

**Benefits**:
- Automatic - no manual `.realize()` calls
- Optimizes for common case (single use)
- Minimal overhead

**Challenges**:
- Need to hook into torch's backward pass start
- Tracking tensor reuse across complex graphs
- Race conditions in multi-threaded scenarios

### Option 3: Realize-on-Wrap with Smart Caching

**What**: Realize tensors when wrapping, but cache and reuse realized buffers.

**How**:
```python
class RealizedTensorCache:
  def __init__(self):
    self.cache = {}  # Maps UOP hash -> realized Tensor
  
  def get_or_realize(self, tensor):
    # Create deterministic hash of UOP graph
    uop_hash = self._hash_uop_graph(tensor.uop)
    
    if uop_hash in self.cache:
      # Reuse previously realized tensor
      return self.cache[uop_hash]
    
    # Realize and cache
    realized = tensor.realize()
    self.cache[uop_hash] = realized
    return realized
  
  def _hash_uop_graph(self, uop):
    # Hash the entire UOP computation graph
    return hash(str(uop.toposort()))  # Simplified

cache = RealizedTensorCache()

def wrap(x: Tensor) -> torch.Tensor:
  # Smart realize: only if UOP graph hasn't been computed before
  x_realized = cache.get_or_realize(x)
  x_realized._strides = strides_for_shape(x_realized.shape)
  if not hasattr(x_realized, '_storage_offset'):
    x_realized._storage_offset = calculate_storage_offset(x_realized)
  return mod.wrap(x_realized, _to_torch_dtype(x_realized.dtype), 
                  _to_torch_device(x_realized.device).index)
```

**Benefits**:
- Deduplicates identical computations
- Better kernel fusion for repeated patterns
- Simple to implement

**Challenges**:
- UOP hashing is expensive and non-trivial
- Cache invalidation is complex
- Memory overhead of cache

### Option 4: Hybrid Lazy-Eager Execution

**What**: Use lazy evaluation for operations that compose well, eager for problematic ops.

**How**:
```python
ALWAYS_REALIZE_OPS = {
  'conv2d',  # Convolutions always realize
  'matmul',  # Matrix multiplications always realize
}

LAZY_OPS = {
  'relu', 'add', 'mul',  # Element-wise ops stay lazy
  'reshape', 'permute',  # View ops stay lazy
}

def smart_wrap(tensor, op_name):
  if op_name in ALWAYS_REALIZE_OPS:
    return wrap(tensor.realize())
  else:
    return wrap(tensor)  # Keep lazy

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, ...):
  result = input.conv2d(weight, bias, ...)
  return smart_wrap(result, 'conv2d')  # Always realize

@torch.library.impl("aten::relu", "privateuseone")
def relu(input):
  result = unwrap(input).relu()
  return smart_wrap(result, 'relu')  # Stay lazy
```

**Benefits**:
- Simple and predictable
- Surgical approach - only realize where needed
- Easy to tune performance

**Challenges**:
- Need to identify all problematic ops
- May over-realize in some cases
- Configuration management

### Option 5: Fix at Tinygrad Core Level

**What**: Modify tinygrad's Tensor wrapper to be torch-autograd-aware.

**How**:
```python
# In tinygrad/tensor.py

class Tensor:
  def __init__(self, ...):
    self._torch_wrapper = None  # NEW: backreference to torch tensor
    self._gradient_consumers = []  # NEW: track who needs gradients from this
  
  def register_torch_wrapper(self, torch_tensor):
    """Called by torch backend when wrapping"""
    self._torch_wrapper = torch_tensor
  
  def realize(self):
    # When realizing, notify torch wrapper
    if self._torch_wrapper is not None:
      self._torch_wrapper._tinygrad_realized = True
    return super().realize()

# In torch backend:
def wrap(x: Tensor):
  torch_t = mod.wrap(x, ...)
  x.register_torch_wrapper(torch_t)  # Bidirectional link
  return torch_t
```

**Benefits**:
- Clean separation of concerns
- Works for all torch backend ops
- Minimal overhead

**Challenges**:
- Requires modifying tinygrad core
- Circular references need careful handling
- May complicate tinygrad's design

## Recommended Approach

**Hybrid Strategy: Options 1 + 4**

1. **Short term**: Implement Option 4 (Hybrid Lazy-Eager)
   - Quick win with minimal changes
   - Realize only known-problematic ops like conv2d
   - Keep element-wise ops lazy for fusion

2. **Medium term**: Implement Option 1 (Custom Autograd Functions)
   - Properly handle all edge cases
   - Full control over gradient computation
   - Better debugging capabilities

3. **Long term**: Work with tinygrad core (Option 5)
   - Make tinygrad natively torch-aware
   - Enable deep integration
   - Benefit all torch backend users

## Implementation Priority

1. **Measure first**: Profile where kernels are NOT fusing
2. **Target specific ops**: Start with element-wise ops (relu, add, mul)
3. **Incremental rollout**: One op category at a time
4. **Extensive testing**: Each change needs comprehensive gradient tests

## Example: Removing Realize from Element-Wise Ops

```python
# Current (everything realizes):
conv1 = conv2d(x, w, b).realize()  # Kernel 1
relu1 = relu(conv1).realize()       # Kernel 2
conv2 = conv2d(relu1, w, b).realize()  # Kernel 3
# Total: 3 kernels

# After fix (fusion possible):
conv1 = conv2d(x, w, b).realize()  # Kernel 1 (still need this)
relu1 = relu(conv1)                 # Lazy - fuses with next op
conv2 = conv2d(relu1, w, b).realize()  # Kernel 2 (conv + relu fused)
# Total: 2 kernels (33% reduction!)
```

## Conclusion

The deepest fix requires making torch's autograd system understand tinygrad's lazy evaluation, OR making tinygrad's lazy evaluation torch-aware. Both require significant architectural work, but the payoff is substantial kernel fusion improvements.

The `.realize()` calls are currently necessary but can be selectively removed with careful engineering.

## Empirical Analysis: Current State (235 Kernels)

### Achievement
Reduced from **417 kernels → 235 kernels** (43.6% reduction) by implementing native batch norm. This represents the ceiling with current architecture.

### The 235-Kernel Ceiling

The current implementation hits a hard limit because:

1. **Convolution requires `.realize()`** - Without it, gradients fail on tensor reuse
2. **Batch norm stats can't fuse** - Mean/invstd calculations become separate tiny kernels
3. **Pattern: 3 kernels per conv+bn block**:
   ```
   Kernel 1: E_16_4    - batch_mean calculation (tiny)
   Kernel 2: E_16_4n1  - invstd calculation (tiny)
   Kernel 3: r_*       - convolution (realized, blocks fusion)
   ```

### Test Case: `test_biased_conv2d`

The minimal test case revealing the fundamental issue:

```python
# Same weight w and bias b used twice
out = conv2d(conv2d(x, w, b).relu(), w, b)
```

**Without `.realize()` in conv forward:**
```
RuntimeError: tensor <Tensor <LB x on DEVICE:meta> ...> not found in
<tinygrad.lazy.LazyBuffer object at ...>
```

**Why it fails:**
1. First conv creates lazy UOP graph: `conv_graph_1` references `w`
2. Second conv creates lazy UOP graph: `conv_graph_2` references same `w`
3. When computing `∂loss/∂w`, tinygrad's `gradient()` tries to find `w` in the lazy graph
4. But `w` appears in TWO separate unrealized graphs - ambiguous which path to backprop through
5. The UOP graph doesn't have bidirectional links to track "this weight is used in multiple computations"

**With `.realize()` in conv forward:**
1. First conv creates BUFFER (realized): `buffer_1`
2. Second conv references `buffer_1` as input (clean dependency)
3. Each weight usage has a clear forward/backward path
4. Gradient computation succeeds: `∂buffer_1/∂w` and `∂out/∂buffer_1` chain properly

This is the fundamental tradeoff: **lazy evaluation enables fusion, but gradient tracking with tensor reuse requires realization**.

### Why Removing `.detach()` From Batch Norm Didn't Help

Attempted optimization:
```python
# Before (with .detach()):
batch_mean = input.mean(axis=(0,2,3)).detach()
batch_var = ((input - batch_mean.reshape(1,-1,1,1))**2).mean(axis=(0,2,3)).detach()

# After (without .detach()):
batch_mean = input.mean(axis=(0,2,3))
batch_var = ((input - batch_mean.reshape(1,-1,1,1))**2).mean(axis=(0,2,3))
```

**Result:** Still 235 kernels. No change.

**Reason:** The real fusion blocker is the `.realize()` in convolution, not the `.detach()` in batch norm. The 2 tiny kernels for mean/invstd are separate because the convolution output is already realized - there's nothing to fuse with.

### Removed Unnecessary Code

Simplified from 4 convolution implementations to 2:

**Removed:**
- `aten::convolution_backward` - PyTorch autograd never calls this
- `aten::convolution_overrideable` - Superseded by direct registration

**Kept:**
- `aten::convolution` - Forward pass (with `.realize()`)
- `aten::convolution_backward_overrideable` - Backward pass (PyTorch's preferred interface)

PyTorch's autograd is hardcoded to call `convolution_backward_overrideable`, not `convolution_backward`. The duplicate implementations were dead code.

### Path to <100 Kernels

Current: **235 kernels** (3 per conv+bn block in typical network)
Target: **<100 kernels** (requires 57% reduction from current state)

**Cannot be achieved without architectural changes.** The options:

1. **Quick win (Option 4):** Remove `.realize()` from element-wise ops (relu, add, mul)
   - These don't cause gradient tracking issues
   - Estimated: 30-50 kernel reduction
   - Still far from <100 kernel goal

2. **Medium-term (Option 1):** Implement custom autograd with dependency tracking
   - Enables full conv+bn fusion
   - Each fused block becomes 1 kernel instead of 3
   - Could reach <100 kernels

3. **Long-term (Option 5):** Fix tinygrad core with bidirectional tensor links
   - Cleanest solution, benefits all users
   - Requires core tinygrad changes

**Recommendation:** Start with Option 1 (custom autograd) for conv2d+batchnorm. This is the highest-leverage change for reaching <100 kernels.

## Deep Dive Analysis (Nov 2025)

The information above is partially outdated. While the explicit `.realize()` call in `convolution_overrideable` was removed, the fundamental fusion break between convolution and subsequent operations (like batch normalization) persists. A deep-dive investigation was performed to find the new root cause.

### Investigation with a Debug Script

To isolate the problem, a targeted debug script was created at `extra/torch_backend/debug_fusion.py`. This script builds a minimal model containing a single `Conv2d -> BatchNorm2d -> ReLU` block, allowing for precise analysis of the fusion between these layers.

By running this script with high verbosity (`DEBUG=5`) and adding custom logging to check the state of the tensors, we can observe the lifecycle of the computation graph.

### The Real Root Cause: Implicit Realization

The analysis of the debug logs revealed the following sequence:

1.  The `torch.nn.Conv2d` layer is executed.
2.  Immediately after the convolution operation returns, the underlying `tinygrad.Tensor`'s `uop` (micro-operation graph) has its `is_realized` property set to `True`.
3.  This proves that the tensor is no longer a symbolic "lazy" graph but has been eagerly computed into a concrete buffer in memory.
4.  When the subsequent `BatchNorm2d` layer is executed, it receives this realized tensor. Since there is no symbolic graph to append to, `tinygrad` is forced to create a new, separate computation graph for the batch norm statistics (mean and variance).
5.  These new graphs are then compiled into their own separate kernels.

**Conclusion:** The fusion break is caused by an **implicit realization** of the convolution's output. The effect is the same as the old explicit `.realize()` call: it creates a hard barrier in the computation graph that prevents fusion.

### Why is it Realizing Implicitly?

The exact trigger for this implicit realization is still under investigation, but it is a side effect of the interaction between PyTorch's `privateuseone` backend and `tinygrad`'s scheduling and autograd engine. The most likely reason is that the autograd system requires a concrete tensor to correctly track gradients, especially for tensors that are reused across different parts of the graph (like convolution weights).

### Next Steps for Debugging

To find the precise code path that triggers this implicit realization, the next step is to instrument the core realization logic in `tinygrad`.

1.  **Add a Stack Trace**: Modify the `Tensor.realize()` method in `tinygrad/tensor.py`.
2.  **Add a Conditional Breakpoint**: Inside the method, add a condition to check if the tensor being realized is the output of the convolution (e.g., by checking its shape: `(1, 16, 224, 224)` in the debug script).
3.  **Print Stack**: If the condition is met, print a full stack trace (`traceback.print_stack()`).

This will capture the exact moment of realization and provide the full call chain, revealing which function in the `tinygrad` or `torch_backend` codebase is forcing the eager computation.
