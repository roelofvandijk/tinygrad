#!/usr/bin/env python3
"""Minimal reproduction of slow backward conv2d kernel"""
import time
from tinygrad import Tensor, Device, GlobalCounters

# Reproduce the slow backward conv2d from resnet layer1 conv1
# Input: (64, 64, 56, 56) - batch, channels, height, width
# Weight: (64, 64, 3, 3) - out_channels, in_channels, kh, kw
# Output gradient: (64, 64, 56, 56)

def benchmark_conv_backward(bs=64, cin=64, cout=64, H=56, W=56, k=3, iters=3):
  print(f"Benchmarking conv2d backward: bs={bs}, cin={cin}, cout={cout}, H={H}, W={W}, k={k}")
  
  # Create inputs
  x = Tensor.randn(bs, cin, H, W, requires_grad=True)
  w = Tensor.randn(cout, cin, k, k, requires_grad=True)
  
  # Forward pass
  y = x.conv2d(w, padding=1)
  
  # Create gradient for backward
  grad_output = Tensor.randn(*y.shape)
  
  # Realize everything to setup
  x.realize()
  w.realize()
  grad_output.realize()
  
  best_time = float('inf')
  for i in range(iters):
    GlobalCounters.reset()

    # Backward pass - this generates the slow kernel
    y = x.conv2d(w, padding=1)

    st = time.perf_counter()
    y.backward(grad_output)
    # Force realization of gradients
    x.grad.realize()
    w.grad.realize()
    Device[Device.DEFAULT].synchronize()
    et = time.perf_counter()

    tm = et - st
    if tm < best_time:
      best_time = tm

    print(f"  Iter {i}: {tm*1000:.2f}ms, {GlobalCounters.kernel_count} kernels, "
          f"{GlobalCounters.global_ops/1e9/tm:.2f} GFLOPS, {GlobalCounters.global_mem/1e9/tm:.2f} GB/s")
  
  print(f"\nBest time: {best_time*1000:.2f}ms")
  return best_time

if __name__ == "__main__":
  # Run the benchmark
  benchmark_conv_backward()
