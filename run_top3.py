#!/usr/bin/env python3
"""Re-run top 3 experiments with more iterations for less noise."""
import subprocess, sys, os, shutil

BENCH_CMD = [sys.executable, "bench_block.py", "30"]
HEURISTIC = "tinygrad/codegen/opt/heuristic.py"
QUANTIZED = "tinygrad/apps/quantized.py"

def backup(f): shutil.copy(f, f + ".bak")
def restore(f): shutil.copy(f + ".bak", f)
def read(f):
  with open(f) as fh: return fh.read()
def write(f, s):
  with open(f, "w") as fh: fh.write(s)

def run_bench(label):
  print(f"\n{'='*60}\n  {label}\n{'='*60}", flush=True)
  r = subprocess.run(BENCH_CMD, capture_output=True, text=True, timeout=300)
  out = r.stdout + r.stderr
  if r.returncode != 0:
    for line in out.strip().split('\n')[-5:]: print(f"  ERR: {line}")
    return None, None
  medians = []
  for line in out.split('\n'):
    if 'median:' in line:
      try: medians.append(float(line.split('median:')[1].split('ms')[0].strip()))
      except: pass
  jit_ms = medians[0] if len(medians) >= 1 else None
  nojit_ms = medians[1] if len(medians) >= 2 else None
  # also get best
  bests = []
  for line in out.split('\n'):
    if 'best:' in line:
      try: bests.append(float(line.split('best:')[1].split('ms')[0].strip()))
      except: pass
  jit_best = bests[0] if bests else None
  print(f"  JIT: median={jit_ms:.2f} best={jit_best:.2f}  FF: {nojit_ms:.2f}ms" if jit_ms and nojit_ms and jit_best else f"  {jit_ms} {nojit_ms}", flush=True)
  return jit_ms, nojit_ms

ORIG_GSZ = "for gsz in [16, 8, 4]:"
ORIG_LINEAR = '''      scale = blocks[:, :, :2].bitcast(dtypes.float16)  # (O, bpr, 1)
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).reshape(*x.shape[:-1], O)'''

FP32_LINEAR = '''      scale = blocks[:, :, :2].bitcast(dtypes.float16).float()  # (O, bpr, 1) -> fp32
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16).float()
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = (packed.bitwise_and(0xF)).float() - 8.0
      hi = (packed.rshift(4)).float() - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).half().reshape(*x.shape[:-1], O)'''

ORIG_EXPERT = '''      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = blocks[:, :, :, 2:]  # (n_sel, O, bpr, 16)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).reshape(B, T, K, O)'''

FP32_EXPERT = '''      scale = blocks[:, :, :, :2].bitcast(dtypes.float16).float()
      packed = blocks[:, :, :, 2:]
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16).float()
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = (packed.bitwise_and(0xF)).float() - 8.0
      hi = packed.rshift(4).float() - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).half().reshape(B, T, K, O)'''

backup(HEURISTIC)
backup(QUANTIZED)
results = []

try:
  # 0. BASELINE (run twice to check variance)
  restore(HEURISTIC); restore(QUANTIZED)
  jit, ff = run_bench("BASELINE run 1")
  results.append(("BASE_1", jit, ff))
  jit, ff = run_bench("BASELINE run 2")
  results.append(("BASE_2", jit, ff))

  # 1. GROUP 32 only
  restore(HEURISTIC); restore(QUANTIZED)
  h = read(HEURISTIC); h = h.replace(ORIG_GSZ, "for gsz in [32, 16, 8, 4]:"); write(HEURISTIC, h)
  jit, ff = run_bench("GROUP 32")
  results.append(("GROUP32", jit, ff))

  # 2. fp32 accumulation only
  restore(HEURISTIC); restore(QUANTIZED)
  q = read(QUANTIZED); q = q.replace(ORIG_LINEAR, FP32_LINEAR); q = q.replace(ORIG_EXPERT, FP32_EXPERT); write(QUANTIZED, q)
  jit, ff = run_bench("fp32 acc")
  results.append(("FP32", jit, ff))

  # 3. fp32 + GROUP 32
  restore(HEURISTIC); restore(QUANTIZED)
  h = read(HEURISTIC); h = h.replace(ORIG_GSZ, "for gsz in [32, 16, 8, 4]:"); write(HEURISTIC, h)
  q = read(QUANTIZED); q = q.replace(ORIG_LINEAR, FP32_LINEAR); q = q.replace(ORIG_EXPERT, FP32_EXPERT); write(QUANTIZED, q)
  jit, ff = run_bench("fp32 + GROUP 32")
  results.append(("FP32+G32", jit, ff))

finally:
  restore(HEURISTIC); restore(QUANTIZED)
  for f in [HEURISTIC, QUANTIZED]:
    try: os.remove(f + ".bak")
    except: pass

print(f"\n{'='*60}")
print(f"  RESULTS (30 iterations each)")
print(f"{'='*60}")
base_jit = results[0][1]
base_ff = results[0][2]
print(f"{'Experiment':<20} {'JIT ms':>8} {'JIT %':>8} {'FF ms':>8} {'FF %':>8}")
print("-" * 55)
for name, jit, ff in results:
  jd = (jit - base_jit) / base_jit * 100 if jit and base_jit else 0
  fd = (ff - base_ff) / base_ff * 100 if ff and base_ff else 0
  print(f"{name:<20} {jit:7.2f}ms {jd:+7.1f}% {ff:7.2f}ms {fd:+7.1f}%")
