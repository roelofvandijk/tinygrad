#!/usr/bin/env python3
"""Combination experiments: test winning changes together."""
import subprocess, sys, os, shutil

BENCH_CMD = [sys.executable, "bench_block.py", "30"]
QUANTIZED = "tinygrad/apps/quantized.py"
MLA = "tinygrad/apps/mla.py"

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
    for line in out.strip().split('\n')[-8:]: print(f"  ERR: {line}")
    return None, None, None
  medians, kernels = [], []
  for line in out.split('\n'):
    if 'median:' in line:
      try: medians.append(float(line.split('median:')[1].split('ms')[0].strip()))
      except: pass
    if 'warmup 2' in line:
      try: kernels.append(int(line.split(':')[1].split('kernel')[0].strip()))
      except: pass
  jit_ms = medians[0] if len(medians) >= 1 else None
  ff_ms = medians[1] if len(medians) >= 2 else None
  k = kernels[-1] if kernels else None
  print(f"  JIT={jit_ms:.2f}ms  FF={ff_ms:.2f}ms  kernels={k}" if jit_ms and ff_ms else f"  {jit_ms} {ff_ms} k={k}", flush=True)
  return jit_ms, ff_ms, k

# Patch functions
def apply_contig_gather(q):
  """Add .contiguous() after expert gather."""
  return q.replace(
    "    sel_blocks = self._expert_blocks[sel.reshape(-1)]  # (n_sel, bpe, bpb)",
    "    sel_blocks = self._expert_blocks[sel.reshape(-1)].contiguous()  # (n_sel, bpe, bpb)")

def apply_no_gated_contig(m):
  """Remove .contiguous() on gated."""
  return m.replace("      gated = (gate.silu() * up).contiguous()", "      gated = gate.silu() * up")

def apply_fp32_expert(q):
  """Use fp32 accumulation in expert Q4_0 path."""
  old = '''      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = blocks[:, :, :, 2:]  # (n_sel, O, bpr, 16)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).reshape(B, T, K, O)'''
  new = '''      scale = blocks[:, :, :, :2].bitcast(dtypes.float16).float()
      packed = blocks[:, :, :, 2:]
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16).float()
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = (packed.bitwise_and(0xF)).float() - 8.0
      hi = packed.rshift(4).float() - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).half().reshape(B, T, K, O)'''
  return q.replace(old, new)

def apply_fp32_linear(q):
  """Use fp32 accumulation in linear Q4_0 path."""
  old = '''      scale = blocks[:, :, :2].bitcast(dtypes.float16)  # (O, bpr, 1)
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).reshape(*x.shape[:-1], O)'''
  new = '''      scale = blocks[:, :, :2].bitcast(dtypes.float16).float()  # (O, bpr, 1) -> fp32
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16).float()
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = (packed.bitwise_and(0xF)).float() - 8.0
      hi = (packed.rshift(4)).float() - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).half().reshape(*x.shape[:-1], O)'''
  return q.replace(old, new)

backup(QUANTIZED)
backup(MLA)
results = []

try:
  # ── 0. BASELINE (x2 for variance) ──
  restore(QUANTIZED); restore(MLA)
  jit, ff, k = run_bench("BASELINE run 1")
  results.append(("BASE_1", jit, ff, k))
  jit, ff, k = run_bench("BASELINE run 2")
  results.append(("BASE_2", jit, ff, k))

  # ── 1. CONTIG_GATHER only ──
  restore(QUANTIZED); restore(MLA)
  write(QUANTIZED, apply_contig_gather(read(QUANTIZED)))
  jit, ff, k = run_bench("1. .contiguous() after gather")
  results.append(("CONTIG_GATH", jit, ff, k))

  # ── 2. NO_CONTIG1 only ──
  restore(QUANTIZED); restore(MLA)
  write(MLA, apply_no_gated_contig(read(MLA)))
  jit, ff, k = run_bench("2. Remove gated .contiguous()")
  results.append(("NO_CONTIG1", jit, ff, k))

  # ── 3. COMBO: CONTIG_GATHER + NO_CONTIG1 ──
  restore(QUANTIZED); restore(MLA)
  write(QUANTIZED, apply_contig_gather(read(QUANTIZED)))
  write(MLA, apply_no_gated_contig(read(MLA)))
  jit, ff, k = run_bench("3. Gather contig + no gated contig")
  results.append(("COMBO_CG+NC", jit, ff, k))

  # ── 4. fp32 accumulation (expert + linear) ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  q = apply_fp32_expert(q)
  q = apply_fp32_linear(q)
  write(QUANTIZED, q)
  jit, ff, k = run_bench("4. fp32 accumulation")
  results.append(("FP32", jit, ff, k))

  # ── 5. TRIPLE: CONTIG_GATHER + NO_CONTIG1 + fp32 ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  q = apply_contig_gather(q)
  q = apply_fp32_expert(q)
  q = apply_fp32_linear(q)
  write(QUANTIZED, q)
  write(MLA, apply_no_gated_contig(read(MLA)))
  jit, ff, k = run_bench("5. All three combined")
  results.append(("TRIPLE", jit, ff, k))

  # ── 6. CONTIG_GATHER + fp32 (no MLA change) ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  q = apply_contig_gather(q)
  q = apply_fp32_expert(q)
  q = apply_fp32_linear(q)
  write(QUANTIZED, q)
  jit, ff, k = run_bench("6. Gather contig + fp32")
  results.append(("CG+FP32", jit, ff, k))

  # ── 7. NO_CONTIG1 + fp32 ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  q = apply_fp32_expert(q)
  q = apply_fp32_linear(q)
  write(QUANTIZED, q)
  write(MLA, apply_no_gated_contig(read(MLA)))
  jit, ff, k = run_bench("7. No gated contig + fp32")
  results.append(("NC+FP32", jit, ff, k))

finally:
  restore(QUANTIZED); restore(MLA)
  for f in [QUANTIZED, MLA]:
    try: os.remove(f + ".bak")
    except: pass

print(f"\n{'='*60}")
print(f"  COMBINATION EXPERIMENTS (sorted by FF time)")
print(f"{'='*60}")
base_ff = results[0][2]
base_jit = results[0][1]
print(f"{'Experiment':<20} {'JIT ms':>8} {'JIT%':>8} {'FF ms':>8} {'FF%':>8} {'#k':>4}")
print("-" * 60)
for name, jit, ff, k in sorted(results, key=lambda x: x[2] or 999):
  jd = (jit - base_jit) / base_jit * 100 if jit and base_jit else 0
  fd = (ff - base_ff) / base_ff * 100 if ff and base_ff else 0
  ks = str(k) if k else "?"
  js = f"{jit:.2f}" if jit else "FAIL"
  fs = f"{ff:.2f}" if ff else "FAIL"
  marker = " <<<" if fd < -5 else ""
  print(f"{name:<20} {js:>7}ms {jd:+7.1f}% {fs:>7}ms {fd:+7.1f}% {ks:>4}{marker}")
