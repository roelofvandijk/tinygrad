#!/usr/bin/env python3
"""Wide experiments: DSL reformulations, layout changes, heuristic changes, and combinations."""
import subprocess, sys, os, shutil, re

BENCH_CMD = [sys.executable, "bench_block.py", "12"]
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
    # Print last 5 lines of error
    for line in out.strip().split('\n')[-5:]: print(f"  ERR: {line}")
    return None, None
  medians = []
  for line in out.split('\n'):
    if 'median:' in line:
      try: medians.append(float(line.split('median:')[1].split('ms')[0].strip()))
      except: pass
  jit_ms = medians[0] if len(medians) >= 1 else None
  nojit_ms = medians[1] if len(medians) >= 2 else None
  print(f"  JIT: {jit_ms:.2f}ms  FF: {nojit_ms:.2f}ms" if jit_ms and nojit_ms else f"  JIT: {jit_ms}  FF: {nojit_ms}", flush=True)
  return jit_ms, nojit_ms

# ── QUANTIZED.PY VARIANTS ──

# Variant A: dequant-then-matmul (standard x @ W.T so MV heuristic matches)
VARIANT_A_LINEAR = '''      self._ensure_q4_0_blocks(x.device)
      O, bpr = self.out_features, self.in_features // 32
      blocks = self._q4_0_blocks
      scale = blocks[:, :, :2].bitcast(dtypes.float16)  # (O, bpr, 1)
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      lo = (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0) * scale  # (O, bpr, 16)
      hi = (packed.rshift(4).cast(dtypes.float16) - 8.0) * scale  # (O, bpr, 16)
      w = lo.cat(hi, dim=-1).reshape(O, self.in_features)  # (O, bpr*32) = (O, I)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      return (x_fp16.reshape(-1, self.in_features) @ w.T).reshape(*x.shape[:-1], O)'''

VARIANT_A_EXPERT = '''      blocks = sel_blocks.reshape(n_sel, O, bpr, 18)
      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = blocks[:, :, :, 2:]  # (n_sel, O, bpr, 16)
      lo = (packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0) * scale
      hi = (packed.rshift(4).cast(dtypes.float16) - 8.0) * scale
      w = lo.cat(hi, dim=-1).reshape(n_sel, O, IN)  # (n_sel, O, I)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      return (x_fp16.reshape(n_sel, 1, IN) @ w.transpose(-1, -2)).reshape(B, T, K, O)'''

# Variant B: pre-separate scale and packed at load time (no slicing in hot path)
VARIANT_B_LINEAR = '''      self._ensure_q4_0_blocks(x.device)
      O, bpr = self.out_features, self.in_features // 32
      blocks = self._q4_0_blocks
      scale = blocks[:, :, :2].bitcast(dtypes.float16)  # (O, bpr, 1)
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).reshape(*x.shape[:-1], O)'''

# Variant C: fp32 accumulation (dequant in fp32)
VARIANT_C_LINEAR = '''      self._ensure_q4_0_blocks(x.device)
      O, bpr = self.out_features, self.in_features // 32
      blocks = self._q4_0_blocks
      scale = blocks[:, :, :2].bitcast(dtypes.float16).float()  # (O, bpr, 1)
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16).float()
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = (packed.bitwise_and(0xF)).float() - 8.0
      hi = (packed.rshift(4)).float() - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).cast(dtypes.float16).reshape(*x.shape[:-1], O)'''

VARIANT_C_EXPERT = '''      blocks = sel_blocks.reshape(n_sel, O, bpr, 18)
      scale = blocks[:, :, :, :2].bitcast(dtypes.float16).float()
      packed = blocks[:, :, :, 2:]
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16).float()
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = (packed.bitwise_and(0xF)).float() - 8.0
      hi = packed.rshift(4).float() - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).cast(dtypes.float16).reshape(B, T, K, O)'''

# Variant D: split into per-expert matmuls (more kernels but smaller, may parallelize better)
VARIANT_D_EXPERT = '''      blocks = sel_blocks.reshape(n_sel, O, bpr, 18)
      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)
      packed = blocks[:, :, :, 2:]
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      # Split into individual expert matmuls
      results = []
      for i in range(n_sel):
        s_i = scale[i]  # (O, bpr, 1)
        p_i = packed[i]  # (O, bpr, 16)
        x_i = x_fp16[i:i+1]  # (1, IN)
        x_pairs_i = x_i.reshape(1, bpr, 2, 16)
        lo_i = (p_i.bitwise_and(0xF).cast(dtypes.float16) - 8.0)
        hi_i = (p_i.rshift(4).cast(dtypes.float16) - 8.0)
        dot_i = (s_i * (lo_i * x_pairs_i[:, :, 0, :] + hi_i * x_pairs_i[:, :, 1, :])).reshape(1, O, bpr*16).sum(axis=-1)
        results.append(dot_i)
      return Tensor.cat(*results, dim=0).reshape(B, T, K, O)'''

# ── ORIGINAL CODE MARKERS ──
ORIG_LINEAR = '''      self._ensure_q4_0_blocks(x.device)
      O, bpr = self.out_features, self.in_features // 32
      blocks = self._q4_0_blocks
      scale = blocks[:, :, :2].bitcast(dtypes.float16)  # (O, bpr, 1)
      packed = blocks[:, :, 2:]  # (O, bpr, 16)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_pairs = x_fp16.reshape(-1, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = (packed.rshift(4)).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(-1, O, bpr * 16).sum(axis=-1).reshape(*x.shape[:-1], O)'''

ORIG_EXPERT = '''      blocks = sel_blocks.reshape(n_sel, O, bpr, 18)
      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = blocks[:, :, :, 2:]  # (n_sel, O, bpr, 16)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).reshape(B, T, K, O)'''

HEUR_ORIG_GSZ = "for gsz in [16, 8, 4]:"

def apply_variant(linear_code=None, expert_code=None, heur_gsz=None):
  restore(QUANTIZED)
  restore(HEURISTIC)
  if linear_code:
    q = read(QUANTIZED)
    q = q.replace(ORIG_LINEAR, linear_code)
    write(QUANTIZED, q)
  if expert_code:
    q = read(QUANTIZED)
    q = q.replace(ORIG_EXPERT, expert_code)
    write(QUANTIZED, q)
  if heur_gsz:
    h = read(HEURISTIC)
    h = h.replace(HEUR_ORIG_GSZ, heur_gsz)
    write(HEURISTIC, h)

backup(HEURISTIC)
backup(QUANTIZED)
results = []

try:
  # 0. Baseline
  apply_variant()
  jit, ff = run_bench("0. BASELINE")
  results.append(("BASELINE", jit))

  # 1. GROUP 32 heuristic only
  apply_variant(heur_gsz="for gsz in [32, 16, 8, 4]:")
  jit, ff = run_bench("1. GROUP 32 heuristic")
  results.append(("GROUP32", jit))

  # 2. Dequant-then-matmul (x @ W.T formulation for MV heuristic)
  apply_variant(linear_code=VARIANT_A_LINEAR, expert_code=VARIANT_A_EXPERT)
  jit, ff = run_bench("2. Dequant-then-matmul (x@W.T)")
  results.append(("DEQUANT_MATMUL", jit))

  # 3. Dequant-then-matmul + GROUP 32
  apply_variant(linear_code=VARIANT_A_LINEAR, expert_code=VARIANT_A_EXPERT, heur_gsz="for gsz in [32, 16, 8, 4]:")
  jit, ff = run_bench("3. Dequant-matmul + GROUP 32")
  results.append(("DQ_MAT+G32", jit))

  # 4. fp32 accumulation
  apply_variant(linear_code=VARIANT_C_LINEAR, expert_code=VARIANT_C_EXPERT)
  jit, ff = run_bench("4. fp32 accumulation")
  results.append(("FP32_ACC", jit))

  # 5. fp32 accumulation + GROUP 32
  apply_variant(linear_code=VARIANT_C_LINEAR, expert_code=VARIANT_C_EXPERT, heur_gsz="for gsz in [32, 16, 8, 4]:")
  jit, ff = run_bench("5. fp32 acc + GROUP 32")
  results.append(("FP32+G32", jit))

  # 6. Per-expert split matmuls (more kernels, smaller each)
  apply_variant(expert_code=VARIANT_D_EXPERT)
  jit, ff = run_bench("6. Per-expert split matmuls")
  results.append(("SPLIT_EXPERT", jit))

  # 7. Per-expert split + GROUP 32
  apply_variant(expert_code=VARIANT_D_EXPERT, heur_gsz="for gsz in [32, 16, 8, 4]:")
  jit, ff = run_bench("7. Split expert + GROUP 32")
  results.append(("SPLIT+G32", jit))

  # 8. Dequant-matmul for LINEAR only + original expert (hybrid)
  apply_variant(linear_code=VARIANT_A_LINEAR)
  jit, ff = run_bench("8. DQ-matmul LINEAR only")
  results.append(("DQ_LINEAR", jit))

  # 9. Dequant-matmul LINEAR + GROUP 32
  apply_variant(linear_code=VARIANT_A_LINEAR, heur_gsz="for gsz in [32, 16, 8, 4]:")
  jit, ff = run_bench("9. DQ LINEAR + GROUP 32")
  results.append(("DQ_LIN+G32", jit))

finally:
  restore(HEURISTIC)
  restore(QUANTIZED)
  for f in [HEURISTIC, QUANTIZED]:
    try: os.remove(f + ".bak")
    except: pass

print(f"\n{'='*60}")
print(f"  RESULTS (sorted by speed)")
print(f"{'='*60}")
print(f"{'Experiment':<25} {'JIT ms':>8} {'vs base':>10}")
print("-" * 45)
base = results[0][1] if results else None
for name, t in sorted(results, key=lambda x: x[1] or 999):
  if t and base:
    delta = (t - base) / base * 100
    marker = " <<<" if delta < -5 else ""
    print(f"{name:<25} {t:7.2f}ms {delta:+8.1f}%{marker}")
  else:
    print(f"{name:<25} {'FAIL':>8}")
