#!/usr/bin/env python3
"""Run multiple optimization experiments sequentially using bench_block."""
import subprocess, sys, os, time, shutil

BENCH_CMD = [sys.executable, "bench_block.py", "10"]
HEURISTIC = "tinygrad/codegen/opt/heuristic.py"
QUANTIZED = "tinygrad/apps/quantized.py"

def backup(f): shutil.copy(f, f + ".bak")
def restore(f): shutil.copy(f + ".bak", f)

def read(f):
  with open(f) as fh: return fh.read()
def write(f, s):
  with open(f, "w") as fh: fh.write(s)

def run_bench(label):
  print(f"\n{'='*60}")
  print(f"  EXPERIMENT: {label}")
  print(f"{'='*60}")
  r = subprocess.run(BENCH_CMD, capture_output=True, text=True, timeout=180)
  out = r.stdout + r.stderr
  print(out)
  # Extract median from "median: X.XX ms"
  for line in out.split('\n'):
    if 'median:' in line and 'full block' not in line and 'feed_forward' not in line:
      parts = line.split('median:')[1].split('ms')[0].strip()
      try: return float(parts)
      except: pass
  return None

# Backup originals
backup(HEURISTIC)
backup(QUANTIZED)

results = []

try:
  # ── Experiment 0: Baseline ──
  restore(HEURISTIC)
  restore(QUANTIZED)
  t = run_bench("BASELINE (original)")
  results.append(("BASELINE", t))

  # ── Experiment 1: Bigger GROUP in Q4_0 heuristic (32 instead of max 16) ──
  restore(HEURISTIC)
  h = read(HEURISTIC)
  h = h.replace("for gsz in [16, 8, 4]:", "for gsz in [32, 16, 8, 4]:")
  write(HEURISTIC, h)
  t = run_bench("Q4_0 heuristic: GROUP 32")
  results.append(("GROUP 32", t))

  # ── Experiment 2: Bigger LOCAL in Q4_0 heuristic ──
  restore(HEURISTIC)
  h = read(HEURISTIC)
  h = h.replace("for lsz, usz in [(4,4), (4,2), (2,4), (2,2)]:",
                "for lsz, usz in [(8,4), (8,2), (4,8), (4,4), (4,2), (2,4), (2,2)]:")
  write(HEURISTIC, h)
  t = run_bench("Q4_0 heuristic: LOCAL 8, UPCAST 8")
  results.append(("LOCAL 8", t))

  # ── Experiment 3: Both bigger GROUP and LOCAL ──
  restore(HEURISTIC)
  h = read(HEURISTIC)
  h = h.replace("for gsz in [16, 8, 4]:", "for gsz in [32, 16, 8, 4]:")
  h = h.replace("for lsz, usz in [(4,4), (4,2), (2,4), (2,2)]:",
                "for lsz, usz in [(8,4), (8,2), (4,8), (4,4), (4,2), (2,4), (2,2)]:")
  write(HEURISTIC, h)
  t = run_bench("Q4_0 heuristic: GROUP 32 + LOCAL 8")
  results.append(("GROUP32+LOCAL8", t))

  # ── Experiment 4: Raise GROUPTOP threshold (2048 -> 32768) ──
  restore(HEURISTIC)
  h = read(HEURISTIC)
  h = h.replace("prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048)",
                "prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 32768)")
  write(HEURISTIC, h)
  t = run_bench("GROUPTOP threshold 32768")
  results.append(("GROUPTOP 32K", t))

  # ── Experiment 5: Disable Q4_0 heuristic entirely (let GROUPTOP handle it) ──
  restore(HEURISTIC)
  h = read(HEURISTIC)
  # Comment out the Q4_0 block by replacing the if condition
  h = h.replace(
    "  # Q4_0 packed-dot GROUP: bitwise ops in reduce chain indicate dequant",
    "  if False: # DISABLED Q4_0 packed-dot GROUP")
  # Also raise GROUPTOP
  h = h.replace("prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 2048)",
                "prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if NOLOCALS else 32768)")
  write(HEURISTIC, h)
  t = run_bench("No Q4_0 heuristic + GROUPTOP 32K")
  results.append(("NO_Q4+GT32K", t))

  # ── Experiment 6: MV heuristic relaxed to match dequant chains ──
  restore(HEURISTIC)
  h = read(HEURISTIC)
  # Relax MV: don't require both sides to be INDEX - allow dequant chain on weight side
  old_mv = "      if mulop.src[act_i].op is not Ops.INDEX: continue"
  new_mv = "      if mulop.src[act_i].op is not Ops.INDEX and not (mulop.src[act_i].op is Ops.CAST and mulop.src[act_i].src[0].op is Ops.MUL): continue"
  h = h.replace(old_mv, new_mv)
  write(HEURISTIC, h)
  t = run_bench("MV relaxed for dequant")
  results.append(("MV_RELAXED", t))

finally:
  # Always restore originals
  restore(HEURISTIC)
  restore(QUANTIZED)
  # Clean up backups
  for f in [HEURISTIC, QUANTIZED]:
    try: os.remove(f + ".bak")
    except: pass

print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Experiment':<25} {'JIT ms':>8} {'vs base':>10}")
print("-" * 45)
base = results[0][1] if results else None
for name, t in results:
  if t and base:
    delta = (t - base) / base * 100
    print(f"{name:<25} {t:7.2f}ms {delta:+8.1f}%")
  else:
    print(f"{name:<25} {'FAIL':>8}")
