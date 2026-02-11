#!/usr/bin/env python3
"""Round 2: drill into GROUP size and combinations."""
import subprocess, sys, os, shutil

BENCH_CMD = [sys.executable, "bench_block.py", "12"]
HEURISTIC = "tinygrad/codegen/opt/heuristic.py"

def backup(f): shutil.copy(f, f + ".bak")
def restore(f): shutil.copy(f + ".bak", f)
def read(f):
  with open(f) as fh: return fh.read()
def write(f, s):
  with open(f, "w") as fh: fh.write(s)

def run_bench(label):
  print(f"\n{'='*60}")
  print(f"  {label}")
  print(f"{'='*60}")
  r = subprocess.run(BENCH_CMD, capture_output=True, text=True, timeout=180)
  out = r.stdout + r.stderr
  # Extract JIT median
  lines = out.split('\n')
  jit_ms = None
  for line in lines:
    if 'median:' in line and 'blocks/s' in line:
      try: jit_ms = float(line.split('median:')[1].split('ms')[0].strip())
      except: pass
  nojit_ms = None
  for line in lines:
    if 'median:' in line and 'blocks/s' not in line and 'feed_forward' not in line:
      try: nojit_ms = float(line.split('median:')[1].split('ms')[0].strip())
      except: pass
  print(f"  JIT: {jit_ms}ms  noJIT_ff: {nojit_ms}ms")
  return jit_ms

backup(HEURISTIC)
results = []
ORIG_LINE = "for gsz in [16, 8, 4]:"
ORIG_LSZU = "for lsz, usz in [(4,4), (4,2), (2,4), (2,2)]:"

try:
  # Baseline
  restore(HEURISTIC)
  t = run_bench("BASELINE")
  results.append(("BASELINE", t))

  # GROUP sizes
  for gsz_list in ["[64, 32, 16, 8, 4]", "[48, 32, 16, 8, 4]", "[32, 16, 8, 4]", "[32, 24, 16, 8, 4]"]:
    restore(HEURISTIC)
    h = read(HEURISTIC)
    h = h.replace(ORIG_LINE, f"for gsz in {gsz_list}:")
    write(HEURISTIC, h)
    t = run_bench(f"GROUP {gsz_list}")
    results.append((f"G{gsz_list[:10]}", t))

  # GROUP 32 + different LOCAL/UPCAST
  for lu_list in [
    "[(2,4), (2,2), (4,4), (4,2)]",  # prefer smaller LOCAL
    "[(2,8), (2,4), (4,4), (4,2), (2,2)]",  # LOCAL 2 + big UPCAST
    "[(4,4), (4,2), (2,4), (2,2)]",  # original LOCAL (just GROUP change)
  ]:
    restore(HEURISTIC)
    h = read(HEURISTIC)
    h = h.replace(ORIG_LINE, "for gsz in [32, 16, 8, 4]:")
    h = h.replace(ORIG_LSZU, f"for lsz, usz in {lu_list}:")
    write(HEURISTIC, h)
    t = run_bench(f"G32 + LU={lu_list[:30]}")
    results.append((f"G32+{lu_list[:20]}", t))

  # GROUP 32 with relaxed reduce_rngs condition (>= 1 instead of >= 2)
  restore(HEURISTIC)
  h = read(HEURISTIC)
  h = h.replace(ORIG_LINE, "for gsz in [32, 16, 8, 4]:")
  h = h.replace("len(reduce_rngs) >= 2", "len(reduce_rngs) >= 1")
  write(HEURISTIC, h)
  t = run_bench("G32 + reduce_rngs>=1")
  results.append(("G32+rngs>=1", t))

  # GROUP 32 + lower reduce_product threshold
  restore(HEURISTIC)
  h = read(HEURISTIC)
  h = h.replace(ORIG_LINE, "for gsz in [32, 16, 8, 4]:")
  h = h.replace("reduce_product >= 256", "reduce_product >= 64")
  write(HEURISTIC, h)
  t = run_bench("G32 + reduce>=64")
  results.append(("G32+rp>=64", t))

finally:
  restore(HEURISTIC)
  try: os.remove(HEURISTIC + ".bak")
  except: pass

print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY (Round 2)")
print(f"{'='*60}")
print(f"{'Experiment':<30} {'JIT ms':>8} {'vs base':>10}")
print("-" * 50)
base = results[0][1] if results else None
for name, t in sorted(results, key=lambda x: x[1] or 999):
  if t and base:
    delta = (t - base) / base * 100
    marker = " <<<" if delta < -5 else ""
    print(f"{name:<30} {t:7.2f}ms {delta:+8.1f}%{marker}")
  else:
    print(f"{name:<30} {'FAIL':>8}")
