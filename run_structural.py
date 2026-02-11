#!/usr/bin/env python3
"""Structural experiments: change how expert data flows to enable fusion."""
import subprocess, sys, os, shutil

BENCH_CMD = [sys.executable, "bench_block.py", "25"]
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
  medians = []
  kernels = []
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

backup(QUANTIZED)
backup(MLA)
results = []

try:
  # ── 0. BASELINE ──
  restore(QUANTIZED); restore(MLA)
  jit, ff, k = run_bench("BASELINE")
  results.append(("BASELINE", jit, ff, k))

  # ── 1. Split gather: separate scale + packed gathers (enable ewise→reduce fusion) ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  old_expert = """    self._ensure_expert_blocks(x.device)
    sel_blocks = self._expert_blocks[sel.reshape(-1)]  # (n_sel, bpe, bpb)

    # Q4_0 packed-dot: subtract 8 inline (matches QuantizedLinear path, avoids separate x_block_sum kernel)
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      O, IN = self.out_features, self.in_features
      bpr = IN // 32
      blocks = sel_blocks.reshape(n_sel, O, bpr, 18)
      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = blocks[:, :, :, 2:]  # (n_sel, O, bpr, 16)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).reshape(B, T, K, O)"""

  new_expert_split = """    self._ensure_expert_blocks(x.device)
    sel_flat = sel.reshape(-1)

    # Q4_0 packed-dot with SPLIT gathers (separate scale/packed to enable ewise->reduce fusion)
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      O, IN = self.out_features, self.in_features
      bpr = IN // 32
      # Pre-reshape to 4D so gathers are targeted
      eb_4d = self._expert_blocks.reshape(self.num_experts, O, bpr, 18)
      # Two separate gathers — each has single consumer, enabling fusion with reduce
      scale = eb_4d[sel_flat, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = eb_4d[sel_flat, :, :, 2:]  # (n_sel, O, bpr, 16)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).reshape(B, T, K, O)"""

  q = q.replace(old_expert, new_expert_split)
  write(QUANTIZED, q)
  jit, ff, k = run_bench("1. Split gather (separate scale/packed)")
  results.append(("SPLIT_GATHER", jit, ff, k))

  # ── 2. Pre-separate at init: store scale and packed as separate tensors ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  # Add _q4_0_scale and _q4_0_packed to slots
  q = q.replace(
    "'expert_first_in_memory', '_blocks_per_expert', '_expert_blocks')",
    "'expert_first_in_memory', '_blocks_per_expert', '_expert_blocks', '_q4_0_scale', '_q4_0_packed')")
  # Init them to None
  q = q.replace(
    "    self._expert_blocks = None\n    assert",
    "    self._expert_blocks = None\n    self._q4_0_scale = None\n    self._q4_0_packed = None\n    assert")
  # Add ensure method
  ensure_sep = """
  def _ensure_q4_0_separated(self, device):
    if self._q4_0_scale is not None and self._q4_0_scale.device == device: return
    self._ensure_expert_blocks(device)
    O, bpr = self.out_features, self.in_features // 32
    eb_4d = self._expert_blocks.reshape(self.num_experts, O, bpr, 18)
    self._q4_0_scale = eb_4d[:, :, :, :2].bitcast(dtypes.float16).contiguous().realize()  # (E, O, bpr, 1)
    self._q4_0_packed = eb_4d[:, :, :, 2:].contiguous().realize()  # (E, O, bpr, 16)
"""
  q = q.replace("  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:", ensure_sep + "  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:")
  # Replace the Q4_0 path
  new_expert_presep = """    self._ensure_q4_0_separated(x.device)
    sel_flat = sel.reshape(-1)

    # Q4_0 with pre-separated scale/packed (no slicing in hot path)
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      O, IN = self.out_features, self.in_features
      bpr = IN // 32
      scale = self._q4_0_scale[sel_flat]  # (n_sel, O, bpr, 1) — single gather
      packed = self._q4_0_packed[sel_flat]  # (n_sel, O, bpr, 16) — single gather
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]
      lo = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0
      hi = packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_lo + hi * x_hi)).reshape(n_sel, O, bpr * 16).sum(axis=-1).reshape(B, T, K, O)"""
  q = q.replace(old_expert, new_expert_presep)
  write(QUANTIZED, q)
  jit, ff, k = run_bench("2. Pre-separated scale/packed tensors")
  results.append(("PRE_SEP", jit, ff, k))

  # ── 3. Remove .contiguous() in MoE path (let scheduler fuse more) ──
  restore(QUANTIZED); restore(MLA)
  m = read(MLA)
  m = m.replace("      gated = (gate.silu() * up).contiguous()", "      gated = gate.silu() * up")
  m = m.replace("      expert_out = self.ffn_down_exps(sel, gated).contiguous()", "      expert_out = self.ffn_down_exps(sel, gated)")
  write(MLA, m)
  jit, ff, k = run_bench("3. Remove .contiguous() in MoE")
  results.append(("NO_CONTIG", jit, ff, k))

  # ── 4. Remove only FIRST .contiguous() (keep down_exps contiguous) ──
  restore(QUANTIZED); restore(MLA)
  m = read(MLA)
  m = m.replace("      gated = (gate.silu() * up).contiguous()", "      gated = gate.silu() * up")
  write(MLA, m)
  jit, ff, k = run_bench("4. Remove gated .contiguous() only")
  results.append(("NO_CONTIG1", jit, ff, k))

  # ── 5. Split gather + remove gated contiguous ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED); q = q.replace(old_expert, new_expert_split); write(QUANTIZED, q)
  m = read(MLA); m = m.replace("      gated = (gate.silu() * up).contiguous()", "      gated = gate.silu() * up"); write(MLA, m)
  jit, ff, k = run_bench("5. Split gather + no gated contig")
  results.append(("SPLIT+NOCON", jit, ff, k))

  # ── 6. Add .contiguous() AFTER expert gather (force materialization boundary) ──
  restore(QUANTIZED); restore(MLA)
  q = read(QUANTIZED)
  q = q.replace(
    "    sel_blocks = self._expert_blocks[sel.reshape(-1)]  # (n_sel, bpe, bpb)",
    "    sel_blocks = self._expert_blocks[sel.reshape(-1)].contiguous()  # (n_sel, bpe, bpb)")
  write(QUANTIZED, q)
  jit, ff, k = run_bench("6. Explicit .contiguous() after gather")
  results.append(("CONTIG_GATH", jit, ff, k))

finally:
  restore(QUANTIZED); restore(MLA)
  for f in [QUANTIZED, MLA]:
    try: os.remove(f + ".bak")
    except: pass

print(f"\n{'='*60}")
print(f"  STRUCTURAL EXPERIMENTS (sorted by FF time)")
print(f"{'='*60}")
base_ff = results[0][2]
print(f"{'Experiment':<20} {'JIT ms':>8} {'FF ms':>8} {'FF%':>8} {'#k':>4}")
print("-" * 52)
for name, jit, ff, k in sorted(results, key=lambda x: x[2] or 999):
  fd = (ff - base_ff) / base_ff * 100 if ff and base_ff else 0
  ks = str(k) if k else "?"
  js = f"{jit:.2f}" if jit else "FAIL"
  fs = f"{ff:.2f}" if ff else "FAIL"
  marker = " <<<" if fd < -5 else ""
  print(f"{name:<20} {js:>7}ms {fs:>7}ms {fd:+7.1f}% {ks:>4}{marker}")
