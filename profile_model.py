#!/usr/bin/env python3
"""One-call debug & profile for tinygrad LLM inference.

Usage:
  .venv2/bin/python3 profile_model.py deepseek-v2-lite          # default 10 tokens
  .venv2/bin/python3 profile_model.py deepseek-v2-lite 20       # 20 tokens
  .venv2/bin/python3 profile_model.py glm-4.7:flash 10
  .venv2/bin/python3 profile_model.py youtu-llm:2b-Q4 20
  .venv2/bin/python3 profile_model.py deepseek-v2-lite 10 MOE_ADDS=0  # with extra env

Runs DEBUG=2 PROFILE=1 benchmark, then parses the output to produce:
  1. Performance summary (warmup vs steady tok/s, param bandwidth)
  2. Total kernel count and ICB batching breakdown
  3. Schedule cache hit/miss stats and per-token pattern
  4. Top kernels by call count and total warm-up time
  5. Kernel categories (q4k, reductions, elementwise, etc.)
  6. PROFILE pickle data: per-ICB batch timing (steady-state GPU time)

NOTE on timing:
  - DEBUG=2 kernel times are REAL for warm-up (pre-JIT, token 1).
    After JIT, kernels run inside ICBs and DEBUG=2 stops printing per-kernel.
  - PROFILE pickle gives per-ICB-batch GPU times (steady-state).
    Individual kernel times WITHIN an ICB are evenly divided (artifact).
  - The "warm-up" times are the best proxy for individual kernel efficiency.
"""
import subprocess, sys, re, os, pickle, collections

def strip_ansi(s):
  return re.sub(r'\x1b\[[0-9;]*m', '', s)

def parse_debug_log(lines):
  """Parse DEBUG=2 output lines into structured data."""
  kernels = []           # (seq, name, time_us, gbps_achieved, gbps_peak)
  schedule_batches = []  # (n_kernels, time_ms, cache_status, hash)
  icb_batches = []       # (n_kernels,)
  perf_lines = []        # (ms, toks, param_gbps)
  sched_time_total = 0.0
  sched_hits = 0
  sched_misses = 0

  for raw in lines:
    line = strip_ansi(raw).rstrip()

    # Kernel execution: *** METAL  N  kernel_name  ...  tm  Xus/Yms  (G GFLOPS  A|P  GB/s)
    m = re.match(r'\*\*\* METAL\s+(\d+)\s+(\S+)\s+.*tm\s+([\d.]+)(us|ms)/', line)
    if m:
      seq, name = int(m.group(1)), m.group(2)
      t = float(m.group(3))
      if m.group(4) == 'ms': t *= 1000
      bw = re.search(r'(\d+)\|(\d+)\s+GB/s', line)
      achieved = int(bw.group(1)) if bw else 0
      peak = int(bw.group(2)) if bw else 0
      kernels.append((seq, name, t, achieved, peak))
      continue

    # Schedule batch
    m = re.match(r'scheduled\s+(\d+) kernels in\s+([\d.]+) ms \|\s+(CACHE MISS|cache hit)\s+(\w+)', line)
    if m:
      n, t = int(m.group(1)), float(m.group(2))
      status, h = m.group(3), m.group(4)
      schedule_batches.append((n, t, status, h))
      sched_time_total += t
      if 'MISS' in status: sched_misses += 1
      else: sched_hits += 1
      continue

    # JIT GRAPHing batch
    m = re.match(r'JIT GRAPHing batch with (\d+) kernels', line)
    if m:
      icb_batches.append(int(m.group(1)))
      continue

    # Performance line: X ms, Y tok/s, Z GB/s, param W GB/s
    m = re.match(r'\s*([\d.]+) ms,\s+([\d.]+) tok/s.*param\s+([\d.]+) GB/s', line)
    if m:
      perf_lines.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
      continue

  return {
    'kernels': kernels, 'schedule_batches': schedule_batches,
    'icb_batches': icb_batches, 'perf_lines': perf_lines,
    'sched_time_total': sched_time_total,
    'sched_hits': sched_hits, 'sched_misses': sched_misses,
  }

def get_profile_path():
  import tempfile, getpass
  return os.path.join(tempfile.gettempdir(), f"profile.pkl.{getpass.getuser()}")

def get_rewrites_path():
  import tempfile, getpass
  return os.path.join(tempfile.gettempdir(), f"rewrites.pkl.{getpass.getuser()}")

def load_profile():
  """Try to load the PROFILE pickle for per-ICB-batch timing."""
  profile_path = get_profile_path()
  if not os.path.exists(profile_path):
    return None
  try:
    sys.path.insert(0, os.getcwd())
    from test.null.test_viz import load_profile as _load
    with open(profile_path, 'rb') as f:
      return _load(pickle.load(f))
  except Exception as e:
    return f"Error: {e}"

def categorize_kernel(name):
  if name.startswith('q4k_linear') or name.startswith('q4k_moe'): return 'q4k (MSL)'
  if name.startswith('q6k_'): return 'q6k (MSL)'
  if name.startswith('copy') or name.startswith('<'): return 'copies/batched'
  if name.startswith('E_'): return 'elementwise'
  if not name.startswith('r_'): return 'other'
  # Reductions: classify by size. Parse dims from name like r_88_32_3_8_4_2_32_8_4_2_32
  parts = name.lstrip('r_').split('_')
  # Filter out non-numeric suffixes (like 'n1', 'start', etc)
  dims = []
  for p in parts:
    m = re.match(r'^(\d+)', p)
    if m: dims.append(int(m.group(1)))
  total_work = 1
  for d in dims: total_work *= d
  if total_work > 10000: return 'large reduce (matmul/MoE)'
  return 'small reduce (norm/softmax/topk)'

def print_report(data, profile, model_name, kernel_sources=None):
  W = 80
  print("=" * W)
  print(f"  PROFILE: {model_name}")
  if kernel_sources: print(f"  (with kernel source, {len(kernel_sources)} kernels captured)")
  print("=" * W)

  # === Performance ===
  perf = data['perf_lines']
  print(f"\n--- Performance ({len(perf)} tokens) ---")
  if perf:
    print(f"  {'Token':<8s} {'ms':>8s} {'tok/s':>8s} {'param GB/s':>11s}")
    for i, (ms, toks, pgbps) in enumerate(perf):
      label = f"  {i+1:<8d}" if i < 3 else f"  {i+1:<8d}"
      print(f"{label} {ms:>8.1f} {toks:>8.2f} {pgbps:>11.1f}")
  if len(perf) >= 5:
    steady = perf[4:]  # skip first 4 (warmup + JIT compile)
    avg_ms = sum(p[0] for p in steady) / len(steady)
    avg_toks = sum(p[1] for p in steady) / len(steady)
    avg_pgbps = sum(p[2] for p in steady) / len(steady)
    print(f"  {'STEADY':>8s} {avg_ms:>8.1f} {avg_toks:>8.2f} {avg_pgbps:>11.1f}")

  # === Kernel Count & ICBs ===
  icbs = data['icb_batches']
  total_k = sum(icbs) if icbs else 0
  print(f"\n--- Kernels: {total_k} per token, {len(icbs)} ICBs ---")
  if icbs:
    print(f"  ICBs: {' + '.join(str(n) for n in icbs)} = {total_k}")
    if len(perf) >= 5:
      steady = perf[4:]
      avg_ms = sum(p[0] for p in steady) / len(steady)
      us_per_kernel = avg_ms * 1000 / total_k if total_k else 0
      print(f"  Avg overhead per kernel: {us_per_kernel:.1f} us  (steady {avg_ms:.1f} ms / {total_k} kernels)")

  # === Scheduling ===
  sb = data['schedule_batches']
  print(f"\n--- Scheduling: {data['sched_misses']} misses, {data['sched_hits']} hits, {data['sched_time_total']:.0f} ms total ---")
  if sb:
    # Group by hash, count hits vs misses per hash
    hash_info = {}  # hash -> {n_kernels, misses, hits, first_miss_ms, hit_ms_list}
    for n, t, status, h in sb:
      info = hash_info.setdefault(h, {'n': n, 'misses': 0, 'hits': 0, 'miss_ms': 0, 'hit_ms': []})
      if 'MISS' in status:
        info['misses'] += 1
        info['miss_ms'] = max(info['miss_ms'], t)
      else:
        info['hits'] += 1
        info['hit_ms'].append(t)
    # Show patterns that repeat (steady-state schedule structure)
    repeated = [(h, info) for h, info in hash_info.items() if info['hits'] + info['misses'] > 1]
    repeated.sort(key=lambda x: x[1]['hits'] + x[1]['misses'], reverse=True)
    if repeated:
      print(f"  Per-token schedule pattern ({len(repeated)} repeated):")
      total_per_tok = sum(info['n'] for _, info in repeated)
      for h, info in repeated[:12]:
        total_calls = info['hits'] + info['misses']
        avg_hit = sum(info['hit_ms']) / len(info['hit_ms']) if info['hit_ms'] else 0
        print(f"    {info['n']:>3d} kernels × {total_calls:>3d} calls  (miss: {info['miss_ms']:.1f}ms, hit: {avg_hit:.2f}ms)  [{h[:8]}]")
      print(f"  Total kernels in repeated patterns: {total_per_tok}")

  # === Warm-up Kernel Analysis (pre-JIT, real per-kernel times) ===
  kerns = data['kernels']
  # Filter out copies, batched, and warmup-only large elementwise
  compute_kerns = [(s, n, t, a, p) for s, n, t, a, p in kerns
                   if not n.startswith('copy') and not n.startswith('<')]
  if compute_kerns:
    name_stats = collections.Counter()
    name_times = collections.defaultdict(list)
    for seq, name, t, achieved, peak in compute_kerns:
      name_stats[name] += 1
      name_times[name].append((t, achieved, peak))

    print(f"\n--- Warm-up Kernel Analysis ({len(compute_kerns)} executions, pre-JIT real times) ---")

    # Top by total time
    print(f"\n  Top by Total Time:")
    print(f"  {'Name':<45s} {'Calls':>5s} {'Tot ms':>8s} {'Avg us':>8s} {'AvgBW':>6s}")
    print(f"  {'-'*45} {'-'*5} {'-'*8} {'-'*8} {'-'*6}")
    by_total = sorted(name_times.items(), key=lambda x: sum(t for t, _, _ in x[1]), reverse=True)
    for i, (name, times) in enumerate(by_total[:20]):
      count = len(times)
      tot_ms = sum(t for t, _, _ in times) / 1000
      avg_us = sum(t for t, _, _ in times) / count
      avg_bw = sum(a for _, a, _ in times) / count
      print(f"  {name:<45s} {count:>5d} {tot_ms:>8.2f} {avg_us:>8.1f} {avg_bw:>5.0f}GB")
      # Show source for top 3 slowest kernels if available
      if kernel_sources and i < 3 and name in kernel_sources:
        print(f"\n    Source for {name}:")
        src_lines = kernel_sources[name].split('\n')
        # Show first 30 lines of source
        for line in src_lines[:30]:
          print(f"    {line}")
        if len(src_lines) > 30:
          print(f"    ... ({len(src_lines)-30} more lines)")
        print()

    # Top by call count
    print(f"\n  Top by Call Count:")
    print(f"  {'Name':<45s} {'Calls':>5s} {'Avg us':>8s} {'Category':<25s}")
    print(f"  {'-'*45} {'-'*5} {'-'*8} {'-'*25}")
    for name, count in name_stats.most_common(20):
      times = name_times[name]
      avg_us = sum(t for t, _, _ in times) / count
      cat = categorize_kernel(name)
      print(f"  {name:<45s} {count:>5d} {avg_us:>8.1f} {cat}")

    # Categories
    print(f"\n  Kernel Categories:")
    cats = collections.defaultdict(lambda: [0, 0.0])
    for _, name, t, achieved, peak in compute_kerns:
      cat = categorize_kernel(name)
      cats[cat][0] += 1
      cats[cat][1] += t
    total_us = sum(v[1] for v in cats.values())
    print(f"  {'Category':<30s} {'Count':>6s} {'Tot ms':>8s} {'Pct':>6s} {'Per-tok':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")
    for cat, (count, tot_us) in sorted(cats.items(), key=lambda x: x[1][1], reverse=True):
      pct = tot_us / total_us * 100 if total_us > 0 else 0
      # Estimate per-token count from ICB total
      per_tok = round(count * total_k / len(compute_kerns)) if compute_kerns else 0
      print(f"  {cat:<30s} {count:>6d} {tot_us/1000:>8.1f} {pct:>5.1f}% {per_tok:>8d}")

  # === PROFILE: Steady-state ICB batch timing ===
  if isinstance(profile, dict) and 'layout' in profile:
    # Only show "METAL Graph" (ICB batches) and "METAL" (individual kernel aggregate)
    for dev_name in ['METAL Graph', 'METAL']:
      dev_data = profile['layout'].get(dev_name)
      if not dev_data or 'events' not in dev_data: continue
      events = [e for e in dev_data['events'] if 'dur' in e]
      if not events: continue

      agg = {}
      total_dur = 0
      for e in events:
        dur_ms = e['dur'] * 1e-6
        name_clean = strip_ansi(e['name'])
        a = agg.setdefault(name_clean, [0.0, 0])
        a[0] += dur_ms
        a[1] += 1
        total_dur += dur_ms

      if dev_name == 'METAL Graph':
        print(f"\n--- Steady-state GPU time: {total_dur:.1f} ms across {len(events)} ICB submissions ---")
        print(f"  (= {len(events)//len(perf) if perf else '?'} ICBs/tok × {len(perf)} tokens)")
        rows = sorted(agg.items(), key=lambda x: x[1][0], reverse=True)
        for name, (tot, cnt) in rows:
          avg = tot / cnt
          pct = tot / total_dur * 100
          print(f"  {name:<25s} {tot:>8.1f} ms  ({cnt:>3d} calls, {avg:>6.1f} ms avg, {pct:>5.1f}%)")
      elif dev_name == 'METAL':
        # NOTE: these times are artifacts (evenly divided within ICBs)
        # But the RELATIVE proportions show kernel count distribution
        print(f"\n--- Kernel call distribution (from PROFILE, {len(events)} events) ---")
        print(f"  NOTE: times within ICBs are evenly divided (artifact). Use call count, not time.")
        rows = sorted(agg.items(), key=lambda x: x[1][1], reverse=True)
        print(f"  {'Name':<45s} {'Calls':>6s} {'Category':<25s}")
        print(f"  {'-'*45} {'-'*6} {'-'*25}")
        for name, (tot, cnt) in rows[:25]:
          cat = categorize_kernel(name)
          print(f"  {name:<45s} {cnt:>6d} {cat}")

  elif isinstance(profile, str):
    print(f"\n  Profile: {profile}")

  print("\n" + "=" * W)

def extract_kernel_sources(output:str) -> dict[str, str]:
  """Extract Metal kernel source code from DEBUG=5 output.

  Parses output like:
    kernel void r_88_32_3_8(device float* data0, ...) {
      ...
    }

  Returns dict mapping kernel name -> full source code.
  """
  sources = {}
  lines = output.split('\n')
  i = 0
  while i < len(lines):
    # Look for kernel declarations: "kernel void <name>(" or "kernel void* <name>("
    m = re.match(r'kernel (?:void\*?|half\*?) (\S+)\(', lines[i])
    if m:
      name = m.group(1)
      # Collect lines until we hit the closing } (matching braces)
      src_lines = [lines[i]]
      i += 1
      brace_count = lines[i-1].count('{') - lines[i-1].count('}')
      while i < len(lines) and brace_count > 0:
        line = lines[i]
        src_lines.append(line)
        brace_count += line.count('{') - line.count('}')
        i += 1
      sources[name] = '\n'.join(src_lines)
    else:
      i += 1
  return sources

def main():
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <model> [n_tokens] [extra_env...] [--with-source]")
    print(f"  e.g.: {sys.argv[0]} deepseek-v2-lite 10")
    print(f"  e.g.: {sys.argv[0]} glm-4.7:flash 10 MOE_ADDS=2")
    print(f"  e.g.: {sys.argv[0]} deepseek-v2-lite 5 --with-source  # Includes kernel source (slower)")
    sys.exit(1)

  with_source = '--with-source' in sys.argv
  args = [a for a in sys.argv[1:] if a != '--with-source']

  model = args[0]
  n_tokens = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10
  extra_env = [a for a in args[1:] if '=' in a]

  env = os.environ.copy()
  env['DEBUG'] = '5' if with_source else '2'
  env['PROFILE'] = '1'
  for e in extra_env:
    k, v = e.split('=', 1)
    env[k] = v

  cmd = ['.venv2/bin/python3', 'tinygrad/apps/llm.py', '--model', model, '--benchmark', str(n_tokens)]
  env_str = ' '.join(f'{k}={v}' for k, v in sorted(env.items()) if k in ('DEBUG', 'PROFILE', *[e.split('=')[0] for e in extra_env]))
  print(f"$ {env_str} {' '.join(cmd)}")
  if with_source:
    print("  (Note: DEBUG=5 is slower due to Metal source printing)")
  print()

  try:
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    all_output = result.stdout + '\n' + result.stderr
  except subprocess.TimeoutExpired:
    print("ERROR: timed out after 180s")
    sys.exit(1)

  data = parse_debug_log(all_output.split('\n'))
  prof = load_profile()
  kernel_sources = extract_kernel_sources(all_output) if with_source else None
  print_report(data, prof, model, kernel_sources)

  log_path = os.path.abspath(f"./profile_{model.replace(':', '_').replace('.', '_')}.log")
  with open(log_path, 'w') as f:
    f.write(all_output)

  # Print file paths for deeper investigation
  profile_pkl = get_profile_path()
  rewrites_pkl = get_rewrites_path()
  print(f"\n--- Files for deeper investigation ---")
  print(f"  Raw DEBUG=2 log:    {log_path}")
  print(f"  Profile pickle:     {profile_pkl}  {'(exists)' if os.path.exists(profile_pkl) else '(not found)'}")
  print(f"  Rewrites pickle:    {rewrites_pkl}  {'(exists)' if os.path.exists(rewrites_pkl) else '(not found — run with VIZ=-1 to capture)'}")
  print()
  print(f"  Dig deeper:")
  print(f"    # View per-kernel profile (aggregated)")
  print(f"    PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --profile --device METAL")
  print(f"    # Inspect a specific kernel's occurrences")
  print(f"    PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --profile --device METAL --kernel '<kernel_name>'")
  print(f"    # View rewrites (requires VIZ=-1 capture)")
  print(f"    PYTHONPATH=. .venv2/bin/python3 extra/viz/cli.py --kernel '<kernel_name>'")
  print(f"    # Find kernel Metal source in debug log")
  print(f"    grep -n 'kernel void <kernel_name>' {log_path}")
  print(f"    # Re-run with DEBUG=5 for full Metal source of all kernels")
  print(f"    DEBUG=5 PROFILE=1 .venv2/bin/python3 tinygrad/apps/llm.py --model {model} --benchmark 3 > ./debug5_{model.replace(':', '_').replace('.', '_')}.log 2>&1")

if __name__ == '__main__':
  main()
