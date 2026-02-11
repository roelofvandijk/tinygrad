#!/usr/bin/env python3
"""
Generic split/fuse search harness for tinygrad benchmark modules.

Module contract (minimal):
  - expose `bench_mask(mask: int, iters: int = ..., beam: int = ...) -> float` returning time in ms (lower is better)
  - optionally expose `SPLIT_POINTS: list[str]` for readable bit names

Example:
  python search2.py --module my_kernel_bench.py --mode exhaustive --beam 4 --iters 8 --repeat 3
"""
from __future__ import annotations
import argparse, importlib, importlib.util, inspect, json, pathlib, statistics, sys, time
from dataclasses import dataclass, asdict

def _load_module(mod: str):
  p = pathlib.Path(mod)
  if p.suffix == ".py" and p.exists():
    spec = importlib.util.spec_from_file_location(f"search2_mod_{p.stem}", p)
    if spec is None or spec.loader is None: raise RuntimeError(f"failed to load module from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
  return importlib.import_module(mod)

def _call_bench(fn, mask: int, iters: int, beam: int) -> float:
  sig = inspect.signature(fn)
  kwargs = {}
  if "mask" in sig.parameters: kwargs["mask"] = mask
  if "iters" in sig.parameters: kwargs["iters"] = iters
  if "beam" in sig.parameters: kwargs["beam"] = beam
  if len(kwargs) == 0: return float(fn(mask))
  return float(fn(**kwargs))

@dataclass(frozen=True)
class Result:
  mask: int
  median_ms: float
  best_ms: float
  p90_ms: float
  samples_ms: tuple[float, ...]

def _eval_mask(fn, mask: int, iters: int, beam: int, repeat: int) -> Result:
  vals: list[float] = []
  for _ in range(repeat): vals.append(_call_bench(fn, mask=mask, iters=iters, beam=beam))
  sv = sorted(vals)
  p90 = sv[min(len(sv)-1, int(0.9 * (len(sv)-1)))]
  return Result(mask=mask, median_ms=float(statistics.median(vals)), best_ms=min(vals), p90_ms=p90, samples_ms=tuple(vals))

def _iter_exhaustive(n_bits: int, max_masks: int | None):
  total = 1 << n_bits
  if max_masks is None or max_masks >= total:
    for m in range(total): yield m
    return
  step = max(1, total // max_masks)
  seen = set()
  for i in range(max_masks):
    m = min(total - 1, i * step)
    if m in seen: continue
    seen.add(m)
    yield m

def _search_greedy(fn, n_bits: int, iters: int, beam: int, repeat: int, eps_ms: float, mode: str) -> list[Result]:
  # mode: greedy-fuse (all split -> clear bits), greedy-split (all fused -> set bits)
  cur_mask = (1 << n_bits) - 1 if mode == "greedy-fuse" else 0
  cur = _eval_mask(fn, cur_mask, iters, beam, repeat)
  out = [cur]
  while True:
    best = cur
    for bit in range(n_bits):
      cand_mask = (cur.mask & ~(1 << bit)) if mode == "greedy-fuse" else (cur.mask | (1 << bit))
      if cand_mask == cur.mask: continue
      cand = _eval_mask(fn, cand_mask, iters, beam, repeat)
      out.append(cand)
      if cand.median_ms + eps_ms < best.median_ms: best = cand
    if best.mask == cur.mask: break
    cur = best
  return out

def _search_hillclimb(fn, n_bits: int, iters: int, beam: int, repeat: int, eps_ms: float, start_mask: int) -> list[Result]:
  cur = _eval_mask(fn, start_mask, iters, beam, repeat)
  out = [cur]
  seen = {cur.mask}
  while True:
    best = cur
    for bit in range(n_bits):
      cand_mask = cur.mask ^ (1 << bit)
      if cand_mask in seen: continue
      cand = _eval_mask(fn, cand_mask, iters, beam, repeat)
      seen.add(cand_mask)
      out.append(cand)
      if cand.median_ms + eps_ms < best.median_ms: best = cand
    if best.mask == cur.mask: break
    cur = best
  return out

def _fmt_mask(mask: int, n_bits: int) -> str:
  width = max(1, (n_bits + 3) // 4)
  return f"0x{mask:0{width}x}"

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--module", required=True, help="Module name or .py path exposing bench_mask")
  ap.add_argument("--fn", default="bench_mask", help="Benchmark function name")
  ap.add_argument("--mode", choices=["exhaustive", "greedy-fuse", "greedy-split", "hillclimb"], default="exhaustive")
  ap.add_argument("--iters", type=int, default=8, help="iters passed to bench function")
  ap.add_argument("--repeat", type=int, default=3, help="repeats per mask (median used)")
  ap.add_argument("--beam", type=int, default=0, help="beam passed to bench function")
  ap.add_argument("--eps-ms", type=float, default=0.05, help="minimum improvement to accept in greedy/hillclimb")
  ap.add_argument("--n-splits", type=int, default=-1, help="override split bit count; default from module SPLIT_POINTS")
  ap.add_argument("--start-mask", type=int, default=0, help="hillclimb start mask")
  ap.add_argument("--max-masks", type=int, default=0, help="cap masks for exhaustive (0 = all)")
  ap.add_argument("--topk", type=int, default=12, help="number of best masks to print")
  ap.add_argument("--json-out", default="", help="optional output path for all results json")
  args = ap.parse_args()

  mod = _load_module(args.module)
  if not hasattr(mod, args.fn): raise RuntimeError(f"{args.module} missing function {args.fn}")
  fn = getattr(mod, args.fn)
  if not callable(fn): raise RuntimeError(f"{args.fn} is not callable")

  split_points = list(getattr(mod, "SPLIT_POINTS", []))
  n_bits = args.n_splits if args.n_splits >= 0 else len(split_points)
  if n_bits <= 0: raise RuntimeError("split bit count is zero; provide module SPLIT_POINTS or --n-splits")
  if len(split_points) == 0: split_points = [f"bit_{i}" for i in range(n_bits)]
  if len(split_points) != n_bits: split_points = [split_points[i] if i < len(split_points) else f"bit_{i}" for i in range(n_bits)]

  print(f"search2: module={args.module} fn={args.fn} mode={args.mode} n_bits={n_bits} beam={args.beam} iters={args.iters} repeat={args.repeat}")
  for i, name in enumerate(split_points): print(f"  bit {i:2d}: {name}")
  st = time.perf_counter()

  results: list[Result] = []
  if args.mode == "exhaustive":
    for idx, mask in enumerate(_iter_exhaustive(n_bits, args.max_masks if args.max_masks > 0 else None)):
      r = _eval_mask(fn, mask, args.iters, args.beam, args.repeat)
      results.append(r)
      print(f"[{idx:5d}] mask={_fmt_mask(mask, n_bits)} median={r.median_ms:.3f} ms best={r.best_ms:.3f} p90={r.p90_ms:.3f}")
  elif args.mode in {"greedy-fuse", "greedy-split"}:
    results = _search_greedy(fn, n_bits, args.iters, args.beam, args.repeat, args.eps_ms, args.mode)
  else:
    results = _search_hillclimb(fn, n_bits, args.iters, args.beam, args.repeat, args.eps_ms, args.start_mask)

  best = sorted(results, key=lambda r: r.median_ms)[:max(1, args.topk)]
  print("\nTop results:")
  for i, r in enumerate(best, 1):
    print(f"{i:2d}. mask={_fmt_mask(r.mask, n_bits)} median={r.median_ms:.3f} ms best={r.best_ms:.3f} p90={r.p90_ms:.3f}")
    split = [split_points[b] for b in range(n_bits) if (r.mask >> b) & 1]
    fused = [split_points[b] for b in range(n_bits) if ((r.mask >> b) & 1) == 0]
    print(f"    split({len(split)}): {split}")
    print(f"    fused({len(fused)}): {fused}")

  print(f"\nDone in {time.perf_counter()-st:.2f}s; evaluated {len(results)} masks")
  if args.json_out:
    payload = {
      "module": args.module, "fn": args.fn, "mode": args.mode, "n_bits": n_bits,
      "beam": args.beam, "iters": args.iters, "repeat": args.repeat, "results": [asdict(r) for r in results],
      "split_points": split_points,
    }
    pathlib.Path(args.json_out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.json_out}")

if __name__ == "__main__":
  main()
