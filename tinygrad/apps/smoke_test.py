#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, time
from datetime import datetime
from pathlib import Path
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, SimpleTokenizer, models

SMOKE_MODELS = {
  "youtu": ("youtu-llm:2b-Q4_0", [37317, 11, 290, 1483, 371]),  # 'Okay, the user is'
  "deepseek": ("deepseek-v2-lite", [790, 40619, 13, 976, 185]),  # '"Paris."\n'
  "glm": ("glm-4.7:flash", [16, 13, 220, 3070, 28203]),          # '1.  **Ident'
}
HISTORY_FILE = Path(__file__).parent / "smoke_test_history.md"

def parse_last_row(model: str) -> dict|None:
  """Parse last timing for a model from history file."""
  if not HISTORY_FILE.exists(): return None
  for line in reversed(HISTORY_FILE.read_text().splitlines()):
    if f"| {model} |" in line:
      parts = line.split("|")
      if len(parts) >= 5:
        try:
          return {"load": float(parts[3].strip()), "prefill": float(parts[4].strip()), "tps": float(parts[5].strip())}
        except ValueError: pass
  return None

def fmt_delta(cur: float, prev: float|None, higher_better: bool = False) -> str:
  if prev is None: return f"{cur:.2f}"
  delta = cur - prev
  sign = "+" if delta >= 0 else ""
  good = (delta > 0) if higher_better else (delta < 0)
  indicator = "↑" if good else "↓" if delta != 0 else ""
  return f"{cur:.2f} ({sign}{delta:.2f}{indicator})"

def test_model(model_key: str, expected_tokens: list[int], prev: dict|None) -> tuple[bool, dict]:
  print(f"Testing {model_key}...")
  try:
    t0 = time.perf_counter()
    model, kv = Transformer.from_gguf(Tensor.from_url(models[model_key]), max_context=512, quantized=True)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    load_time = time.perf_counter() - t0

    bos_id = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
    eos_id = kv['tokenizer.ggml.eos_token_id']
    ids = tok.build_chat_ids([{"role": "user", "content": "What is the capital of France?"}], bos_id, eos_id)

    # Use generate() like benchmark does
    gen = model.generate(ids, 0)
    tokens = []

    # First token (prefill + JIT compile)
    t0 = time.perf_counter()
    tokens.append(next(gen))
    prefill_time = time.perf_counter() - t0

    # Warmup: 7 more tokens
    for _ in range(7):
      tokens.append(next(gen))

    # Timed generation: final token
    t0 = time.perf_counter()
    tokens.append(next(gen))
    tps = 1.0 / (time.perf_counter() - t0)

    timing = {"load": load_time, "prefill": prefill_time, "tps": tps}
    success = tokens[:5] == expected_tokens
    status = "✓" if success else f"✗ got {tokens[:5]}"
    load_str = fmt_delta(load_time, prev.get("load") if prev else None)
    prefill_str = fmt_delta(prefill_time, prev.get("prefill") if prev else None)
    tps_str = fmt_delta(tps, prev.get("tps") if prev else None, higher_better=True)
    print(f"  load: {load_str}s  prefill: {prefill_str}s  gen: {tps_str} tok/s  {status}")
    return success, timing
  except Exception as e:
    print(f"  ✗ Error: {e}")
    return False, {}
  finally:
    gc.collect()

def append_to_history(timings: dict[str, dict]):
  """Append results to markdown history file."""
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
  if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text("# Smoke Test History\n\n| timestamp | model | load | prefill | tok/s |\n|-----------|-------|------|---------|-------|\n")
  with open(HISTORY_FILE, "a") as f:
    for model, t in timings.items():
      f.write(f"| {timestamp} | {model} | {t['load']:.1f} | {t['prefill']:.1f} | {t['tps']:.2f} |\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("models", nargs="*", help="Models to test (default: all)")
  parser.add_argument("-youtu", action="store_true", help="Compare youtu Q8 vs Q4")
  args = parser.parse_args()

  if args.youtu:
    to_test = ["youtu", "youtu-q4"]
  elif args.models:
    to_test = args.models
  else:
    to_test = ["youtu", "deepseek", "glm"]  # default excludes youtu-q4
  results, timings = [], {}
  for name in to_test:
    model_key, expected = SMOKE_MODELS[name]
    prev = parse_last_row(model_key)
    success, timing = test_model(model_key, expected, prev)
    results.append(success)
    if timing: timings[model_key] = timing

  if timings:
    append_to_history(timings)

  print(f"Passed: {sum(results)}/{len(results)}")
  exit(0 if all(results) else 1)
