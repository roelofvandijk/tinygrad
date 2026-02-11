#!/usr/bin/env python3
"""Realistic decode benchmarks for GLM-4.7-Flash Q4_0.

Tests actual scheduler fusion behavior with JIT (MetalGraph ICB batching).
Builds one MLA block from GGUF metadata so tensor shapes match the real model.
"""
import argparse, os, pathlib, time
from dataclasses import dataclass

os.environ.setdefault("DEVICE", "METAL")
os.environ.setdefault("BEAM_REDUCE_ONLY", "1")

from tinygrad import Tensor, UOp, nn, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import GlobalCounters, Context
from tinygrad.engine.jit import TinyJit
from tinygrad.apps.llm import models
from tinygrad.apps.mla import MLATransformerBlock, merge_gate_up_experts, merge_gate_up_shared_expert, load_mla_params_from_gguf
from tinygrad.apps.quantized import QuantizedLinear, QuantizedExpertWeights
from tinygrad.apps.rope import precompute_freqs_cis, load_yarn_params_from_gguf
from tinygrad.nn.state import GGML_QUANT_INFO

DEFAULT_BENCH_MODEL = "glm-4.7:flash-unsloth-Q4_0"
MLA_SPLIT_POINTS = [
  "attn_norm", "attn_q_proj", "attn_kv_proj", "attn_q_cat", "attn_k_cache_view", "attn_scores", "attn_softmax",
  "attn_ctx", "attn_out_proj", "ffn_norm", "ffn_gate_up_exps", "ffn_gated", "ffn_expert_out", "ffn_moe_combine",
  "ffn_shexp_gate_up", "ffn_shexp_down", "ffn_moe_plus_shexp",
]
SPLIT_POINTS = MLA_SPLIT_POINTS

def _median(vals: list[float]) -> float:
  v = sorted(vals)
  n = len(v)
  return v[n//2] if n % 2 else 0.5 * (v[n//2 - 1] + v[n//2])

def _quantile(vals: list[float], q: float) -> float:
  assert 0 <= q <= 1 and len(vals) > 0
  v = sorted(vals)
  i = int((len(v) - 1) * q)
  return v[i]

def _stats(vals: list[float]) -> dict[str, float]:
  med = _median(vals)
  p10, p25 = _quantile(vals, 0.10), _quantile(vals, 0.25)
  p75, p90 = _quantile(vals, 0.75), _quantile(vals, 0.90)
  mad = _median([abs(x - med) for x in vals])
  avg = sum(vals) / len(vals)
  return {"median": med, "best": min(vals), "avg": avg, "p10": p10, "p90": p90, "iqr": p75 - p25, "mad": mad}

@dataclass(frozen=True)
class BenchCfg:
  model_name: str
  dim: int
  n_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_nope_head_dim: int
  qk_rope_head_dim: int
  v_head_dim: int
  num_experts: int
  num_experts_per_tok: int
  n_shared_experts: int
  moe_hidden_dim: int
  norm_eps: float
  max_context: int
  rope_theta: float
  mscale: float
  num_blocks: int
  leading_dense_blocks: int
  expert_first_in_memory: bool
  shexp_gate_type: int
  shexp_up_type: int
  shexp_down_type: int

_BENCH_MASK_CACHE: dict[tuple[str, int], tuple[BenchCfg, MLATransformerBlock]] = {}

def _resolve_model_source(name_or_path: str) -> str:
  model_src = models.get(name_or_path, name_or_path)
  if isinstance(model_src, str) and model_src.startswith("http"):
    local = pathlib.Path("models") / pathlib.Path(model_src).name
    if local.is_file(): return str(local.resolve())
  p = pathlib.Path(name_or_path)
  if p.exists(): return str(p.resolve())
  return model_src

def load_bench_cfg(model_name: str, max_context: int) -> BenchCfg:
  src = _resolve_model_source(model_name)
  kv, _, quantized_tensors = nn.state.gguf_load(Tensor.from_url(src).to(None), quantized=True)
  arch = kv["general.architecture"]
  mla = load_mla_params_from_gguf(kv, arch)
  rope_theta = kv[f"{arch}.rope.freq_base"]
  _, mscale, _ = load_yarn_params_from_gguf(kv, arch, rope_theta)
  ctx = min(max_context, kv[f"{arch}.context_length"])

  expert_first = True
  for name, (_, _, _, efim) in quantized_tensors.items():
    if "ffn_gate_exps.weight" in name:
      expert_first = efim
      break

  # Block 0 is dense in GLM; use block 1 to pick the representative MoE quantization types.
  block1_qtypes = {name: info[2] for name, info in quantized_tensors.items() if name.startswith("blk.1.")}
  shexp_gate_type = block1_qtypes.get("blk.1.ffn_gate_shexp.weight", 2)
  shexp_up_type = block1_qtypes.get("blk.1.ffn_up_shexp.weight", 2)
  shexp_down_type = block1_qtypes.get("blk.1.ffn_down_shexp.weight", 2)

  return BenchCfg(
    model_name=model_name,
    dim=kv[f"{arch}.embedding_length"],
    n_heads=kv[f"{arch}.attention.head_count"],
    q_lora_rank=mla["q_lora_rank"],
    kv_lora_rank=mla["kv_lora_rank"],
    qk_nope_head_dim=mla["qk_nope_head_dim"],
    qk_rope_head_dim=mla["qk_rope_head_dim"],
    v_head_dim=mla["v_head_dim"],
    num_experts=kv.get(f"{arch}.expert_count", 0),
    num_experts_per_tok=kv.get(f"{arch}.expert_used_count", 0),
    n_shared_experts=mla["n_shared_experts"],
    moe_hidden_dim=mla["moe_hidden_dim"],
    norm_eps=kv[f"{arch}.attention.layer_norm_rms_epsilon"],
    max_context=ctx,
    rope_theta=rope_theta,
    mscale=mscale,
    num_blocks=kv[f"{arch}.block_count"],
    leading_dense_blocks=mla["leading_dense_blocks"],
    expert_first_in_memory=expert_first,
    shexp_gate_type=shexp_gate_type,
    shexp_up_type=shexp_up_type,
    shexp_down_type=shexp_down_type,
  )

def make_quant_blocks(n_blocks: int, ggml_type: int) -> Tensor:
  _, bytes_per_block, _ = GGML_QUANT_INFO[ggml_type]
  blocks = Tensor.randint(n_blocks, bytes_per_block, high=256, dtype=dtypes.uint8)
  # Keep scale fields finite/stable so kernels benchmark compute+memory, not NaN handling.
  if ggml_type == 2:     # Q4_0: [d:fp16][packed]
    blocks[:, 0].assign(0)
    blocks[:, 1].assign(0x3C)  # d=1.0
  elif ggml_type == 13:  # Q5_K: [d:fp16][dmin:fp16]...
    blocks[:, 0].assign(0)
    blocks[:, 1].assign(0x3C)  # d=1.0
    blocks[:, 2].assign(0)
    blocks[:, 3].assign(0)     # dmin=0.0
  elif ggml_type == 14:  # Q6_K: [...][scales:int8*16][d:fp16]
    blocks[:, 192:208].assign(1)  # small int8 scales
    blocks[:, 208].assign(0)
    blocks[:, 209].assign(0x3C)   # d=1.0
  return blocks.contiguous().realize()

def make_quantized_linear(out_features: int, in_features: int, ggml_type: int) -> QuantizedLinear:
  el_per_block, _, _ = GGML_QUANT_INFO[ggml_type]
  bpr = in_features // el_per_block
  return QuantizedLinear(make_quant_blocks(out_features * bpr, ggml_type), (out_features, in_features), ggml_type=ggml_type)

def make_q4_0_expert_weights(num_experts: int, out_features: int, in_features: int, expert_first_in_memory: bool) -> QuantizedExpertWeights:
  bpr = in_features // 32
  return QuantizedExpertWeights(
    make_quant_blocks(num_experts * out_features * bpr, 2),
    (num_experts, out_features, in_features),
    ggml_type=2,
    expert_first_in_memory=expert_first_in_memory,
  )

def build_block(cfg: BenchCfg) -> MLATransformerBlock:
  block = MLATransformerBlock(
    dim=cfg.dim,
    hidden_dim=cfg.moe_hidden_dim,
    n_heads=cfg.n_heads,
    norm_eps=cfg.norm_eps,
    max_context=cfg.max_context,
    q_lora_rank=cfg.q_lora_rank,
    kv_lora_rank=cfg.kv_lora_rank,
    qk_nope_head_dim=cfg.qk_nope_head_dim,
    qk_rope_head_dim=cfg.qk_rope_head_dim,
    v_head_dim=cfg.v_head_dim,
    num_experts=cfg.num_experts,
    num_experts_per_tok=cfg.num_experts_per_tok,
    n_shared_experts=cfg.n_shared_experts,
    moe_hidden_dim=cfg.moe_hidden_dim,
    expert_gating_func=2,
    expert_weights_norm=True,
    expert_weights_scale=1.8,
    mscale=cfg.mscale,
  )

  # Attention linears -> Q4_0
  block.attn_q_a = make_quantized_linear(cfg.q_lora_rank, cfg.dim, 2)
  block.attn_q_b = make_quantized_linear(cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank, 2)
  block.attn_kv_a_mqa = make_quantized_linear(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.dim, 2)
  block.attn_output = make_quantized_linear(cfg.dim, cfg.n_heads * cfg.v_head_dim, 2)

  # Expert weights -> Q4_0
  block.ffn_gate_exps = make_q4_0_expert_weights(cfg.num_experts, cfg.moe_hidden_dim, cfg.dim, cfg.expert_first_in_memory)
  block.ffn_up_exps = make_q4_0_expert_weights(cfg.num_experts, cfg.moe_hidden_dim, cfg.dim, cfg.expert_first_in_memory)
  block.ffn_down_exps = make_q4_0_expert_weights(cfg.num_experts, cfg.dim, cfg.moe_hidden_dim, cfg.expert_first_in_memory)

  # Shared expert -> Q4_0
  block.ffn_gate_shexp = make_quantized_linear(cfg.moe_hidden_dim, cfg.dim, cfg.shexp_gate_type)
  block.ffn_up_shexp = make_quantized_linear(cfg.moe_hidden_dim, cfg.dim, cfg.shexp_up_type)
  block.ffn_down_shexp = make_quantized_linear(cfg.dim, cfg.moe_hidden_dim, cfg.shexp_down_type)

  # Router (fp16 -> fp32)
  block.ffn_gate_inp.weight = Tensor.randn(cfg.num_experts, cfg.dim).cast(dtypes.float16).realize()
  block.ffn_gate_inp_f32 = block.ffn_gate_inp.weight.float().realize()

  # Norms
  for mod in [block.attn_norm, block.ffn_norm, block.attn_q_a_norm, block.attn_kv_a_norm]:
    mod.weight = mod.weight.realize()

  # Per-head weights (fp16, not quantized)
  block.attn_k_b.weight = Tensor.randn(cfg.n_heads, cfg.kv_lora_rank, cfg.qk_nope_head_dim).cast(dtypes.float16).realize()
  block.attn_v_b.weight = Tensor.randn(cfg.n_heads, cfg.v_head_dim, cfg.kv_lora_rank).cast(dtypes.float16).realize()
  block.exp_probs_b.bias = Tensor.zeros(cfg.num_experts).realize()

  # Apply merges (same as real model)
  merge_gate_up_experts(block)
  if os.getenv("BENCH_MERGE_SHEXP", "1") == "1":
    merge_gate_up_shared_expert(block)

  # RoPE frequencies
  freqs_cis_cache = precompute_freqs_cis(cfg.qk_rope_head_dim, cfg.max_context, cfg.rope_theta).realize()
  block.freqs_cos_cache = freqs_cis_cache[..., 0].reshape(1, 1, cfg.max_context, -1).cast(dtypes.float16).realize()
  block.freqs_sin_cache = freqs_cis_cache[..., 1].reshape(1, 1, cfg.max_context, -1).cast(dtypes.float16).realize()

  return block

def bench_full_block(block: MLATransformerBlock, cfg: BenchCfg, n_iter: int = 20):
  x = Tensor.randn(1, 1, cfg.dim).cast(dtypes.float16).realize()
  v_sp = UOp.variable("start_pos", 0, cfg.max_context - 1)

  @TinyJit
  def run_block(x, start_pos):
    return block(x, start_pos).contiguous()

  print(f"── full block with JIT ({n_iter} iterations) ──")
  out = block(x, start_pos=0).realize()
  Device.default.synchronize()

  for i in range(3):
    GlobalCounters.reset()
    out = run_block(x, v_sp.bind(1 + i)).realize()
    Device.default.synchronize()
    print(f"  warmup {i} (pos={1+i}): {GlobalCounters.kernel_count} kernels")

  times = []
  for i in range(n_iter):
    Device.default.synchronize()
    st = time.perf_counter()
    out = run_block(x, v_sp.bind(10 + i)).realize()
    Device.default.synchronize()
    times.append((time.perf_counter() - st) * 1000)

  st = _stats(times)
  moe_blocks = max(cfg.num_blocks - cfg.leading_dense_blocks, 1)
  print(f"  median: {st['median']:.2f} ms  best: {st['best']:.2f} ms  avg: {st['avg']:.2f} ms")
  print(f"  p10/p90: {st['p10']:.2f}/{st['p90']:.2f} ms  IQR: {st['iqr']:.2f} ms  MAD: {st['mad']:.2f} ms")
  est_tok_s = 1000 / (st["median"] * moe_blocks)
  print(f"  → {1000/st['median']:.1f} blocks/s  (est. {est_tok_s:.1f} tok/s × {moe_blocks} MoE blocks)")
  return st["median"]

def bench_decode_stack(block: MLATransformerBlock, cfg: BenchCfg, n_iter: int = 8, decode_blocks: int | None = None,
                      quiet: bool = False, warmups: int = 2):
  x = Tensor.randn(1, 1, cfg.dim).cast(dtypes.float16).realize()
  v_sp = UOp.variable("start_pos", 0, cfg.max_context - 1)
  moe_blocks = max(cfg.num_blocks - cfg.leading_dense_blocks, 1)
  n_blocks = moe_blocks if decode_blocks is None else decode_blocks

  @TinyJit
  def run_decode(x, start_pos):
    for _ in range(n_blocks):
      x = block(x, start_pos)
    return x.contiguous()

  if not quiet: print(f"\n── decode stack with JIT ({n_blocks} blocks, {n_iter} iterations) ──")
  out = run_decode(x, v_sp.bind(1)).realize()
  Device.default.synchronize()
  for i in range(warmups):
    GlobalCounters.reset()
    out = run_decode(x, v_sp.bind(2 + i)).realize()
    Device.default.synchronize()
    if not quiet: print(f"  warmup {i}: {GlobalCounters.kernel_count} kernels")

  times = []
  for i in range(n_iter):
    Device.default.synchronize()
    st = time.perf_counter()
    out = run_decode(x, v_sp.bind(10 + i)).realize()
    Device.default.synchronize()
    times.append((time.perf_counter() - st) * 1000)

  st = _stats(times)
  if not quiet:
    print(f"  median: {st['median']:.2f} ms  best: {st['best']:.2f} ms  avg: {st['avg']:.2f} ms")
    print(f"  p10/p90: {st['p10']:.2f}/{st['p90']:.2f} ms  IQR: {st['iqr']:.2f} ms  MAD: {st['mad']:.2f} ms")
    print(f"  → decode proxy: {1000/st['median']:.2f} tok/s")
  return st["median"]

def bench_feed_forward(block: MLATransformerBlock, cfg: BenchCfg, n_iter: int = 20):
  x = Tensor.randn(1, 1, cfg.dim).cast(dtypes.float16).realize()
  print(f"\n── feed_forward only, no JIT ({n_iter} iterations) ──")

  for i in range(3):
    GlobalCounters.reset()
    out = block._feed_forward(x).realize()
    Device.default.synchronize()
    print(f"  warmup {i}: {GlobalCounters.kernel_count} kernels")

  times = []
  for _ in range(n_iter):
    Device.default.synchronize()
    st = time.perf_counter()
    out = block._feed_forward(x).realize()
    Device.default.synchronize()
    times.append((time.perf_counter() - st) * 1000)

  st = _stats(times)
  print(f"  median: {st['median']:.2f} ms  best: {st['best']:.2f} ms  avg: {st['avg']:.2f} ms")
  print(f"  p10/p90: {st['p10']:.2f}/{st['p90']:.2f} ms  IQR: {st['iqr']:.2f} ms  MAD: {st['mad']:.2f} ms")
  return st["median"]

def run_benchmark(model: str = DEFAULT_BENCH_MODEL, max_context: int = 4096, count: int = 20, rounds: int = 1) -> tuple[float, float, float]:
  print(f"Loading GGUF metadata for {model}...")
  cfg = load_bench_cfg(model, max_context)
  print(
    f"Config: dim={cfg.dim} moe_hidden={cfg.moe_hidden_dim} experts={cfg.num_experts} x {cfg.num_experts_per_tok} "
    f"heads={cfg.n_heads} ctx={cfg.max_context} expert_first={cfg.expert_first_in_memory}"
  )
  print("Building block with Q4_0 weights...")
  # Setup kernels (rand/copy/init) are not decode hot path; skip BEAM here to keep iteration loop fast.
  with Context(BEAM=0):
    block = build_block(cfg)
  print(f"Block built. Device: {Device.DEFAULT}\n")
  full, decode, ff = [], [], []
  decode_iters = max(4, count // 4)
  for r in range(rounds):
    if rounds > 1: print(f"=== Round {r+1}/{rounds} ===")
    full.append(bench_full_block(block, cfg, count))
    decode.append(bench_decode_stack(block, cfg, decode_iters))
    ff.append(bench_feed_forward(block, cfg, count))
    if rounds > 1 and r != rounds - 1: print()
  if rounds > 1:
    fstats, dstats, ffstats = _stats(full), _stats(decode), _stats(ff)
    print("\n== Aggregate (median of round medians) ==")
    print(f"  full block: {fstats['median']:.2f} ms  p10/p90 {fstats['p10']:.2f}/{fstats['p90']:.2f}  IQR {fstats['iqr']:.2f}")
    print(f"  decode stack: {dstats['median']:.2f} ms  p10/p90 {dstats['p10']:.2f}/{dstats['p90']:.2f}  IQR {dstats['iqr']:.2f}")
    print(f"  feed_forward: {ffstats['median']:.2f} ms  p10/p90 {ffstats['p10']:.2f}/{ffstats['p90']:.2f}  IQR {ffstats['iqr']:.2f}")
    return fstats["median"], dstats["median"], ffstats["median"]
  return full[0], decode[0], ff[0]

def _get_cached_block(model: str, max_context: int) -> tuple[BenchCfg, MLATransformerBlock]:
  key = (model, max_context)
  if key in _BENCH_MASK_CACHE: return _BENCH_MASK_CACHE[key]
  cfg = load_bench_cfg(model, max_context)
  with Context(BEAM=0):
    block = build_block(cfg)
  _BENCH_MASK_CACHE[key] = (cfg, block)
  return cfg, block

def bench_mask(mask: int, iters: int = 8, beam: int = 0, model: str = DEFAULT_BENCH_MODEL, max_context: int = 4096) -> float:
  cfg, block = _get_cached_block(model, max_context)
  os.environ["MLA_SPLIT_MASK"] = str(mask)
  with Context(BEAM=beam):
    return bench_decode_stack(block, cfg, n_iter=max(4, iters), quiet=True, warmups=1)

def run_fusion_search(model: str = DEFAULT_BENCH_MODEL, max_context: int = 4096, count: int = 16, passes: int = 2,
                      eps_ms: float = 0.10):
  print(f"Loading GGUF metadata for {model}...")
  cfg = load_bench_cfg(model, max_context)
  print("Building block with Q4_0 weights...")
  with Context(BEAM=0):
    block = build_block(cfg)
  print(f"Block built. Device: {Device.DEFAULT}")

  decode_iters = max(4, count // 4)
  n_bits = len(MLA_SPLIT_POINTS)
  all_split = (1 << n_bits) - 1

  def measure(mask: int) -> float:
    os.environ["MLA_SPLIT_MASK"] = str(mask)
    t = bench_decode_stack(block, cfg, decode_iters, quiet=True, warmups=1)
    print(f"  mask=0x{mask:05x} decode={t:.2f} ms ({1000/t:.2f} tok/s proxy)")
    return t

  print(f"\nFusion search over {n_bits} split points (all-split -> greedy re-fuse):")
  for i, name in enumerate(MLA_SPLIT_POINTS): print(f"  bit {i:2d}: {name}")
  best_mask, best_t = all_split, measure(all_split)
  print(f"Start (all split): 0x{best_mask:05x}, {best_t:.2f} ms")

  for p in range(passes):
    print(f"\nPass {p+1}/{passes}")
    improved, pass_best_t, pass_best_mask, pass_best_bit = False, best_t, best_mask, -1
    for bit in range(n_bits):
      if ((best_mask >> bit) & 1) == 0: continue
      cand_mask = best_mask & ~(1 << bit)
      cand_t = measure(cand_mask)
      if cand_t + eps_ms < pass_best_t:
        pass_best_t, pass_best_mask, pass_best_bit = cand_t, cand_mask, bit
        improved = True
    if not improved:
      print("  no improving fuse step in this pass")
      break
    print(f"  keep: fuse bit {pass_best_bit} ({MLA_SPLIT_POINTS[pass_best_bit]}) -> {pass_best_t:.2f} ms")
    best_t, best_mask = pass_best_t, pass_best_mask

  os.environ["MLA_SPLIT_MASK"] = str(best_mask)
  fused_bits = [i for i in range(n_bits) if ((best_mask >> i) & 1) == 0]
  split_bits = [i for i in range(n_bits) if ((best_mask >> i) & 1) == 1]
  print("\nBest mask summary:")
  print(f"  mask=0x{best_mask:05x} decode={best_t:.2f} ms ({1000/best_t:.2f} tok/s proxy)")
  print(f"  fused bits ({len(fused_bits)}): {[MLA_SPLIT_POINTS[i] for i in fused_bits]}")
  print(f"  split bits ({len(split_bits)}): {[MLA_SPLIT_POINTS[i] for i in split_bits]}")
  return best_mask, best_t

def run_fusion_beam_search(model: str = DEFAULT_BENCH_MODEL, max_context: int = 4096, count: int = 8, beam_amt: int = 4,
                           max_steps: int = 4, eps_ms: float = 0.05):
  print(f"Loading GGUF metadata for {model}...")
  cfg = load_bench_cfg(model, max_context)
  print("Building block with Q4_0 weights...")
  with Context(BEAM=0):
    block = build_block(cfg)
  print(f"Block built. Device: {Device.DEFAULT}")

  decode_iters = max(4, count // 2)
  n_bits = len(MLA_SPLIT_POINTS)
  seen: set[int] = set()

  def measure(mask: int) -> float:
    os.environ["MLA_SPLIT_MASK"] = str(mask)
    with Context(BEAM=beam_amt):
      t = bench_decode_stack(block, cfg, decode_iters, quiet=True, warmups=1)
    print(f"  mask=0x{mask:05x} decode={t:.2f} ms ({1000/t:.2f} tok/s proxy)")
    return t

  print(f"\nFusion+BEAM search ({n_bits} split points, BEAM={beam_amt})")
  for i, name in enumerate(MLA_SPLIT_POINTS): print(f"  bit {i:2d}: {name}")

  cur_mask = 0
  cur_t = measure(cur_mask)
  seen.add(cur_mask)
  print(f"Start (all fused): 0x{cur_mask:05x}, {cur_t:.2f} ms")

  for step in range(max_steps):
    print(f"\nStep {step+1}/{max_steps}: try 1-bit neighbors")
    best_mask, best_t = cur_mask, cur_t
    for bit in range(n_bits):
      cand_mask = cur_mask ^ (1 << bit)
      if cand_mask in seen: continue
      cand_t = measure(cand_mask)
      seen.add(cand_mask)
      if cand_t + eps_ms < best_t:
        best_mask, best_t = cand_mask, cand_t
    if best_mask == cur_mask:
      print("  no improving neighbor")
      break
    changed = cur_mask ^ best_mask
    bit = changed.bit_length() - 1
    action = "split" if ((best_mask >> bit) & 1) else "fuse"
    print(f"  keep: {action} bit {bit} ({MLA_SPLIT_POINTS[bit]}) -> {best_t:.2f} ms")
    cur_mask, cur_t = best_mask, best_t

  os.environ["MLA_SPLIT_MASK"] = str(cur_mask)
  fused_bits = [i for i in range(n_bits) if ((cur_mask >> i) & 1) == 0]
  split_bits = [i for i in range(n_bits) if ((cur_mask >> i) & 1) == 1]
  print("\nBest mask summary:")
  print(f"  mask=0x{cur_mask:05x} decode={cur_t:.2f} ms ({1000/cur_t:.2f} tok/s proxy)")
  print(f"  fused bits ({len(fused_bits)}): {[MLA_SPLIT_POINTS[i] for i in fused_bits]}")
  print(f"  split bits ({len(split_bits)}): {[MLA_SPLIT_POINTS[i] for i in split_bits]}")
  return cur_mask, cur_t

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("count", nargs="?", type=int, default=20, help="Benchmark iterations")
  parser.add_argument("--rounds", type=int, default=1, help="Repeat benchmark rounds and aggregate medians")
  parser.add_argument("--model", default=DEFAULT_BENCH_MODEL, choices=[DEFAULT_BENCH_MODEL], help="Model to benchmark")
  parser.add_argument("--max_context", type=int, default=4096, help="Context length to emulate")
  parser.add_argument("--fusion_search", action="store_true", help="Greedy split->fuse search using decode-stack proxy")
  parser.add_argument("--fusion_passes", type=int, default=2, help="Fusion-search passes")
  parser.add_argument("--fusion_eps_ms", type=float, default=0.10, help="Min decode improvement (ms) to keep a fuse step")
  parser.add_argument("--fusion_beam_search", action="store_true", help="Greedy 1-bit split/fuse search with BEAM inside each candidate")
  parser.add_argument("--beam_amt", type=int, default=4, help="BEAM width for --fusion_beam_search")
  parser.add_argument("--fusion_steps", type=int, default=4, help="Max greedy neighbor steps for --fusion_beam_search")
  args = parser.parse_args()
  if args.fusion_beam_search:
    run_fusion_beam_search(args.model, args.max_context, args.count, args.beam_amt, args.fusion_steps, args.fusion_eps_ms)
  elif args.fusion_search:
    run_fusion_search(args.model, args.max_context, args.count, args.fusion_passes, args.fusion_eps_ms)
  else:
    run_benchmark(args.model, args.max_context, args.count, args.rounds)
