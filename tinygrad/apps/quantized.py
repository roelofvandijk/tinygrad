from __future__ import annotations
from tinygrad import Tensor, UOp, nn, getenv
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.nn.state import GGML_QUANT_INFO

def custom_q4_0_linear(out:UOp, x:UOp, scale:UOp, packed:UOp) -> UOp:
  # out: (N, O), x: (N, I), scale: (O, I//32, 1), packed: (O, I//32, 16)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(scale.shape) == 3 and len(packed.shape) == 3
  assert all(isinstance(s, int) for s in out.shape+x.shape+scale.shape+packed.shape), "custom q4_0 kernel requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  bpr = I // 32
  assert x.shape[0] == N and scale.shape == (O, bpr, 1) and packed.shape == (O, bpr, 16)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  # Process two packed bytes (4 q4 values) per reduction step to reduce loop trip count.
  # This keeps math identical to the byte-wise formulation but improves reduction shape.
  r = UOp.range(bpr*8, 2, axis_type=AxisType.REDUCE)
  br, jb = r//8, (r%8)*2

  s = scale[o, br, 0].cast(dtypes.float)
  q0, q1 = packed[o, br, jb], packed[o, br, jb+1]
  q0_lo = (q0 & 0xF).cast(dtypes.float) - 8.0
  q0_hi = (q0 >> 4).cast(dtypes.float) - 8.0
  q1_lo = (q1 & 0xF).cast(dtypes.float) - 8.0
  q1_hi = (q1 >> 4).cast(dtypes.float) - 8.0
  x0 = x[n, br*32 + jb].cast(dtypes.float)
  x1 = x[n, br*32 + jb + 1].cast(dtypes.float)
  x2 = x[n, br*32 + jb + 16].cast(dtypes.float)
  x3 = x[n, br*32 + jb + 17].cast(dtypes.float)
  # Keep this as an explicit REDUCE (instead of AFTER accumulation), so GROUP/GROUPTOP
  # legalization can reason about reduction domains correctly.
  dot = s * (q0_lo * x0 + q1_lo * x1 + q0_hi * x2 + q1_hi * x3)
  acc = dot.reduce(r, arg=Ops.ADD, dtype=dtypes.float)
  return out[n, o].store(acc.cast(out.dtype.base)).end(n, o).sink(
    arg=KernelInfo(name=f"custom_q4_0_linear_{N}_{O}_{I}", opts_to_apply=_q4_0_linear_opts(N, O, I)))

def custom_q4_0_linear_split(partial:UOp, x:UOp, scale:UOp, packed:UOp) -> UOp:
  """
  Two-stage DSL reduction for the dense Q4_0 hotspot:
    stage 1 (this kernel): compute chunked partial dot-products over reduction chunks
    stage 2 (regular Tensor sum): reduce partial chunks to final output

  This mirrors the MSL lane-parallel idea using pure DSL: increase parallel work-items
  over the reduction dimension (via chunk index `g`) instead of one long serial loop/thread.
  """
  assert len(partial.shape) == 3 and len(x.shape) == 2 and len(scale.shape) == 3 and len(packed.shape) == 3
  assert all(isinstance(s, int) for s in partial.shape+x.shape+scale.shape+packed.shape), "custom q4_0 split kernel requires static shapes"
  N, O, G = partial.shape
  I = x.shape[1]
  bpr = I // 32
  assert bpr % G == 0, f"bpr {bpr} must be divisible by chunk groups {G}"
  chunk_bpr = bpr // G
  assert x.shape[0] == N and scale.shape == (O, bpr, 1) and packed.shape == (O, bpr, 16)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  g = UOp.range(G, 2)
  # Same 2-byte (4 q4 values) step as custom_q4_0_linear, but scoped to one chunk.
  r = UOp.range(chunk_bpr*8, 3, axis_type=AxisType.REDUCE)
  br = g*chunk_bpr + (r//8)
  jb = (r%8)*2

  s = scale[o, br, 0].cast(dtypes.float)
  q0, q1 = packed[o, br, jb], packed[o, br, jb+1]
  q0_lo = (q0 & 0xF).cast(dtypes.float) - 8.0
  q0_hi = (q0 >> 4).cast(dtypes.float) - 8.0
  q1_lo = (q1 & 0xF).cast(dtypes.float) - 8.0
  q1_hi = (q1 >> 4).cast(dtypes.float) - 8.0
  x0 = x[n, br*32 + jb].cast(dtypes.float)
  x1 = x[n, br*32 + jb + 1].cast(dtypes.float)
  x2 = x[n, br*32 + jb + 16].cast(dtypes.float)
  x3 = x[n, br*32 + jb + 17].cast(dtypes.float)
  # Same explicit REDUCE form as custom_q4_0_linear for correctness-safe grouped scheduling.
  dot = s * (q0_lo * x0 + q1_lo * x1 + q0_hi * x2 + q1_hi * x3)
  acc = dot.reduce(r, arg=Ops.ADD, dtype=dtypes.float)
  return partial[n, o, g].store(acc.cast(partial.dtype.base)).end(n, o, g).sink(
    arg=KernelInfo(name=f"custom_q4_0_linear_split_{N}_{O}_{I}_g{G}", opts_to_apply=_q4_0_linear_split_opts(N, O, I, G)))

def custom_q4_0_linear_vec2(out:UOp, x:UOp, scale:UOp, packed:UOp) -> UOp:
  # Two-output-row kernel with one vector store.
  # This is a structural DSL variant mirroring the MSL NR idea while keeping one STORE path
  # (avoids multi-store control-flow issues seen in earlier NR2 attempts).
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(scale.shape) == 3 and len(packed.shape) == 3
  assert all(isinstance(s, int) for s in out.shape+x.shape+scale.shape+packed.shape), "custom q4_0 kernel requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  bpr = I // 32
  assert O % 2 == 0 and x.shape[0] == N and scale.shape == (O, bpr, 1) and packed.shape == (O, bpr, 16)

  n = UOp.range(N, 0)
  o2 = UOp.range(O//2, 1)
  o0, o1 = o2*2, o2*2+1
  r = UOp.range(bpr*8, 2, axis_type=AxisType.REDUCE)
  br, jb = r//8, (r%8)*2

  acc = UOp.placeholder((1,), dtypes.float.vec(2), 0, addrspace=AddrSpace.REG)
  acc = acc.after(n, o2)[0].set(UOp.const(dtypes.float.vec(2), (0.0, 0.0)))

  s0 = scale[o0, br, 0].cast(dtypes.float)
  s1 = scale[o1, br, 0].cast(dtypes.float)
  q00, q01 = packed[o0, br, jb], packed[o0, br, jb+1]
  q10, q11 = packed[o1, br, jb], packed[o1, br, jb+1]
  q00_lo = (q00 & 0xF).cast(dtypes.float) - 8.0
  q00_hi = (q00 >> 4).cast(dtypes.float) - 8.0
  q01_lo = (q01 & 0xF).cast(dtypes.float) - 8.0
  q01_hi = (q01 >> 4).cast(dtypes.float) - 8.0
  q10_lo = (q10 & 0xF).cast(dtypes.float) - 8.0
  q10_hi = (q10 >> 4).cast(dtypes.float) - 8.0
  q11_lo = (q11 & 0xF).cast(dtypes.float) - 8.0
  q11_hi = (q11 >> 4).cast(dtypes.float) - 8.0
  x0 = x[n, br*32 + jb].cast(dtypes.float)
  x1 = x[n, br*32 + jb + 1].cast(dtypes.float)
  x2 = x[n, br*32 + jb + 16].cast(dtypes.float)
  x3 = x[n, br*32 + jb + 17].cast(dtypes.float)
  dot0 = s0 * (q00_lo * x0 + q01_lo * x1 + q00_hi * x2 + q01_hi * x3)
  dot1 = s1 * (q10_lo * x0 + q11_lo * x1 + q10_hi * x2 + q11_hi * x3)
  acc = acc[0].set(acc.after(r)[0] + dot0.vectorize(dot1), end=r)

  # Explicit global addrspace keeps pointer-cast rendering legal on Metal.
  out_ptr = out[n, o0].cast(out.dtype.base.vec(2).ptr(size=out.dtype.size, addrspace=AddrSpace.GLOBAL))
  out_vec = acc[0].cast(out.dtype.base.vec(2))
  return out_ptr.store(out_vec).end(n, o2).sink(
    arg=KernelInfo(name=f"custom_q4_0_linear_vec2_{N}_{O}_{I}", opts_to_apply=_q4_0_linear_opts(N, O, I)))

def _q4_0_linear_opts(n_rows:int, out_features:int, in_features:int):
  # Fast, correctness-safe defaults for decode-hot dense Q4 shapes.
  # These are intentionally fixed (no env tuning path) so runtime behavior is deterministic.
  if n_rows == 1 and out_features == 2048 and in_features == 5120:
    # Explicit REDUCE form in custom_q4_0_linear makes grouped-reduce legal and correct here.
    # For the decode-hot 1x2048x5120 shape, smaller LOCAL (8) with GROUPTOP(4) gave better
    # end-to-end latency than LOCAL(16) in our microbench A/B, while keeping numerical parity.
    # This is currently the fastest known DSL schedule for this kernel family.
    return (Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 1, 4))
  if n_rows == 1 and out_features == 5120 and in_features == 768:
    return (Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUP, 1, 8))
  if n_rows == 1 and out_features == 768 and in_features == 2048:
    return (Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 1, 8))
  if n_rows == 1 and out_features == 10240 and in_features == 2048:
    return (Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUP, 1, 8))
  if n_rows == 1 and out_features == 2048 and in_features == 10240:
    return (Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUP, 1, 8))
  return ()

def _q4_0_linear_split_opts(n_rows:int, out_features:int, in_features:int, groups:int):
  if n_rows == 1 and (out_features, in_features) in {(2048, 5120), (5120, 768), (768, 2048)}:
    return (Opt(OptOps.LOCAL, 0, 16),)
  return ()

def _q4_0_linear_split_groups(n_rows:int, out_features:int, in_features:int) -> int:
  # Split-K dense path stays available but is currently not default.
  # Single-kernel grouped reduction now benchmarks better for decode-hot shapes.
  return 1

def _q4_0_mul_mat_id_opts(n_sel:int, out_features:int, in_features:int):
  # Decode-specialized defaults (n_sel=4) for Q4 expert matmul-id.
  if n_sel == 4 and out_features == 3072 and in_features == 2048:
    # GROUPTOP gave the best kernel-level latency on this dominant gate/up shape.
    return (Opt(OptOps.LOCAL, 1, 8), Opt(OptOps.GROUPTOP, 1, 8))
  if n_sel == 4 and out_features == 2048 and in_features == 1536:
    # Explicit REDUCE form enables correctness-safe GROUPTOP here and it's significantly faster
    # than GROUP in microbench for the down expert kernel shape.
    return (Opt(OptOps.LOCAL, 1, 8), Opt(OptOps.GROUPTOP, 1, 8))
  return ()

def _q4_0_mul_mat_id_split_opts(n_sel:int, out_features:int, in_features:int, groups:int):
  if n_sel == 4 and (out_features, in_features) in {(3072, 2048), (2048, 1536)}:
    return (Opt(OptOps.LOCAL, 1, 16),)
  return ()

def _q4_0_mul_mat_id_split_groups(n_sel:int, out_features:int, in_features:int) -> int:
  # Split-K MoE path stays available, but grouped single-kernel path currently benchmarks better.
  return 1

def _fp16_mul_mat_id_opts(n_sel:int, out_features:int, in_features:int):
  if n_sel == 4 and out_features == 3072 and in_features == 2048:
    return (Opt(OptOps.LOCAL, 1, 8), Opt(OptOps.GROUP, 1, 16))
  if n_sel == 4 and out_features == 2048 and in_features == 1536:
    return (Opt(OptOps.LOCAL, 1, 16), Opt(OptOps.GROUP, 1, 16))
  return ()

def custom_q4_0_mul_mat_id(out:UOp, x:UOp, scale:UOp, packed:UOp, sel:UOp) -> UOp:
  # out: (N, O), x: (N, I), scale: (E, O, I//32, 1), packed: (E, O, I//32, 16), sel: (N,)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(scale.shape) == 4 and len(packed.shape) == 4 and len(sel.shape) == 1
  assert all(isinstance(s, int) for s in out.shape+x.shape+scale.shape+packed.shape+sel.shape), "custom q4_0 kernel requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  bpr = I // 32
  E = scale.shape[0]
  assert x.shape == (N, I) and sel.shape[0] == N
  assert scale.shape == (E, O, bpr, 1) and packed.shape == (E, O, bpr, 16)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  # Process two packed bytes (4 q4 values) per step, mirroring dense Q4 path.
  # This reduces loop trip count and increases per-iteration arithmetic intensity.
  r = UOp.range(bpr*8, 2, axis_type=AxisType.REDUCE)
  br, jb = r//8, (r%8)*2
  # Clamp dynamic expert id once; avoids expensive wraparound index math in the reduction loop.
  e = sel[n].cast(dtypes.int).maximum(0).minimum(E-1).cast(dtypes.index)
  base = ((e * O) + o) * bpr + br
  scale_flat = scale.reshape(E * O * bpr)
  packed_flat = packed.reshape(E * O * bpr * 16)

  s = scale_flat[base].cast(dtypes.float)
  q0, q1 = packed_flat[base * 16 + jb], packed_flat[base * 16 + jb + 1]
  q0_lo = (q0 & 0xF).cast(dtypes.float) - 8.0
  q0_hi = (q0 >> 4).cast(dtypes.float) - 8.0
  q1_lo = (q1 & 0xF).cast(dtypes.float) - 8.0
  q1_hi = (q1 >> 4).cast(dtypes.float) - 8.0
  x0 = x[n, br*32 + jb].cast(dtypes.float)
  x1 = x[n, br*32 + jb + 1].cast(dtypes.float)
  x2 = x[n, br*32 + jb + 16].cast(dtypes.float)
  x3 = x[n, br*32 + jb + 17].cast(dtypes.float)
  # Explicit REDUCE form keeps grouped-reduction legalization valid for expert kernels too.
  dot = s * (q0_lo * x0 + q1_lo * x1 + q0_hi * x2 + q1_hi * x3)
  acc = dot.reduce(r, arg=Ops.ADD, dtype=dtypes.float)
  return out[n, o].store(acc.cast(out.dtype.base)).end(n, o).sink(
    arg=KernelInfo(name=f"custom_q4_0_mul_mat_id_{N}_{O}_{I}", opts_to_apply=_q4_0_mul_mat_id_opts(N, O, I)))

def custom_q4_0_mul_mat_id_split(partial:UOp, x:UOp, scale:UOp, packed:UOp, sel:UOp) -> UOp:
  """
  Two-stage Q4 MoE reduction:
    stage 1: chunked partials over block rows
    stage 2: Tensor sum over chunk axis

  This gives the DSL path a structural parallel-reduction analogue to MSL's lane-parallel kernel.
  """
  assert len(partial.shape) == 3 and len(x.shape) == 2 and len(scale.shape) == 4 and len(packed.shape) == 4 and len(sel.shape) == 1
  assert all(isinstance(s, int) for s in partial.shape+x.shape+scale.shape+packed.shape+sel.shape), "custom q4_0 split kernel requires static shapes"
  N, O, G = partial.shape
  I = x.shape[1]
  bpr = I // 32
  E = scale.shape[0]
  assert bpr % G == 0, f"bpr {bpr} must be divisible by chunk groups {G}"
  chunk_bpr = bpr // G
  assert x.shape == (N, I) and sel.shape[0] == N
  assert scale.shape == (E, O, bpr, 1) and packed.shape == (E, O, bpr, 16)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  g = UOp.range(G, 2)
  # Same 2-byte reduction step as custom_q4_0_mul_mat_id, scoped to one chunk.
  r = UOp.range(chunk_bpr*8, 3, axis_type=AxisType.REDUCE)
  br = g*chunk_bpr + (r//8)
  jb = (r % 8) * 2

  e = sel[n].cast(dtypes.int).maximum(0).minimum(E-1).cast(dtypes.index)
  base = ((e * O) + o) * bpr + br
  scale_flat = scale.reshape(E * O * bpr)
  packed_flat = packed.reshape(E * O * bpr * 16)

  s = scale_flat[base].cast(dtypes.float)
  q0, q1 = packed_flat[base * 16 + jb], packed_flat[base * 16 + jb + 1]
  q0_lo = (q0 & 0xF).cast(dtypes.float) - 8.0
  q0_hi = (q0 >> 4).cast(dtypes.float) - 8.0
  q1_lo = (q1 & 0xF).cast(dtypes.float) - 8.0
  q1_hi = (q1 >> 4).cast(dtypes.float) - 8.0
  x0 = x[n, br*32 + jb].cast(dtypes.float)
  x1 = x[n, br*32 + jb + 1].cast(dtypes.float)
  x2 = x[n, br*32 + jb + 16].cast(dtypes.float)
  x3 = x[n, br*32 + jb + 17].cast(dtypes.float)
  dot = s * (q0_lo * x0 + q1_lo * x1 + q0_hi * x2 + q1_hi * x3)
  acc = dot.reduce(r, arg=Ops.ADD, dtype=dtypes.float)
  return partial[n, o, g].store(acc.cast(partial.dtype.base)).end(n, o, g).sink(
    arg=KernelInfo(name=f"custom_q4_0_mul_mat_id_split_{N}_{O}_{I}_g{G}",
                   opts_to_apply=_q4_0_mul_mat_id_split_opts(N, O, I, G)))

def custom_fp16_mul_mat_id(out:UOp, x:UOp, weights:UOp, sel:UOp) -> UOp:
  # out: (N, O), x: (N, I), weights: (E, O, I), sel: (N,)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(weights.shape) == 3 and len(sel.shape) == 1
  assert all(isinstance(s, int) for s in out.shape+x.shape+weights.shape+sel.shape), "custom fp16 mul_mat_id requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  E = weights.shape[0]
  assert x.shape == (N, I) and sel.shape[0] == N and weights.shape == (E, O, I)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  r = UOp.range(I, 2, axis_type=AxisType.REDUCE)
  e = sel[n].cast(dtypes.int).maximum(0).minimum(E-1).cast(dtypes.index)

  w_flat = weights.reshape(E * O * I)
  w = w_flat[((e * O) + o) * I + r].cast(dtypes.float)
  xv = x[n, r].cast(dtypes.float)
  acc = (w * xv).reduce(r, arg=Ops.ADD, dtype=dtypes.float)
  return out[n, o].store(acc.cast(out.dtype.base)).end(n, o).sink(
    arg=KernelInfo(name=f"custom_fp16_mul_mat_id_{N}_{O}_{I}", opts_to_apply=_fp16_mul_mat_id_opts(N, O, I)))

def _fp16_linear_opts(n_rows:int, out_features:int, in_features:int):
  # Keep fp16 custom-linear opts conservative until grouped schedule legality is tuned.
  return ()

def custom_fp16_linear(out:UOp, x:UOp, w:UOp) -> UOp:
  # out: (N, O), x: (N, I), w: (O, I)
  assert len(out.shape) == 2 and len(x.shape) == 2 and len(w.shape) == 2
  assert all(isinstance(s, int) for s in out.shape+x.shape+w.shape), "custom fp16 linear requires static shapes"
  N, O = out.shape
  I = x.shape[1]
  assert x.shape == (N, I) and w.shape == (O, I)

  n = UOp.range(N, 0)
  o = UOp.range(O, 1)
  r = UOp.range(I, 2, axis_type=AxisType.REDUCE)

  xv = x[n, r].cast(dtypes.float)
  wv = w[o, r].cast(dtypes.float)
  acc = (xv * wv).reduce(r, arg=Ops.ADD, dtype=dtypes.float)
  return out[n, o].store(acc.cast(out.dtype.base)).end(n, o).sink(
    arg=KernelInfo(name=f"custom_fp16_linear_{N}_{O}_{I}", opts_to_apply=_fp16_linear_opts(N, O, I)))

class QuantizedLinear:
  __slots__ = ('blocks', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_dequant_fn', '_dequant_cache', '_q4_0_blocks',
               '_q4_0_scale', '_q4_0_packed', '_blocks_cache')
  def __init__(self, blocks:Tensor, shape:tuple[int, int], ggml_type:int):
    self.blocks = blocks
    self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, _, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self._dequant_cache = None
    self._q4_0_blocks = None
    self._q4_0_scale = None
    self._q4_0_packed = None
    self._blocks_cache = None

  def _ensure_dequant_cache(self, device: str|tuple[str, ...]) -> None:
    if self._dequant_cache is not None and self._dequant_cache.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    w = self._dequant_fn(blocks).reshape(self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast('float16')
    self._dequant_cache = w.realize()

  def _ensure_q4_0_blocks(self, device: str|tuple[str, ...]) -> None:
    if self._q4_0_blocks is not None and self._q4_0_blocks.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    bpr = self.in_features // 32
    self._q4_0_blocks = blocks.reshape(self.out_features, bpr, 18)

  def _ensure_blocks_cache(self, device: str|tuple[str, ...]) -> None:
    if self._blocks_cache is not None and self._blocks_cache.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    self._blocks_cache = blocks.contiguous().realize()

  def _ensure_q4_0_separated(self, device: str|tuple[str, ...]) -> None:
    """Pre-separate Q4_0 scale and packed data for better memory coalescing."""
    if self._q4_0_scale is not None and self._q4_0_scale.device == device: return
    self._ensure_q4_0_blocks(device)
    self._q4_0_scale = self._q4_0_blocks[:, :, :2].bitcast(dtypes.float16).contiguous().realize()  # (O, bpr, 1)
    self._q4_0_packed = self._q4_0_blocks[:, :, 2:].contiguous().realize()                         # (O, bpr, 16)

  def __call__(self, x:Tensor) -> Tensor:
    use_cache = getenv("QL_CACHE_ALL", 0) == 1
    if not use_cache:
      use_cache = (
        (getenv("QL_CACHE_ATTN_Q_A", 0) == 1 and self.out_features == 1536 and self.in_features == 2048) or
        (getenv("QL_CACHE_ATTN_KV_A", 0) == 1 and self.out_features == 576 and self.in_features == 2048) or
        (getenv("QL_CACHE_ATTN_Q_B", 0) == 1 and self.out_features == 3072 and self.in_features == 1536) or
        (getenv("QL_CACHE_ATTN_OUT", 0) == 1 and self.out_features == 2048 and self.in_features == 2048) or
        (getenv("QL_CACHE_FFN_IN", 0) == 1 and self.out_features == 6144 and self.in_features == 2048) or
        (getenv("QL_CACHE_FFN_DOWN", 0) == 1 and self.out_features == 2048 and self.in_features == 6144)
      )
    # Q4_0 packed-dot: read quantized blocks directly (3.56x less bandwidth than fp16 cache)
    if self.ggml_type == 2 and self.in_features % 32 == 0 and not use_cache:
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_flat = x_fp16.reshape(-1, self.in_features)
      out = Tensor.empty(x_flat.shape[0], self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
      use_q4_linear_msl = getenv("QL_MOE_MSL", 0) == 1 and isinstance(x.device, str) and x.device.startswith("METAL")
      if use_q4_linear_msl:
        from tinygrad.apps.q4_linear_msl import custom_q4_0_linear_msl
        self._ensure_q4_0_separated(x.device)
        out = Tensor.custom_kernel(out, x_flat, self._q4_0_scale, self._q4_0_packed, fxn=custom_q4_0_linear_msl)[0]
        return out.reshape(*x.shape[:-1], self.out_features)
      self._ensure_q4_0_separated(x.device)
      groups = _q4_0_linear_split_groups(x_flat.shape[0], self.out_features, self.in_features)
      if groups > 1 and (self.in_features // 32) % groups == 0:
        # Split-K stage (parallel partial sums) + final reduction.
        partial = Tensor.empty(x_flat.shape[0], self.out_features, groups, dtype=x_fp16.dtype, device=x_fp16.device)
        partial = Tensor.custom_kernel(partial, x_flat, self._q4_0_scale, self._q4_0_packed, fxn=custom_q4_0_linear_split)[0]
        out = partial.sum(axis=2)
      else:
        out = Tensor.custom_kernel(out, x_flat, self._q4_0_scale, self._q4_0_packed, fxn=custom_q4_0_linear)[0]
      return out.reshape(*x.shape[:-1], self.out_features)
    use_shexp_msl = getenv("QL_SHEXP_MSL", 0) == 1 and isinstance(x.device, str) and x.device.startswith("METAL")
    if use_shexp_msl and self.ggml_type == 13 and self.out_features == 3072 and self.in_features == 2048:
      from tinygrad.apps.qk_linear_msl import custom_q5_k_linear_msl
      self._ensure_blocks_cache(x.device)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_flat = x_fp16.reshape(-1, self.in_features)
      out = Tensor.empty(x_flat.shape[0], self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
      out = Tensor.custom_kernel(out, x_flat, self._blocks_cache, fxn=custom_q5_k_linear_msl)[0]
      return out.reshape(*x.shape[:-1], self.out_features)
    if use_shexp_msl and self.ggml_type == 14 and self.out_features == 2048 and self.in_features == 1536:
      from tinygrad.apps.qk_linear_msl import custom_q6_k_linear_msl
      self._ensure_blocks_cache(x.device)
      x_fp16 = x.cast(dtypes.float16) if x.dtype != dtypes.float16 else x
      x_flat = x_fp16.reshape(-1, self.in_features)
      out = Tensor.empty(x_flat.shape[0], self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
      out = Tensor.custom_kernel(out, x_flat, self._blocks_cache, fxn=custom_q6_k_linear_msl)[0]
      return out.reshape(*x.shape[:-1], self.out_features)
    self._ensure_dequant_cache(x.device)
    return x.linear(self._dequant_cache.T, None)

class QuantizedExpertWeights:
  __slots__ = ('blocks', 'num_experts', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_bytes_per_block', '_dequant_fn',
               'expert_first_in_memory', '_blocks_per_expert', '_expert_blocks', '_q4_0_scale', '_q4_0_packed', '_expert_dequant_cache')
  def __init__(self, blocks:Tensor, shape:tuple[int, int, int], ggml_type:int, expert_first_in_memory:bool=True):
    self.num_experts, self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, self._bytes_per_block, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self.blocks = blocks
    self.expert_first_in_memory = expert_first_in_memory
    self._blocks_per_expert = self.blocks.shape[0] // self.num_experts
    self._expert_blocks = None
    self._q4_0_scale = None
    self._q4_0_packed = None
    self._expert_dequant_cache = None
    assert self.blocks.shape[0] % self.num_experts == 0, f"blocks {self.blocks.shape[0]} not divisible by num_experts {self.num_experts}"

  def _ensure_expert_blocks(self, device: str|tuple[str, ...]) -> None:
    """Reshape blocks to (num_experts, blocks_per_expert, bytes_per_block) - just a view, no copy."""
    if self._expert_blocks is not None and self._expert_blocks.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    if self.expert_first_in_memory:
      self._expert_blocks = blocks.reshape(self.num_experts, self._blocks_per_expert, self._bytes_per_block)
    else:
      blocks_per_row = (self.in_features + self._el_per_block - 1) // self._el_per_block
      reshaped = blocks.reshape(self.out_features, blocks_per_row, self.num_experts, self._bytes_per_block)
      self._expert_blocks = reshaped.permute(2, 0, 1, 3).contiguous().reshape(self.num_experts, self._blocks_per_expert, self._bytes_per_block)

  def _ensure_q4_0_separated(self, device: str|tuple[str, ...]) -> None:
    """Pre-separate Q4_0 scale and packed data for better memory coalescing in matmul kernels."""
    if self._q4_0_scale is not None and self._q4_0_scale.device == device: return
    self._ensure_expert_blocks(device)
    O, bpr = self.out_features, self.in_features // 32
    eb_4d = self._expert_blocks.reshape(self.num_experts, O, bpr, 18)
    self._q4_0_scale = eb_4d[:, :, :, :2].bitcast(dtypes.float16).contiguous().realize()   # (E, O, bpr, 1)
    self._q4_0_packed = eb_4d[:, :, :, 2:].contiguous().realize()                          # (E, O, bpr, 16)

  def _ensure_dequant_expert_cache(self, device: str|tuple[str, ...]) -> None:
    if self._expert_dequant_cache is not None and self._expert_dequant_cache.device == device: return
    self._ensure_expert_blocks(device)
    w = self._dequant_fn(self._expert_blocks.reshape(-1, self._bytes_per_block))
    w = w.reshape(self.num_experts, self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast(dtypes.float16)
    self._expert_dequant_cache = w.contiguous().realize()

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    B, T, K = sel.shape
    n_sel = B * T * K
    if len(x.shape) == 3: x = x.reshape(B, T, 1, x.shape[-1])
    xk = x if x.shape[2] == K else x.expand(B, T, K, x.shape[-1])
    x_flat = xk.reshape(n_sel, self.in_features)

    # Q4_0 expert path via mul_mat_id-style primitive call.
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      self._ensure_q4_0_separated(x.device)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      sel_flat = sel.reshape(-1)
      if getenv("QL_MOE_MSL", 0) == 1 and isinstance(x_fp16.device, str) and x_fp16.device.startswith("METAL"):
        out = Tensor.empty(n_sel, self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
        from tinygrad.apps.q4_moe_msl import custom_q4_0_mul_mat_id_msl
        out = Tensor.mul_mat_id(out, x_fp16, self._q4_0_scale, self._q4_0_packed, sel_flat, fxn=custom_q4_0_mul_mat_id_msl)
      else:
        groups = _q4_0_mul_mat_id_split_groups(n_sel, self.out_features, self.in_features)
        if groups > 1 and (self.in_features // 32) % groups == 0:
          partial = Tensor.empty(n_sel, self.out_features, groups, dtype=x_fp16.dtype, device=x_fp16.device)
          partial = Tensor.mul_mat_id(partial, x_fp16, self._q4_0_scale, self._q4_0_packed, sel_flat, fxn=custom_q4_0_mul_mat_id_split)
          out = partial.sum(axis=2)
        else:
          out = Tensor.empty(n_sel, self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
          out = Tensor.mul_mat_id(out, x_fp16, self._q4_0_scale, self._q4_0_packed, sel_flat, fxn=custom_q4_0_mul_mat_id)
      return out.reshape(B, T, K, self.out_features)

    # Q5_K/Q6_K expert path: keep primitive contract (expert-id indexed matmul) with dequantized expert cache.
    if self.ggml_type in (13, 14):
      self._ensure_dequant_expert_cache(x.device)
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      sel_flat = sel.reshape(-1)
      out = Tensor.empty(n_sel, self.out_features, dtype=x_fp16.dtype, device=x_fp16.device)
      out = Tensor.mul_mat_id(out, x_fp16, self._expert_dequant_cache, sel_flat, fxn=custom_fp16_mul_mat_id)
      return out.reshape(B, T, K, self.out_features)

    # Fallback: gather combined blocks then dequant
    self._ensure_expert_blocks(x.device)
    sel_blocks = self._expert_blocks[sel.reshape(-1)]  # (n_sel, bpe, bpb)
    w = self._dequant_fn(sel_blocks.reshape(-1, self._bytes_per_block)).reshape(n_sel, self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast(dtypes.float16)
    return (x_flat.reshape(n_sel, 1, self.in_features) @ w.transpose(-1, -2)).reshape(B, T, K, self.out_features)

def replace_quantized_modules(model, quantized_tensors: dict, state_dict: dict) -> tuple[int, int, int]:
  """Replace Linear/ExpertWeights modules with quantized versions. Returns (linear_replaced, expert_replaced, dequantized)."""
  q_replaced_linear, q_replaced_expert, q_dequant = 0, 0, 0
  for name in list(quantized_tensors.keys()):
    tensor_data = quantized_tensors[name]
    if len(tensor_data) == 4:
      blocks, shape, ggml_type, expert_first_in_memory = tensor_data
    else:
      blocks, shape, ggml_type = tensor_data
      expert_first_in_memory = True
    if not name.endswith('.weight') or 'attn_kv_b' in name: continue
    parts = name[:-7].split('.')
    obj, parent, attr, found = model, None, None, False
    for i, part in enumerate(parts):
      if part == 'blk': continue
      if parts[i-1] == 'blk' and part.isdigit(): obj = obj.blk[int(part)]
      elif hasattr(obj, part): parent, attr, obj = obj, part, getattr(obj, part)
      else: break
    else: found = True
    if found:
      if isinstance(obj, nn.Linear):
        del quantized_tensors[name]
        setattr(parent, attr, QuantizedLinear(blocks, shape, ggml_type))
        q_replaced_linear += 1
      elif type(obj).__name__ == 'ExpertWeights':
        del quantized_tensors[name]
        setattr(parent, attr, QuantizedExpertWeights(blocks, shape, ggml_type, expert_first_in_memory))
        q_replaced_expert += 1
      else: found = False
    if not found:
      del quantized_tensors[name]
      dequant_fn = GGML_QUANT_INFO[ggml_type][2]
      state_dict[name] = dequant_fn(blocks).reshape(*shape)
      state_dict[name] = state_dict[name].cast('float16') if getenv("HALF", 1) else state_dict[name]
      q_dequant += 1
  return q_replaced_linear, q_replaced_expert, q_dequant
