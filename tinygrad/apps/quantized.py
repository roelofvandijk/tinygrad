from __future__ import annotations
from tinygrad import Tensor, nn, getenv
from tinygrad.dtype import dtypes
from tinygrad.nn.state import GGML_QUANT_INFO

class QuantizedLinear:
  __slots__ = ('blocks', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_dequant_fn', '_dequant_cache')
  def __init__(self, blocks:Tensor, shape:tuple[int, int], ggml_type:int):
    self.blocks = blocks
    self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, _, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self._dequant_cache = None

  def _ensure_dequant_cache(self, device: str|tuple[str, ...]) -> None:
    if self._dequant_cache is not None and self._dequant_cache.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    w = self._dequant_fn(blocks).reshape(self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast('float16')
    self._dequant_cache = w.realize()

  def __call__(self, x:Tensor) -> Tensor:
    self._ensure_dequant_cache(x.device)
    return x.linear(self._dequant_cache.T, None)

class QuantizedExpertWeights:
  __slots__ = ('blocks', 'num_experts', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_bytes_per_block', '_dequant_fn',
               'expert_first_in_memory', '_blocks_per_expert', '_expert_blocks')
  def __init__(self, blocks:Tensor, shape:tuple[int, int, int], ggml_type:int, expert_first_in_memory:bool=True):
    self.num_experts, self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, self._bytes_per_block, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self.blocks = blocks
    self.expert_first_in_memory = expert_first_in_memory
    self._blocks_per_expert = self.blocks.shape[0] // self.num_experts
    self._expert_blocks = None
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

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    B, T, K = sel.shape
    n_sel = B * T * K
    if len(x.shape) == 3: x = x.reshape(B, T, 1, x.shape[-1])
    xk = x if x.shape[2] == K else x.expand(B, T, K, x.shape[-1])
    x_flat = xk.reshape(n_sel, self.in_features)

    self._ensure_expert_blocks(x.device)
    sel_blocks = self._expert_blocks[sel.reshape(-1)]  # (n_sel, bpe, bpb)

    # Q4_0 packed-dot: compute matmul directly on packed nibbles without expanding to full weight matrix
    # sum((nib - 8) * x) = sum(nib * x) - 8 * sum(x)  per block, then scale and reduce across blocks
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      O, IN = self.out_features, self.in_features
      bpr = IN // 32  # blocks per row
      blocks = sel_blocks.reshape(n_sel, O, bpr, 18)
      scale = blocks[:, :, :, :2].bitcast(dtypes.float16)  # (n_sel, O, bpr, 1)
      packed = blocks[:, :, :, 2:]  # (n_sel, O, bpr, 16) packed nibble bytes
      x_fp16 = x_flat.cast(dtypes.float16) if x_flat.dtype != dtypes.float16 else x_flat
      x_pairs = x_fp16.reshape(n_sel, 1, bpr, 2, 16)
      x_lo, x_hi = x_pairs[:, :, :, 0, :], x_pairs[:, :, :, 1, :]  # (n_sel, 1, bpr, 16) each
      lo = packed.bitwise_and(0xF).cast(dtypes.float16)  # (n_sel, O, bpr, 16)
      hi = packed.rshift(4).cast(dtypes.float16)  # (n_sel, O, bpr, 16)
      nib_dot = (lo * x_lo + hi * x_hi).sum(axis=-1)  # (n_sel, O, bpr)
      x_block_sum = x_fp16.reshape(n_sel, 1, bpr, 32).sum(axis=-1)  # (n_sel, 1, bpr)
      out = (scale.squeeze(-1) * (nib_dot - 8.0 * x_block_sum)).sum(axis=-1)  # (n_sel, O)
      return out.reshape(B, T, K, O)

    # Fallback: dequant to full weight matrix, then matmul
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
