from __future__ import annotations
from tinygrad import Tensor, nn, getenv
from tinygrad.dtype import dtypes
from tinygrad.nn.state import GGML_QUANT_INFO
from tinygrad.nn.metal_q4k import q4k_linear_msl

class QuantizedLinear:
  __slots__ = ('blocks', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_dequant_fn', '_q4k_blocks', '_dequant_cache')
  def __init__(self, blocks:Tensor, shape:tuple[int, int], ggml_type:int):
    self.blocks = blocks
    self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, _, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self._q4k_blocks = None
    self._dequant_cache = None

  def _ensure_q4k_blocks(self, device: str|tuple[str, ...]) -> None:
    if self._q4k_blocks is not None and self._q4k_blocks.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    self._q4k_blocks = blocks.cast(dtypes.uint8).contiguous() if blocks.dtype != dtypes.uint8 else blocks.contiguous()

  def _ensure_dequant_cache(self, device: str|tuple[str, ...]) -> None:
    if self._dequant_cache is not None and self._dequant_cache.device == device: return
    blocks = self.blocks.to(device) if self.blocks.device != device else self.blocks
    w = self._dequant_fn(blocks).reshape(self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast('float16')
    self._dequant_cache = w.realize()

  def __call__(self, x:Tensor) -> Tensor:
    dev = x.device[0] if isinstance(x.device, tuple) else x.device
    is_metal = isinstance(dev, str) and dev.startswith("METAL")

    if self.ggml_type == 12:
      blocks_per_row = self.in_features // 256
      if self.in_features % 256 != 0 or self.blocks.shape[0] != self.out_features * blocks_per_row:
        raise RuntimeError(f"Q4_K fused path requires blocks_per_row={blocks_per_row}, got blocks.shape[0]={self.blocks.shape[0]}")
      if not is_metal:
        raise RuntimeError("Q4_K fused path requires METAL device")
      self._ensure_q4k_blocks(x.device)
      x_flat = x.reshape(-1, self.in_features).cast(dtypes.float16).contiguous()
      out = q4k_linear_msl(x_flat, self._q4k_blocks, self.out_features, self.in_features)
      return out[:, :self.out_features].reshape(*x.shape[:-1], self.out_features)

    # Non-Q4K: dequantize and cache
    blocks_per_row = self.in_features // self._el_per_block
    if self.in_features % self._el_per_block != 0 or self.blocks.shape[0] != self.out_features * blocks_per_row:
      w = self._dequant_fn(self.blocks).reshape(self.out_features, self.in_features)
      return x.linear((w.cast('float16') if getenv("HALF", 1) else w).T, None)
    self._ensure_dequant_cache(x.device)
    return x.linear(self._dequant_cache.T, None)

class QuantizedExpertWeights:
  __slots__ = ('blocks', 'num_experts', 'out_features', 'in_features', 'ggml_type', '_el_per_block', '_bytes_per_block', '_dequant_fn',
               'expert_first_in_memory', '_blocks_per_expert')
  def __init__(self, blocks:Tensor, shape:tuple[int, int, int], ggml_type:int, expert_first_in_memory:bool=True):
    self.num_experts, self.out_features, self.in_features = shape
    self.ggml_type = ggml_type
    self._el_per_block, self._bytes_per_block, self._dequant_fn = GGML_QUANT_INFO[ggml_type]
    self.blocks = blocks
    self.expert_first_in_memory = expert_first_in_memory
    self._blocks_per_expert = self.blocks.shape[0] // self.num_experts
    assert self.blocks.shape[0] % self.num_experts == 0, f"blocks {self.blocks.shape[0]} not divisible by num_experts {self.num_experts}"

  def _reshape_blocks(self) -> Tensor:
    if self.expert_first_in_memory:
      return self.blocks.reshape(self.num_experts, self._blocks_per_expert, self._bytes_per_block)
    blocks_per_row = (self.in_features + self._el_per_block - 1) // self._el_per_block
    reshaped = self.blocks.reshape(self.out_features, blocks_per_row, self.num_experts, self._bytes_per_block)
    return reshaped.permute(2, 0, 1, 3).reshape(self.num_experts, self._blocks_per_expert, self._bytes_per_block)

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    B, T, K = sel.shape
    n_sel = B * T * K
    if len(x.shape) == 3: x = x.reshape(B, T, 1, x.shape[-1])
    xk = x if x.shape[2] == K else x.expand(B, T, K, x.shape[-1])
    x_flat = xk.reshape(n_sel, self.in_features)

    selected_blocks = self._reshape_blocks()[sel.reshape(-1)]
    w_flat = self._dequant_fn(selected_blocks.reshape(n_sel * self._blocks_per_expert, self._bytes_per_block))
    # Q6K/Q5K: split dequant/matmul fusion for ~20% speedup; Q4K keeps fusion (higher compression)
    if self.ggml_type in (13, 14): w_flat = w_flat.half().contiguous()

    blocks_per_row = (self.in_features + self._el_per_block - 1) // self._el_per_block
    padded_row_len = blocks_per_row * self._el_per_block
    w = w_flat.reshape(n_sel, self.out_features, padded_row_len)
    if padded_row_len > self.in_features: w = w[:, :, :self.in_features]

    return (x_flat.float().reshape(n_sel, 1, self.in_features) @ w.float().transpose(-1, -2)).reshape(B, T, K, self.out_features)

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
