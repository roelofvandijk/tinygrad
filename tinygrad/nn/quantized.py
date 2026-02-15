from __future__ import annotations
from tinygrad import Tensor, getenv
from tinygrad.helpers import prod
from tinygrad.dtype import dtypes
from tinygrad.nn.state import ggml_data_to_tensor


class QuantizedExpertWeights:
  def __init__(self, blocks:Tensor, shape:tuple[int, int, int], ggml_type:int):
    self.num_experts, self.out_features, self.in_features, self.ggml_type = *shape, ggml_type
    bpb = blocks.shape[1]
    self.expert_blocks = blocks.reshape(self.num_experts, blocks.shape[0] // self.num_experts, bpb)

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    def _q4_0_dot(blocks: Tensor, x: Tensor) -> Tensor:
      scale, packed = blocks[..., :2].bitcast(dtypes.float16), blocks[..., 2:]
      x_pairs = x.cast(dtypes.float16).reshape(-1, 1, blocks.shape[-2], 2, 16)
      lo, hi = packed.bitwise_and(0xF).cast(dtypes.float16) - 8.0, packed.rshift(4).cast(dtypes.float16) - 8.0
      return (scale * (lo * x_pairs[:, :, :, 0, :] + hi * x_pairs[:, :, :, 1, :])).flatten(-2).sum(axis=-1)
    B, T, K, n_sel = sel.shape[0], sel.shape[1], sel.shape[2], sel.shape[0] * sel.shape[1] * sel.shape[2]
    x = x.reshape(B, T, 1, x.shape[-1]) if len(x.shape) == 3 else x
    x_flat = (x if x.shape[2] == K else x.expand(B, T, K, x.shape[-1])).reshape(n_sel, self.in_features)
    sel_blocks = self.expert_blocks[sel.reshape(-1)]
    if self.ggml_type == 2 and self.in_features % 32 == 0:
      return _q4_0_dot(sel_blocks.reshape(n_sel, self.out_features, -1, 18), x_flat).reshape(B, T, K, self.out_features)
    w = ggml_data_to_tensor(sel_blocks.reshape(-1, sel_blocks.shape[-1]).flatten(), int(n_sel * self.out_features * self.in_features), self.ggml_type)
    w = w.reshape(n_sel, self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.cast('float16')
    return (x_flat.reshape(n_sel, 1, self.in_features) @ w.transpose(-1, -2)).reshape(B, T, K, self.out_features)

def replace_quantized_modules(model, quantized_tensors: dict, state_dict: dict):
  for name in list(quantized_tensors.keys()):
    blocks, shape, ggml_type = quantized_tensors.pop(name)
    if not name.endswith('.weight'): continue
    parts = name[:-7].split('.')  # Strip '.weight'
    parent = model
    for part in parts[:-1]:
      parent = parent[int(part)] if isinstance(parent, list) and part.isdigit() else getattr(parent, part, None)
      if parent is None: break
    if parent is not None:
      attr = parts[-1]
      obj = parent[int(attr)] if isinstance(parent, list) and attr.isdigit() else getattr(parent, attr, None)
      if obj is not None and type(obj).__name__ == 'ExpertWeights':
        setattr(parent, attr, QuantizedExpertWeights(blocks, shape, ggml_type))
        continue
    # Dequantize nn.Linear and others at load time
    w = ggml_data_to_tensor(blocks.flatten(), prod(shape), ggml_type).reshape(*shape)
    state_dict[name] = w.cast('float16') if getenv("HALF", 1) else w
