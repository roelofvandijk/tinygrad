from __future__ import annotations
import math, functools
from tinygrad import Tensor, nn, UOp
from tinygrad.dtype import dtypes

@functools.cache
def _precompute_freqs(dim: int, end: int, theta: float) -> Tensor:
  freqs = Tensor.arange(end).float().unsqueeze(1) * (1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))).unsqueeze(0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).cast(dtypes.float16).contiguous()

def _apply_rope_interleaved(x: Tensor, freqs_cis: Tensor) -> Tensor:
  cos, sin = freqs_cis[..., 0].reshape(1, 1, x.shape[2], -1), freqs_cis[..., 1].reshape(1, 1, x.shape[2], -1)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  return (x1 * cos - x2 * sin).unsqueeze(-1).cat((x2 * cos + x1 * sin).unsqueeze(-1), dim=-1).flatten(-2)

# *** MLA architecture ***

class MLATransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, norm_eps:float, max_context:int,
               q_lora_rank:int=0, kv_lora_rank:int=0, qk_nope_head_dim:int=0, qk_rope_head_dim:int=0, v_head_dim:int=0,
               num_experts:int=0, num_experts_per_tok:int=0, n_shared_experts:int=0, moe_hidden_dim:int=0,
               expert_weights_norm:bool=False, expert_weights_scale:float=1.0,
               rope_theta:float=10000.0, **_):
    self.n_heads, self.qk_nope_head_dim, self.qk_rope_head_dim = n_heads, qk_nope_head_dim, qk_rope_head_dim
    self.q_head_dim, self.kv_lora_rank, self.max_context = qk_nope_head_dim + qk_rope_head_dim, kv_lora_rank, max_context
    self._attn_scale = 1.0 / math.sqrt(self.q_head_dim)
    self._rope_theta = rope_theta
    if q_lora_rank > 0:
      self.attn_q_a = nn.Linear(dim, q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(q_lora_rank, norm_eps)
      self.attn_q_b = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)
    else: self.attn_q = nn.Linear(dim, n_heads * self.q_head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(kv_lora_rank, norm_eps)
    self.v_head_dim = v_head_dim
    class Weight:
      def __init__(self, *shape): self.weight = Tensor.empty(*shape)
    self.attn_k_b = Weight(n_heads, kv_lora_rank, qk_nope_head_dim)
    self.attn_v_b = Weight(n_heads, v_head_dim, kv_lora_rank)
    self.attn_output = nn.Linear(n_heads * v_head_dim, dim, bias=False)
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    if num_experts > 0:
      from tinygrad.apps.llm import ExpertWeights
      self.num_experts_per_tok = num_experts_per_tok
      self.expert_weights_norm = expert_weights_norm
      self.expert_weights_scale = expert_weights_scale
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
      self.exp_probs_b = Tensor.zeros(num_experts)
      self.ffn_gate_exps = ExpertWeights(num_experts, dim, moe_hidden_dim)
      self.ffn_up_exps = ExpertWeights(num_experts, dim, moe_hidden_dim)
      self.ffn_down_exps = ExpertWeights(num_experts, moe_hidden_dim, dim)
      if n_shared_experts > 0:
        shexp_hidden = n_shared_experts * moe_hidden_dim
        self.ffn_gate_shexp = nn.Linear(dim, shexp_hidden, bias=False)
        self.ffn_up_shexp = nn.Linear(dim, shexp_hidden, bias=False)
        self.ffn_down_shexp = nn.Linear(shexp_hidden, dim, bias=False)
    else:
      self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm, B, T = self.attn_norm(x), x.shape[0], x.shape[1]
    q = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x_norm))) if hasattr(self, 'attn_q_a') else self.attn_q(x_norm)
    kv_out = self.attn_kv_a_mqa(x_norm)
    q = q.reshape(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv, k_pe = kv_out.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
    freqs_cis = _precompute_freqs(self.qk_rope_head_dim, self.max_context, self._rope_theta)[start_pos:start_pos+T]
    q_pe, k_pe = _apply_rope_interleaved(q_pe, freqs_cis), _apply_rope_interleaved(k_pe, freqs_cis)
    q = (q_nope @ self.attn_k_b.weight.transpose(-1, -2)).cat(q_pe, dim=-1)
    kv_normed = self.attn_kv_a_norm(compressed_kv).unsqueeze(1)
    k_new = kv_normed.cat(k_pe, dim=-1)
    cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
    if not hasattr(self, "cache_k") or start_pos == 0:
      self.cache_k = Tensor.empty((B, 1, self.max_context, cache_dim), dtype=kv_normed.dtype, device=kv_normed.device).contiguous().realize()
    self.cache_k[:, :, start_pos:start_pos+T, :].assign(k_new).realize()
    k = self.cache_k[:, :, 0:start_pos+T, :]
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    qk = q.matmul(k.transpose(-2, -1)) * self._attn_scale
    if mask is not None: qk = qk + mask
    attn = (qk.softmax(-1).matmul(k[:, :, :, :self.kv_lora_rank]) @ self.attn_v_b.weight.transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return x + self.attn_output(attn)

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      gate_scores = (h_norm.float() @ self.ffn_gate_inp.weight.float().T).sigmoid()
      _, sel = (gate_scores + self.exp_probs_b).topk(self.num_experts_per_tok)
      probs = gate_scores.gather(-1, sel)
      if self.expert_weights_norm: probs = probs / probs.sum(axis=-1, keepdim=True).maximum(6.103515625e-5)
      x = h_norm.unsqueeze(2)
      gated = self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x)
      weighted_gated = (gated * probs.unsqueeze(-1).cast(gated.dtype)).contiguous()
      expert_out = self.ffn_down_exps(sel, weighted_gated)
      moe = expert_out.sum(axis=2)
      out = moe * self.expert_weights_scale
      if hasattr(self, 'ffn_gate_shexp'):
        out = out.contiguous() + self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
      return h + out.cast(h.dtype)
    return h + self.ffn_down(self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm))

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

def load_mla_params_from_gguf(kv: dict, arch: str) -> dict:
  def ak(s, d=0): return kv.get(f'{arch}.{s}', d)
  qk_rope_head_dim = ak('rope.dimension_count')
  key_length = ak('attention.key_length_mla', ak('attention.key_length'))
  return dict(q_lora_rank=ak('attention.q_lora_rank'), kv_lora_rank=ak('attention.kv_lora_rank'), qk_rope_head_dim=qk_rope_head_dim,
    qk_nope_head_dim=key_length - qk_rope_head_dim if key_length > 0 else 0, v_head_dim=ak('attention.value_length_mla', ak('attention.value_length')),
    n_shared_experts=ak('expert_shared_count'), moe_hidden_dim=ak('expert_feed_forward_length'), leading_dense_blocks=ak('leading_dense_block_count'),
    expert_weights_norm=ak('expert_weights_norm', False),
    expert_weights_scale=ak('expert_weights_scale', 1.0))

