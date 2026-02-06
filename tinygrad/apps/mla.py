from __future__ import annotations
import math
from tinygrad import Tensor, nn, UOp
from tinygrad.dtype import dtypes
from tinygrad.apps.rope import apply_rope_interleaved, YarnParams

def _topk_pairwise(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
  """O(n^2) pairwise comparison topk. 3 kernels vs 29 for bitonic sort. Fine for small n (e.g. 64 experts)."""
  n = scores.shape[-1]
  s_col = scores.unsqueeze(-1)  # (..., n, 1)
  s_row = scores.unsqueeze(-2)  # (..., 1, n)
  # rank[i] = # of j where (scores[j] > scores[i]) or (scores[j]==scores[i] and j<i)
  gt = (s_row > s_col)
  eq = (s_row == s_col)
  j_idx = Tensor.arange(n).reshape(1, 1, n)
  i_idx = Tensor.arange(n).reshape(1, n, 1)
  ranks = (gt | (eq & (j_idx < i_idx))).sum(-1)  # (..., n), 0=largest
  target = Tensor.arange(k).reshape(1, k)
  match = (ranks.unsqueeze(-1) == target.unsqueeze(-2)).float()  # (..., n, k)
  i_range = Tensor.arange(n, dtype=dtypes.float).reshape(1, n, 1)
  indices = (match * i_range).sum(-2).cast(dtypes.int)
  values = scores.gather(-1, indices)
  return values, indices

class PerHeadWeights:
  def __init__(self, n_heads:int, dim1:int, dim2:int):
    self.weight = Tensor.zeros(n_heads, dim1, dim2)

class MLATransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, norm_eps:float, rope_theta:float, max_context:int,
               q_lora_rank:int, kv_lora_rank:int, qk_nope_head_dim:int, qk_rope_head_dim:int, v_head_dim:int,
               num_experts:int=0, num_experts_per_tok:int=0, n_shared_experts:int=0, moe_hidden_dim:int=0,
               expert_gating_func:int=0, expert_weights_norm:bool=False, expert_weights_scale:float=1.0,
               mscale:float=1.0, yarn_scaling_factor:float=1.0, yarn_params:YarnParams|None=None):
    self.n_heads = n_heads
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.kv_lora_rank = kv_lora_rank
    self.q_lora_rank = q_lora_rank
    self.max_context = max_context
    self.mscale = mscale
    if q_lora_rank > 0:
      self.attn_q_a = nn.Linear(dim, q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(q_lora_rank, norm_eps)
      self.attn_q_b = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(dim, n_heads * self.q_head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(kv_lora_rank, norm_eps)
    self.attn_k_b = PerHeadWeights(n_heads, kv_lora_rank, qk_nope_head_dim)
    self.attn_v_b = PerHeadWeights(n_heads, v_head_dim, kv_lora_rank)
    self.attn_output = nn.Linear(n_heads * v_head_dim, dim, bias=False)
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    from tinygrad.apps.llm import ExpertWeights
    if num_experts > 0:
      self.num_experts_per_tok = num_experts_per_tok
      self.expert_gating_func = expert_gating_func
      self.expert_weights_norm = expert_weights_norm
      self.expert_weights_scale = expert_weights_scale
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
      class ExpProbsBias:
        def __init__(self): self.bias = Tensor.zeros(num_experts)
      self.exp_probs_b = ExpProbsBias()
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
    x_norm = self.attn_norm(x)
    B, T, _ = x.shape
    if self.q_lora_rank > 0:
      q = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x_norm)))
    else:
      q = self.attn_q(x_norm)
    q = q.reshape(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv = self.attn_kv_a_mqa(x_norm)
    compressed_kv, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
    freqs_cis = self.freqs_cis_cache[start_pos:start_pos+T]
    q_pe = apply_rope_interleaved(q_pe, freqs_cis)
    k_pe = apply_rope_interleaved(k_pe, freqs_cis)
    kv_normed = self.attn_kv_a_norm(compressed_kv)
    # Absorbed attention: absorb K projection into Q  (weight is h,kv_lora,qk_nope — contract over qk_nope)
    q_absorbed = Tensor.einsum("bhtd,hnd->bhtn", q_nope, self.attn_k_b.weight)
    q = q_absorbed.cat(q_pe, dim=-1)
    kv_normed_4d = kv_normed.reshape(B, T, 1, self.kv_lora_rank).transpose(1, 2)
    k = kv_normed_4d.cat(k_pe, dim=-1)
    cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
    if not hasattr(self, "cache_k") or start_pos == 0:
      self.cache_k = Tensor.empty((B, 1, self.max_context, cache_dim), dtype=k.dtype, device=k.device).contiguous().realize()
      self.cache_v = Tensor.empty((B, 1, self.max_context, self.kv_lora_rank), dtype=kv_normed.dtype, device=kv_normed.device).contiguous().realize()
    self.cache_k[:, :, start_pos:start_pos+T, :].assign(k).realize()
    self.cache_v[:, :, start_pos:start_pos+T, :].assign(kv_normed_4d).realize()
    k = self.cache_k[:, :, 0:start_pos+T, :]
    v = self.cache_v[:, :, 0:start_pos+T, :]
    scale = self.mscale * self.mscale / math.sqrt(self.q_head_dim)
    qk = q.matmul(k.transpose(-2, -1)) * scale
    if T > 1:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1)
      qk = qk + mask
    attn_weights = qk.float().softmax(-1).cast(qk.dtype)
    attn = attn_weights.matmul(v)
    # Absorbed attention: absorb V projection
    attn = Tensor.einsum("bhtn,hvn->bhtv", attn, self.attn_v_b.weight)
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    return x + self.attn_output(attn)

  def _feed_forward(self, h: Tensor, start_pos: int|UOp = 0) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      router_logits = h_norm.float() @ self.ffn_gate_inp.weight.float().T
      if self.expert_gating_func == 1: gate_scores = router_logits.softmax(-1)
      elif self.expert_gating_func == 2: gate_scores = router_logits.sigmoid()
      elif self.expert_gating_func == 3: gate_scores = router_logits
      else: gate_scores = router_logits.softmax(-1)
      selection_scores = gate_scores + self.exp_probs_b.bias if hasattr(self, 'exp_probs_b') else gate_scores
      _, sel = _topk_pairwise(selection_scores, self.num_experts_per_tok)
      probs = gate_scores.gather(-1, sel)
      if self.expert_gating_func == 3: probs = probs.softmax(-1)
      elif self.expert_weights_norm: probs = probs / probs.sum(axis=-1, keepdim=True).maximum(6.103515625e-5)
      gated = self.ffn_gate_exps(sel, h_norm).float().silu() * self.ffn_up_exps(sel, h_norm).float()
      expert_out = self.ffn_down_exps(sel, gated).float()
      # Break fusion with contiguous before weighted sum
      expert_out = expert_out.contiguous()
      out = (expert_out * probs.unsqueeze(-1)).sum(axis=2) * self.expert_weights_scale
      if hasattr(self, 'ffn_gate_shexp'):
        out = out + self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).float().silu() * self.ffn_up_shexp(h_norm).float()).float()
      return h + out.cast(h.dtype)
    h_norm_f = h_norm.float()
    gated = self.ffn_gate(h_norm_f).silu().contiguous() * self.ffn_up(h_norm_f)
    return h + self.ffn_down(gated).float().cast(h.dtype)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos), start_pos).contiguous()

def load_mla_params_from_gguf(kv: dict, arch: str) -> dict:
  """Extract MLA architecture params from GGUF metadata. Returns dict of MLA params."""
  qk_rope_head_dim = kv.get(f'{arch}.rope.dimension_count', 0)
  key_length = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', 0))
  return {
    'q_lora_rank': kv.get(f'{arch}.attention.q_lora_rank', 0),
    'kv_lora_rank': kv.get(f'{arch}.attention.kv_lora_rank', 0),
    'qk_rope_head_dim': qk_rope_head_dim,
    'qk_nope_head_dim': key_length - qk_rope_head_dim if key_length > 0 else 0,
    'v_head_dim': kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', 0)),
    'n_shared_experts': kv.get(f'{arch}.expert_shared_count', 0),
    'moe_hidden_dim': kv.get(f'{arch}.expert_feed_forward_length', 0),
    'leading_dense_blocks': kv.get(f'{arch}.leading_dense_block_count', 0),
    'expert_gating_func': kv.get(f'{arch}.expert_gating_func', 0) or (2 if arch == 'glm4' else 1),
    'expert_weights_norm': kv.get(f'{arch}.expert_weights_norm', False),
    'expert_weights_scale': kv.get(f'{arch}.expert_weights_scale', 1.0),
  }

def split_kv_b(kv_b: Tensor, n_heads: int, qk_nope_head_dim: int, v_head_dim: int, kv_lora_rank: int) -> tuple[Tensor, Tensor]:
  """Split combined attn_kv_b into separate attn_k_b and attn_v_b weights."""
  kv_b_reshaped = kv_b.reshape(n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
  k_b, v_b = kv_b_reshaped.split([qk_nope_head_dim, v_head_dim], dim=1)
  return k_b.transpose(1, 2), v_b  # k_b: (n_heads, kv_lora_rank, qk_nope), v_b: (n_heads, v_head, kv_lora_rank)
