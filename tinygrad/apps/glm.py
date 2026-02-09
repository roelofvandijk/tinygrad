from __future__ import annotations
import math
from tinygrad import Tensor, UOp, nn, getenv
from tinygrad.apps.rope import YarnParams, precompute_freqs_cis, precompute_freqs_cis_yarn, apply_rope_interleaved
from tinygrad.dtype import dtypes

def _next_pow2(x:int) -> int: return 1 if x <= 1 else 1 << (x - 1).bit_length()
def _bucket_len(cache_len:int, max_context:int, min_bucket:int) -> int:
  return min(max_context, max(min_bucket, _next_pow2(cache_len)))

def topk_pairwise(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
  """Pairwise top-k for small expert counts. Avoids the heavy bitonic path in decode."""
  n = scores.shape[-1]
  s_col, s_row = scores.unsqueeze(-1), scores.unsqueeze(-2)
  gt, eq = (s_row > s_col), (s_row == s_col)
  j_idx, i_idx = Tensor.arange(n).reshape(1, 1, n), Tensor.arange(n).reshape(1, n, 1)
  ranks = (gt | (eq & (j_idx < i_idx))).sum(-1)
  target = Tensor.arange(k).reshape(1, k)
  match = (ranks.unsqueeze(-1) == target.unsqueeze(-2)).float()
  i_range = Tensor.arange(n, dtype=dtypes.float).reshape(1, n, 1)
  indices = (match * i_range).sum(-2).cast(dtypes.int)
  return scores.gather(-1, indices), indices

class ExpertWeights:
  """Like nn.Linear but with an explicit expert axis."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B,T,K), x: (B,T,D) or (B,T,K,D) -> (B,T,K,O)
    if len(x.shape) == 3: x = x.unsqueeze(2)
    xk = x if x.shape[2] == sel.shape[2] else x.expand(x.shape[0], x.shape[1], sel.shape[2], x.shape[-1])
    return (xk.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

class PerHeadWeights:
  def __init__(self, n_heads:int, dim1:int, dim2:int):
    self.weight = Tensor.zeros(n_heads, dim1, dim2)

class GLMTransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, norm_eps:float, max_context:int, q_lora_rank:int, kv_lora_rank:int,
               qk_nope_head_dim:int, qk_rope_head_dim:int, v_head_dim:int, num_experts:int=0, num_experts_per_tok:int=0,
               n_shared_experts:int=0, moe_hidden_dim:int=0, expert_gating_func:int=2, expert_weights_norm:bool=False,
               expert_weights_scale:float=1.0, mscale:float=1.0):
    self.n_heads, self.max_context = n_heads, max_context
    self.q_lora_rank, self.kv_lora_rank = q_lora_rank, kv_lora_rank
    self.qk_nope_head_dim, self.qk_rope_head_dim = qk_nope_head_dim, qk_rope_head_dim
    self.q_head_dim, self.mscale = qk_nope_head_dim + qk_rope_head_dim, mscale
    self._attn_scale = mscale * mscale / math.sqrt(self.q_head_dim)
    self.split_moe_boundaries = bool(getenv("MLA_MOE_SPLIT_BOUNDARIES", 1))
    self.min_attn_bucket = getenv("GLM_ATTN_MIN_BUCKET", 0)

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
    self.attn_norm, self.ffn_norm = nn.RMSNorm(dim, norm_eps), nn.RMSNorm(dim, norm_eps)

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
        shared_hidden = n_shared_experts * moe_hidden_dim
        self.ffn_gate_shexp = nn.Linear(dim, shared_hidden, bias=False)
        self.ffn_up_shexp = nn.Linear(dim, shared_hidden, bias=False)
        self.ffn_down_shexp = nn.Linear(shared_hidden, dim, bias=False)
    else:
      self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, freqs_cis:Tensor) -> Tensor:
    x_norm = self.attn_norm(x)
    B, T, _ = x.shape

    if self.q_lora_rank > 0: q = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x_norm)))
    else: q = self.attn_q(x_norm)
    kv_out = self.attn_kv_a_mqa(x_norm)

    q = q.reshape(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv, k_pe = kv_out.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)

    q_pe, k_pe = apply_rope_interleaved(q_pe, freqs_cis), apply_rope_interleaved(k_pe, freqs_cis)
    q_nope = q_nope @ self.attn_k_b.weight.transpose(-1, -2)
    q = q_nope.cat(q_pe, dim=-1)

    kv_normed = self.attn_kv_a_norm(compressed_kv).unsqueeze(1)
    k_new = kv_normed.cat(k_pe, dim=-1)
    cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
    if not hasattr(self, "cache_k") or start_pos == 0:
      self.cache_k = Tensor.empty((B, 1, self.max_context, cache_dim), dtype=kv_normed.dtype, device=kv_normed.device).contiguous().realize()
    valid_len = start_pos + T
    self.cache_k[:, :, start_pos:valid_len, :].assign(k_new).realize()
    if self.min_attn_bucket > 0 and isinstance(start_pos, int):
      attn_len = _bucket_len(valid_len, self.max_context, self.min_attn_bucket)
      k = self.cache_k[:, :, :attn_len, :]
    else:
      attn_len = valid_len
      k = self.cache_k[:, :, :valid_len, :]

    qk = q.matmul(k.transpose(-2, -1)) * self._attn_scale
    if isinstance(attn_len, int) and isinstance(valid_len, int) and attn_len > valid_len:
      valid = Tensor.arange(attn_len, requires_grad=False, device=q.device).reshape(1, 1, 1, attn_len) < valid_len
      qk = valid.where(qk, qk.full_like(float("-inf")))
    if T > 1:
      qk = qk + Tensor.full((1, 1, T, valid_len), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1)
      attn_weights = qk.softmax(-1)
    else:
      e = qk.float().exp()
      attn_weights = (e / e.sum(-1, keepdim=True)).half()

    v = (attn_weights.matmul(k[:, :, :, :self.kv_lora_rank]) @ self.attn_v_b.weight.transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return x + self.attn_output(v)

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, "ffn_gate_exps"):
      router_logits = h_norm.float() @ self.ffn_gate_inp.weight.float().T
      if self.expert_gating_func == 2: gate_scores = router_logits.sigmoid()
      elif self.expert_gating_func == 3: gate_scores = router_logits
      else: gate_scores = router_logits.softmax(-1)
      selection_scores = gate_scores + self.exp_probs_b.bias if hasattr(self, "exp_probs_b") else gate_scores
      _, sel = topk_pairwise(selection_scores, self.num_experts_per_tok)
      probs = gate_scores.gather(-1, sel)
      if self.expert_gating_func == 3: probs = probs.softmax(-1)
      elif self.expert_weights_norm: probs = probs / probs.sum(axis=-1, keepdim=True).maximum(6.103515625e-5)

      gate = self.ffn_gate_exps(sel, h_norm).silu()
      up = self.ffn_up_exps(sel, h_norm)
      if self.split_moe_boundaries: gate, up = gate.contiguous(), up.contiguous()
      expert_out = self.ffn_down_exps(sel, gate * up)
      if self.split_moe_boundaries: expert_out = expert_out.contiguous()
      out = (expert_out * probs.unsqueeze(-1)).sum(axis=2) * self.expert_weights_scale
      if hasattr(self, "ffn_gate_shexp"):
        shexp_gate = self.ffn_gate_shexp(h_norm).silu().contiguous()
        shexp_up = self.ffn_up_shexp(h_norm).contiguous()
        if self.split_moe_boundaries: out = out.contiguous()
        out = out + self.ffn_down_shexp(shexp_gate * shexp_up)
      return h + out.cast(h.dtype)

    gated = self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp, freqs_cis:Tensor):
    return self._feed_forward(self._attention(x, start_pos, freqs_cis)).contiguous()

class GLMTransformer:
  def __init__(self, *, num_blocks:int, dim:int, hidden_dim:int, n_heads:int, norm_eps:float, vocab_size:int, rope_theta:float,
               max_context:int, q_lora_rank:int, kv_lora_rank:int, qk_nope_head_dim:int, qk_rope_head_dim:int, v_head_dim:int,
               num_experts:int, num_experts_per_tok:int, n_shared_experts:int, moe_hidden_dim:int, leading_dense_blocks:int,
               expert_gating_func:int, expert_weights_norm:bool, expert_weights_scale:float, mscale:float,
               yarn_scaling_factor:float=1.0, yarn_params:YarnParams|None=None):
    if yarn_params is not None:
      freqs_cis_cache = precompute_freqs_cis_yarn(qk_rope_head_dim, max_context, yarn_params).realize()
    elif yarn_scaling_factor > 1.0:
      freqs_cis_cache = precompute_freqs_cis(qk_rope_head_dim, max_context, rope_theta, yarn_scaling_factor).realize()
    else:
      freqs_cis_cache = precompute_freqs_cis(qk_rope_head_dim, max_context, rope_theta).realize()

    self.blk = []
    for i in range(num_blocks):
      is_dense = i < leading_dense_blocks
      blk = GLMTransformerBlock(dim, hidden_dim, n_heads, norm_eps, max_context, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                                qk_rope_head_dim, v_head_dim, 0 if is_dense else num_experts, num_experts_per_tok,
                                n_shared_experts, moe_hidden_dim, expert_gating_func, expert_weights_norm,
                                expert_weights_scale, mscale)
      self.blk.append(blk)
    self.freqs_cis_cache = freqs_cis_cache

    self.token_embd = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self.forward_jit = None

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)
    seqlen = tokens.shape[1]
    freqs_cis = self.freqs_cis_cache[start_pos:start_pos+seqlen]
    for block in self.blk: x = block(x, start_pos, freqs_cis)
    return self.output(self.output_norm(x))[:, -1, :].argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    if self.forward_jit is None:
      from tinygrad import TinyJit
      self.forward_jit = TinyJit(self.forward)
    use_jit = getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp)
    return (self.forward_jit if use_jit else self.forward)(tokens, start_pos)

  def generate(self, tokens:list[int], start_pos:int=0):
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    t = Tensor([tokens[start_pos:]], dtype="int32")
    use_sym = getenv("SYM", 1) and getenv("GLM_ATTN_MIN_BUCKET", 0) == 0
    while len(tokens) < self.max_context:
      pos = v_start_pos.bind(start_pos) if use_sym and start_pos != 0 and t.shape[-1] == 1 else start_pos
      t = self(t, pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id
