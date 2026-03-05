from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, itertools
from types import SimpleNamespace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.uop.ops import resolve
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    preset = {"qwen35": "qwen2"}.get(kv["tokenizer.ggml.pre"], kv["tokenizer.ggml.pre"])
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), preset)

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'qwen2': return [eos_id] + self.encode("\n")
    return [eos_id]

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

class DenseFFN:
  def __init__(self, block, dim:int, hidden_dim:int):
    block.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
    block.ffn_up   = nn.Linear(dim, hidden_dim, bias=False)
    block.ffn_down = nn.Linear(hidden_dim, dim, bias=False)
  def __call__(self, block, h_norm:Tensor) -> Tensor: return block.ffn_down(block.ffn_gate(h_norm).silu().contiguous() * block.ffn_up(h_norm))

class MoeFFN:
  def __init__(self, block, dim:int, hidden_dim:int, num_experts:int, num_experts_per_tok:int, shared_expert_hidden_dim:int,expert_weights_norm:bool):
    self.num_experts_per_tok = num_experts_per_tok
    self.expert_weights_norm = expert_weights_norm
    self.has_shared_experts  = shared_expert_hidden_dim > 0
    block.ffn_gate_inp       = nn.Linear(dim, num_experts, bias=False)
    block.ffn_gate_exps      = ExpertWeights(num_experts, dim, hidden_dim)
    block.ffn_up_exps        = ExpertWeights(num_experts, dim, hidden_dim)
    block.ffn_down_exps      = ExpertWeights(num_experts, hidden_dim, dim)
    if self.has_shared_experts:
      block.ffn_gate_inp_shexp = SimpleNamespace(weight=Tensor.empty(dim))
      block.ffn_gate_shexp     = nn.Linear(dim, shared_expert_hidden_dim, bias=False)
      block.ffn_up_shexp       = nn.Linear(dim, shared_expert_hidden_dim, bias=False)
      block.ffn_down_shexp     = nn.Linear(shared_expert_hidden_dim, dim, bias=False)
  def __call__(self, block, h_norm:Tensor) -> Tensor:
    x = h_norm.unsqueeze(2)
    probs, sel = block.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)
    if self.expert_weights_norm: probs = probs / probs.sum(-1, keepdim=True).maximum(2**-14)
    x_down = block.ffn_down_exps(sel, block.ffn_gate_exps(sel, x).silu() * block.ffn_up_exps(sel, x))
    out = (x_down * probs.unsqueeze(-1)).sum(axis=2)
    if not self.has_shared_experts: return out
    shared = block.ffn_down_shexp(block.ffn_gate_shexp(h_norm).silu() * block.ffn_up_shexp(h_norm))
    gate = (h_norm * block.ffn_gate_inp_shexp.weight).sum(axis=-1, keepdim=True).sigmoid()
    return out.contiguous() + shared * gate

class ResidualBlock:
  ffn: DenseFFN|MoeFFN
  @function(precompile=bool(getenv("PRECOMPILE", 0)))
  def _feed_forward(self, h: Tensor) -> Tensor: return h + self.ffn(self, self.norm_ffn_input(h))
  def init_block_state(self, x:Tensor, start_pos:int|UOp): pass
  def attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError
  def norm_ffn_input(self, h:Tensor) -> Tensor: raise NotImplementedError
  def __call__(self, x: Tensor, start_pos: int|UOp):
    self.init_block_state(x, start_pos)
    return self._feed_forward(self.attention(x, start_pos)).contiguous()

class TransformerBlock(ResidualBlock):
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float, max_context:int=0,
               qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, gated_attn:bool=False, rope_dim:int=0, shared_expert_hidden_dim:int=0,
               expert_weights_norm:bool=False):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_dim     = rope_dim or head_dim
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm
    self.gated_attn   = gated_attn

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = self.head_dim * n_heads * (2 if gated_attn else 1)
    kv_proj_out      = self.head_dim * n_kv_heads
    self.attn_q      = nn.Linear(dim, q_proj_out,  bias=False)
    self.attn_k      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_v      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_output = nn.Linear(self.head_dim * n_heads, dim,  bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(dim, norm_eps)
    if gated_attn: self.post_attention_norm = nn.RMSNorm(dim, norm_eps)
    else: self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    if qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(qk_norm, norm_eps), nn.RMSNorm(qk_norm, norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if num_experts == 0: self.ffn = DenseFFN(self, dim, hidden_dim)
    else: self.ffn = MoeFFN(self, dim, hidden_dim, num_experts, num_experts_per_tok, shared_expert_hidden_dim, expert_weights_norm)

  @function(precompile=bool(getenv("PRECOMPILE", 0)))
  def attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
    B, T, _ = x.shape
    if self.gated_attn: q, gate = q.reshape(B, T, self.n_heads, self.head_dim * 2).chunk(2, dim=-1)
    elif self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.qk_norm == self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs_cis = precompute_freqs_cis(self.rope_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q = apply_rope(q[..., :r], freqs_cis).cat(q[..., r:], dim=-1) if (r:=self.rope_dim) < self.head_dim else apply_rope(q, freqs_cis)
    k = apply_rope(k[..., :r], freqs_cis).cat(k[..., r:], dim=-1) if r < self.head_dim else apply_rope(k, freqs_cis)

    # TODO: fix assign to behave like this
    assigned_kv = self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.assign(Tensor.stack(k, v).contiguous().uop))
    tensor_assigned_kv = Tensor(assigned_kv, device=assigned_kv.device)
    k = tensor_assigned_kv[0, :, :, 0:start_pos+T, :]
    v = tensor_assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    if self.gated_attn: attn = attn * gate.reshape(B, T, -1).sigmoid()
    attn = self.attn_output(attn)
    return x + attn

  def init_block_state(self, x:Tensor, start_pos:int|UOp):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # NOTE: clone is used to promise the creation of a specific buffer
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.n_kv_heads, self.max_context, self.head_dim, device=x.device).clone()

  def norm_ffn_input(self, h:Tensor) -> Tensor: return self.post_attention_norm(h) if self.gated_attn else self.ffn_norm(h)

class GatedDeltaNetBlock(ResidualBlock):
  def __init__(self, dim:int, hidden_dim:int, norm_eps:float, head_k_dim:int, num_k_heads:int, num_v_heads:int, head_v_dim:int, conv_kernel:int,
               num_experts:int=0, num_experts_per_tok:int=0, shared_expert_hidden_dim:int=0,
               expert_weights_norm:bool=False):
    self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim = num_k_heads, num_v_heads, head_k_dim, head_v_dim
    self.conv_kernel, self.norm_eps, self.key_dim, self.value_dim = conv_kernel, norm_eps, head_k_dim * num_k_heads, head_v_dim * num_v_heads
    self.conv_channels = self.key_dim * 2 + self.value_dim
    self.attn_qkv            = nn.Linear(dim, self.conv_channels, bias=False)
    self.attn_gate           = nn.Linear(dim, self.value_dim, bias=False)
    self.ssm_beta            = nn.Linear(dim, num_v_heads, bias=False)
    self.ssm_alpha           = nn.Linear(dim, num_v_heads, bias=False)
    self.ssm_conv1d          = SimpleNamespace(weight=Tensor.zeros(self.conv_channels, conv_kernel))
    self.ssm_dt              = SimpleNamespace(bias=Tensor.zeros(num_v_heads))
    self.ssm_a               = Tensor.zeros(num_v_heads)
    self.ssm_norm            = nn.RMSNorm(head_v_dim, norm_eps)
    self.ssm_out             = nn.Linear(self.value_dim, dim, bias=False)
    self.attn_norm           = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)
    if num_experts == 0: self.ffn = DenseFFN(self, dim, hidden_dim)
    else: self.ffn = MoeFFN(self, dim, hidden_dim, num_experts, num_experts_per_tok, shared_expert_hidden_dim, expert_weights_norm)

  def init_state(self, x:Tensor):
    self.conv_state = Tensor.zeros(B:=x.shape[0], self.conv_kernel-1, self.conv_channels, dtype="float32", device=x.device).contiguous().realize()
    self.ssm_state  = Tensor.zeros(B, self.num_v_heads, self.head_v_dim, self.head_v_dim, dtype="float32", device=x.device).contiguous().realize()

  def attention(self, x:Tensor, _start_pos:int|UOp) -> Tensor:
    if (T:=x.shape[1].val if isinstance(x.shape[1], UOp) else x.shape[1]) == 1: return self.attention_step(x)
    return self.attention_prefill(x, T)

  def attention_prefill(self, x:Tensor, T:int) -> Tensor:
    # TODO - remove realize for @function support
    out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device).contiguous()
    for t in range(T):
      assigned = out.uop.after(out[:, t:t+1, :].uop.assign(self.attention_step(x[:, t:t+1, :]).contiguous().realize().uop))
      out = Tensor(assigned, device=assigned.device)
    return out

  def attention_step(self, x:Tensor) -> Tensor:
    # TODO - remove realizes for @function support
    B, xn = x.shape[0], self.attn_norm(x)
    qkv, z, beta, alpha = self.attn_qkv(xn), self.attn_gate(xn), self.ssm_beta(xn).sigmoid(), self.ssm_alpha(xn)
    gate = self.ssm_a * (alpha + self.ssm_dt.bias).softplus()
    ci = self.conv_state.cat(qkv, dim=1).contiguous()  # contiguous needed for correctness
    qkv = (ci * self.ssm_conv1d.weight.transpose(0, 1)).sum(axis=1, keepdim=True).silu()
    self.conv_state.assign(ci[:, 1:, :]).realize()
    q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
    q = q.reshape(B, 1, self.num_k_heads, self.head_k_dim).transpose(1, 2)
    k = k.reshape(B, 1, self.num_k_heads, self.head_k_dim).transpose(1, 2)
    v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim).transpose(1, 2)
    q = q.normalize(dim=-1, eps=self.norm_eps) * (self.head_k_dim**-0.5)
    k = k.normalize(dim=-1, eps=self.norm_eps)
    if self.num_k_heads != self.num_v_heads: q, k = q.repeat(1,r:=self.num_v_heads // self.num_k_heads, 1, 1), k.repeat(1, r, 1, 1)
    q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
    beta, gate = beta.reshape(B, self.num_v_heads, 1), gate.reshape(B, self.num_v_heads, 1, 1)
    St = (self.ssm_state * gate.exp()).transpose(-1, -2)
    d = (v - (k.unsqueeze(-2) @ St).squeeze(-2)) * beta
    St = St + k.unsqueeze(-1) * d.unsqueeze(-2)
    self.ssm_state.assign(St.transpose(-1, -2).contiguous()).realize()
    out = (q.unsqueeze(-2) @ St).squeeze(-2).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    out = (self.ssm_norm(out) * z.reshape(B, 1, self.num_v_heads, self.head_v_dim).silu()).reshape(B, 1, -1).cast(x.dtype)
    return x + self.ssm_out(out)

  def init_block_state(self, x:Tensor, start_pos:int|UOp):
    if not hasattr(self, 'conv_state') or (isinstance(start_pos, int) and start_pos == 0): self.init_state(x)

  def norm_ffn_input(self, h:Tensor) -> Tensor: return self.post_attention_norm(h)

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, full_attn_interval:int=0, rope_dim:int=0,
               ssm_state_size:int=0, ssm_group_count:int=0, ssm_time_step_rank:int=0, ssm_inner_size:int=0, ssm_conv_kernel:int=0,
               shared_expert_hidden_dim:int=0, expert_weights_norm:bool=False):
    def make_block(i):
      if (gated_attn:=bool(ssm_conv_kernel)) and full_attn_interval and (i+1)%full_attn_interval:
        return GatedDeltaNetBlock(dim, hidden_dim, norm_eps, ssm_state_size, ssm_group_count, ssm_time_step_rank, ssm_inner_size//ssm_time_step_rank,
                                   ssm_conv_kernel, num_experts, num_experts_per_tok, shared_expert_hidden_dim, expert_weights_norm)
      return TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context, qk_norm, num_experts,
                              num_experts_per_tok, gated_attn, rope_dim, shared_expert_hidden_dim, expert_weights_norm)
    self.blk = [make_block(i) for i in range(num_blocks)]
    self.has_gdn = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None).realize())

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
    if arch == 'llama':
      for name in state_dict:
        if 'attn_q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
        if 'attn_k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

    def g(k, d=0): return kv.get(f'{arch}.{k}', d)
    model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=(dim:=kv[f'{arch}.embedding_length']),
                        hidden_dim=g('expert_feed_forward_length') or kv[f'{arch}.feed_forward_length'],
                        n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
                        vocab_size=len(kv['tokenizer.ggml.tokens']),
                        head_dim=g('attention.key_length') or dim // n_heads,
                        rope_theta=kv[f'{arch}.rope.freq_base'], max_context=max_context,
                        qk_norm=next((int(state_dict[k].shape[0]) for k in state_dict if 'attn_q_norm.weight' in k), 0),
                        num_experts=g('expert_count', 0), num_experts_per_tok=g('expert_used_count', 0),
                        shared_expert_hidden_dim=(shared_expert_hidden_dim:=g('expert_shared_feed_forward_length', 0)),
                        # models with shared experts default to normalized expert weights
                        expert_weights_norm=g('expert_weights_norm', shared_expert_hidden_dim > 0),
                        full_attn_interval=g('full_attention_interval', 0), rope_dim=g('rope.dimension_count', 0),
                        ssm_state_size=g('ssm.state_size', 0), ssm_group_count=g('ssm.group_count', 0),
                        ssm_time_step_rank=g('ssm.time_step_rank', 0), ssm_inner_size=g('ssm.inner_size', 0),
                        ssm_conv_kernel=g('ssm.conv_kernel', 0))
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]):
    if self.has_gdn: return 0
    return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))

  def generate(self, tokens:list[int], chunk_size:int=32):
    if self.has_gdn: chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size) if chunk_size > 1 else None
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the kv cache
    start_pos = self.get_start_pos(tokens)
    prompt_len = len(tokens)
    out = t[:, :1]
    while len(tokens) < self.max_context:
      sp = v_start_pos.bind(start_pos)
      if start_pos < prompt_len:
        nt = v_toks.bind(min(chunk_size, prompt_len - start_pos)) if v_toks is not None else 1
        out = self(t[:, sp:sp+nt], sp).realize()
        start_pos += nt.val if isinstance(nt, UOp) else nt
        # chunked prefill: keep processing until all prompt tokens are consumed
        if start_pos < prompt_len: continue
      else:
        out = self(t[:, sp:sp+1] if self.has_gdn else out, sp).realize()
        start_pos += 1
      tokens.append(int(out.item()))
      if self.has_gdn: t[:, start_pos:start_pos+1].assign(out).realize()
      self._cached_tokens = tokens[:]
      yield tokens[-1]

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
}

# *** simple OpenAI compatible server on 11434 to match ollama ***
# OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama uvx --from gpt-command-line gpt

CHAT_HTML = b'''<!DOCTYPE html><html><head><title>tinygrad chat</title><style>
  * { margin: 0 }
  body { background: #212121; color: #e3e3e3; font-family: system-ui;
         height: 100vh; display: flex; flex-direction: column }
  #chat { flex: 1; overflow-y: auto; padding: 20px }
  .msg { padding: 10px 16px; margin: 8px 0; white-space: pre-wrap; border-radius: 18px }
  .user { background: #2f2f2f; margin-left: auto; width: fit-content; max-width: 70% }
  #input { max-width: 768px; width: 100%; margin: 20px auto; padding: 14px 20px;
           background: #2f2f2f; color: inherit; font: inherit;
           border: none; outline: none; resize: none; border-radius: 24px; field-sizing: content }
</style></head><body><div id="chat"></div>
<textarea id="input" rows="1" placeholder="Ask anything" autofocus></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'llama', messages: msgs, stream: true})});
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      for (const ln of dec.decode(value).split('\\n'))
        if (ln.startsWith('data: ') && !ln.includes('[DONE]'))
          try { d.textContent += JSON.parse(ln.slice(6)).choices[0]?.delta?.content || '' } catch {}
      chat.scrollTop = chat.scrollHeight;
    }
    msgs.push({role: 'assistant', content: d.textContent});
  }
</script></body></html>'''

class Handler(HTTPRequestHandler):
  def log_request(self, code='-', size='-'): pass
  def do_GET(self): self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False):
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"gen:{len(out)/(time.perf_counter()-pt):4.0f} tok/s  {colored('--', 'BLACK')}  out:{len(out):5d}\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens
      ids: list[int] = [bos_id] if bos_id is not None else []
      for msg in body["messages"]:
        ids += tok.role(msg["role"])
        # content can be a str or a list
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        ids += tok.end_turn(eos_id)
      ids += tok.role("assistant")

      # reply
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out = []
        for c in chunks: out.append(c["choices"][0]["delta"].get("content", "") if c["choices"] else "")
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":"stop"}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  raw_model = Tensor.from_url(models[args.model])
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  if DEBUG >= 1 or args.benchmark:
    print(f"using model {args.model} with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # TODO: why this is required to free the RAM of the GGUF copy?
  import gc
  gc.collect()

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # do benchmark
  if args.benchmark:
    gen = model.generate(toks:=[bos_id or 0])
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # start server
  if args.serve:
    # warmup: run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))
    TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  # interactive chat
  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
    except EOFError:
      break
    for next_id in model.generate(ids):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
