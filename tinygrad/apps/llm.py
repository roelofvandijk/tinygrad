from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools
from types import SimpleNamespace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored
from tinygrad.dtype import dtypes
from tinygrad.nn.state import ggml_data_to_tensor, GGML_BLOCK_SIZES
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","glm4"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
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
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"])

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
    if self.preset in ('olmo', 'glm4'): return self.encode("<|" + role + "|>\n")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'glm4': return []
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
  def __init__(self, num_experts:int, in_features:int, out_features:int, ggml_type:int=0):
    self.in_features, self.out_features, self.ggml_type = in_features, out_features, ggml_type
    if ggml_type:
      epb, bpb = GGML_BLOCK_SIZES[ggml_type]
      self.expert_blocks = Tensor.empty(num_experts, out_features * in_features // epb, bpb, dtype=dtypes.uint8)
    else: self.weight = Tensor.empty(num_experts, out_features, in_features)

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    if not self.ggml_type: return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)  # unquantized path
    out_shape = (*sel.shape, self.out_features)
    x_flat = x.expand(*sel.shape, self.in_features).reshape(-1, self.in_features)
    sel_blocks = self.expert_blocks[sel.reshape(-1)]
    n_elements = int(sel.numel() * self.out_features * self.in_features)
    w = ggml_data_to_tensor(sel_blocks.reshape(-1, sel_blocks.shape[-1]).flatten(), n_elements, self.ggml_type)
    w = w.reshape(-1, self.out_features, self.in_features)
    if getenv("HALF", 1): w = w.half()
    return (x_flat.unsqueeze(-2) @ w.transpose(-1, -2)).reshape(out_shape)

def apply_rope(x:Tensor, freqs_cis:Tensor, interleaved:bool=False) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  if interleaved:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return (x1 * cos - x2 * sin).unsqueeze(-1).cat((x2 * cos + x1 * sin).unsqueeze(-1), dim=-1).flatten(-2)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

class FFN:
  def __init__(self, dim:int, hidden_dim:int, norm_eps:float, num_experts:int=0, num_experts_per_tok:int=0, n_shared_experts:int=0,
               moe_hidden_dim:int=0, expert_weights_norm:bool=False, expert_weights_scale:float=1.0, expert_ggml_types:tuple[int,int,int]=(0,0,0)):
    self.norm        = nn.RMSNorm(dim, norm_eps)
    self.num_experts = num_experts
    if num_experts > 0:
      self.num_experts_per_tok  = num_experts_per_tok
      self.expert_weights_norm  = expert_weights_norm
      self.expert_weights_scale = expert_weights_scale
      self.gate_inp             = nn.Linear(dim, num_experts, bias=False)
      if expert_weights_norm or n_shared_experts > 0: self.exp_probs_b = SimpleNamespace(bias=Tensor.empty(num_experts))
      self.gate_exps   = ExpertWeights(num_experts, dim, moe_hidden_dim or hidden_dim, expert_ggml_types[0])
      self.up_exps     = ExpertWeights(num_experts, dim, moe_hidden_dim or hidden_dim, expert_ggml_types[1])
      self.down_exps   = ExpertWeights(num_experts, moe_hidden_dim or hidden_dim, dim, expert_ggml_types[2])
      if n_shared_experts > 0:
        shexp_hidden    = n_shared_experts * moe_hidden_dim
        self.gate_shexp = nn.Linear(dim, shexp_hidden, bias=False)
        self.up_shexp   = nn.Linear(dim, shexp_hidden, bias=False)
        self.down_shexp = nn.Linear(shexp_hidden, dim, bias=False)
    else:
      self.gate = nn.Linear(dim, hidden_dim, bias=False)
      self.up   = nn.Linear(dim, hidden_dim,  bias=False)
      self.down = nn.Linear(hidden_dim, dim,  bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    h = self.norm(x)
    if self.num_experts == 0: return x + self.down(self.gate(h).silu() * self.up(h))  # Standard FFN (SwiGLU)
    # Compute expert selection and probabilities
    if not hasattr(self, 'exp_probs_b'): probs, sel = self.gate_inp(h).softmax(-1).topk(self.num_experts_per_tok)
    else:
      gate_scores = (h.float() @ self.gate_inp.weight.float().T).sigmoid()
      _, sel = (gate_scores + self.exp_probs_b.bias).topk(self.num_experts_per_tok)
      probs = gate_scores.gather(-1, sel)
      if self.expert_weights_norm: probs = probs / probs.sum(axis=-1, keepdim=True).maximum(2**-14)
    x_exp = h.unsqueeze(2)
    # Expert computation: gate * silu * up → down
    gated = self.gate_exps(sel, x_exp).silu() * self.up_exps(sel, x_exp)
    down = self.down_exps(sel, gated * probs.unsqueeze(-1).cast(gated.dtype) if hasattr(self, 'exp_probs_b') else gated)
    out = down.sum(axis=2) if hasattr(self, 'exp_probs_b') else (down * probs.unsqueeze(-1)).sum(axis=2)
    if hasattr(self, 'exp_probs_b'): out = out * self.expert_weights_scale
    if hasattr(self, 'gate_shexp'): out = out + self.down_shexp(self.gate_shexp(h).silu() * self.up_shexp(h))  # Shared expert
    return x + out.cast(x.dtype)

class Attention:
  def __init__(self, dim:int, n_heads:int, n_kv_heads:int, head_dim:int, max_context:int, rope_theta:float, qk_norm:int=0, norm_eps:float=1e-5):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out        = self.head_dim * n_heads
    kv_proj_out       = self.head_dim * n_kv_heads
    self.q            = nn.Linear(dim, q_proj_out,  bias=False)
    self.k            = nn.Linear(dim, kv_proj_out, bias=False)
    self.v            = nn.Linear(dim, kv_proj_out, bias=False)
    self.output       = nn.Linear(q_proj_out, dim,  bias=False)
    if qk_norm: self.q_norm, self.k_norm = nn.RMSNorm(qk_norm, norm_eps), nn.RMSNorm(qk_norm, norm_eps)
    self.norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, start_pos:int|UOp, mask:Tensor|None) -> Tensor:
    x = self.norm(x)
    q, k, v = self.q(x), self.k(x), self.v(x)
    if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.q_norm(q), self.k_norm(k)
    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
    if self.qk_norm == self.head_dim: q, k = self.q_norm(q), self.k_norm(k)
    freqs_cis = precompute_freqs_cis(self.head_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q, k = apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.empty(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=k.dtype, device=k.device)
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    k, v = self.cache_kv[0, :, :, 0:start_pos+T, :], self.cache_kv[1, :, :, 0:start_pos+T, :]
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    return self.output(attn.transpose(1, 2).reshape(B, T, -1))

class MLAAttention:
  def __init__(self, dim:int, n_heads:int, norm_eps:float, max_context:int, q_lora_rank:int, kv_lora_rank:int,
               qk_nope_head_dim:int, qk_rope_head_dim:int, v_head_dim:int, rope_theta:float=10000.0):
    self.n_heads, self.qk_nope_head_dim, self.qk_rope_head_dim = n_heads, qk_nope_head_dim, qk_rope_head_dim
    self.q_head_dim, self.kv_lora_rank, self.max_context = qk_nope_head_dim + qk_rope_head_dim, kv_lora_rank, max_context
    self._attn_scale, self._rope_theta = self.q_head_dim ** -0.5, rope_theta
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.q_a = nn.Linear(dim, q_lora_rank, bias=False)
    self.q_a_norm = nn.RMSNorm(q_lora_rank, norm_eps)
    self.q_b = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)
    self.kv_a_mqa, self.kv_a_norm = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False), nn.RMSNorm(kv_lora_rank, norm_eps)
    self.k_b = SimpleNamespace(weight=Tensor.empty(n_heads, kv_lora_rank, qk_nope_head_dim))
    self.v_b = SimpleNamespace(weight=Tensor.empty(n_heads, v_head_dim, kv_lora_rank))
    self.output = nn.Linear(n_heads * v_head_dim, dim, bias=False)

  def __call__(self, x:Tensor, start_pos:int|UOp, mask:Tensor|None) -> Tensor:
    x = self.norm(x)
    B, T = x.shape[0], x.shape[1]
    q = self.q_b(self.q_a_norm(self.q_a(x)))
    kv_out = self.kv_a_mqa(x)
    q = q.reshape(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv, k_pe = kv_out.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    freqs_cis = precompute_freqs_cis(self.qk_rope_head_dim, self.max_context, self._rope_theta)[start_pos:start_pos+T].half()
    q_pe = apply_rope(q_pe, freqs_cis, interleaved=True)
    k_pe = apply_rope(k_pe.reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2), freqs_cis, interleaved=True)
    q = (q_nope @ self.k_b.weight.transpose(-1, -2)).cat(q_pe, dim=-1)
    kv_normed = self.kv_a_norm(compressed_kv).unsqueeze(1)
    k_new = kv_normed.cat(k_pe, dim=-1)
    if not hasattr(self, "cache_k") or start_pos == 0:
      self.cache_k = Tensor.empty((B, 1, self.max_context, self.kv_lora_rank + self.qk_rope_head_dim), dtype=kv_normed.dtype, device=kv_normed.device)
    self.cache_k[:, :, start_pos:start_pos+T, :].assign(k_new).realize()
    k = self.cache_k[:, :, 0:start_pos+T, :]
    qk = q.matmul(k.transpose(-2, -1)) * self._attn_scale
    if mask is not None: qk = qk + mask
    attn_val = (qk.softmax(-1).matmul(k[:, :, :, :self.kv_lora_rank]) @ self.v_b.weight.transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.output(attn_val)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, expert_ggml_types:tuple[int,int,int]=(0,0,0),
               kv_lora_rank:int=0, q_lora_rank:int=0, qk_nope_head_dim:int=0, qk_rope_head_dim:int=0, v_head_dim:int=0,
               n_shared_experts:int=0, moe_hidden_dim:int=0, expert_weights_norm:bool=False, expert_weights_scale:float=1.0):
    if kv_lora_rank > 0:
      self.attn = MLAAttention(dim, n_heads, norm_eps, max_context, q_lora_rank, kv_lora_rank,
                               qk_nope_head_dim, qk_rope_head_dim, v_head_dim, rope_theta)
    else:
      self.attn = Attention(dim, n_heads, n_kv_heads, head_dim, max_context, rope_theta, qk_norm, norm_eps)
    self.ffn = FFN(dim, hidden_dim, norm_eps, num_experts, num_experts_per_tok,
                   n_shared_experts, moe_hidden_dim, expert_weights_norm, expert_weights_scale, expert_ggml_types)

  def __call__(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    T = x.shape[1]
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    x = x + self.attn(x, start_pos, mask)
    return self.ffn(x).contiguous()


class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0,
               expert_ggml_types:list[tuple[int,int,int]]|None=None, leading_dense_blocks:int=0,
               kv_lora_rank:int=0, q_lora_rank:int=0, qk_nope_head_dim:int=0, qk_rope_head_dim:int=0, v_head_dim:int=0,
               n_shared_experts:int=0, moe_hidden_dim:int=0, expert_weights_norm:bool=False, expert_weights_scale:float=1.0):
    egt = expert_ggml_types or [(0,0,0)] * num_blocks
    self.blk: list[TransformerBlock] = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context, qk_norm,
                                                         num_experts=0 if i < leading_dense_blocks else num_experts,
                                                         num_experts_per_tok=num_experts_per_tok, expert_ggml_types=egt[i],
                                                         kv_lora_rank=kv_lora_rank, q_lora_rank=q_lora_rank,
                                                         qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim,
                                                         n_shared_experts=n_shared_experts, moe_hidden_dim=moe_hidden_dim,
                                                         expert_weights_norm=expert_weights_norm, expert_weights_scale=expert_weights_scale)
                                        for i in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def _process_quantized_experts(state_dict: dict[str, Tensor], kv: dict) -> list[tuple[int,int,int]]:
    """Reshapes expert weight blocks in state_dict and returns per-block (gate, up, down) ggml_type tuples."""
    bpb_to_type = {v[1]: k for k, v in GGML_BLOCK_SIZES.items()}
    num_experts = kv.get(f'{kv["general.architecture"]}.expert_count', 0)
    num_blocks = kv[f'{kv["general.architecture"]}.block_count']
    block_types: dict[int, dict[str, int]] = {}
    for name in [k for k in state_dict if k.endswith('.weight') and '_exps.' in k and state_dict[k].dtype == dtypes.uint8]:
      parts = name.split('.')
      blk_idx, kind = int(parts[1]), parts[3].removesuffix('_exps')
      block_types.setdefault(blk_idx, {})[kind] = bpb_to_type[state_dict[name].shape[-1]]
      blocks = state_dict.pop(name)
      state_dict[name.removesuffix('.weight') + '.expert_blocks'] = blocks.reshape(num_experts, blocks.shape[0] // num_experts, blocks.shape[1])
    bt = block_types
    return [(bt.get(i, {}).get('gate', 0), bt.get(i, {}).get('up', 0), bt.get(i, {}).get('down', 0)) for i in range(num_blocks)]

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=True, quantized:bool=False) -> tuple[Transformer, dict]:
    kv, state_dict = nn.state.gguf_load(gguf, keep_quantized=(lambda name, shape, typ: '_exps.' in name) if quantized else None)

    # Remap GGUF flat keys (attn_q, ffn_gate) to component hierarchy (attn.q, ffn.gate)
    state_dict = {k.replace('exp_probs_b.', 'ffn.exp_probs_b.').replace('attn_', 'attn.').replace('ffn_', 'ffn.'): v for k, v in state_dict.items()}

    expert_ggml_types_list = Transformer._process_quantized_experts(state_dict, kv) if quantized else None

    # all non-block state items should be float16, not float32
    state_dict = {k: v.cast('float16') if getenv("HALF", 1) and v.dtype != dtypes.uint8 else v for k, v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict and 'token_embd.weight' in state_dict:
      state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
    if arch == 'llama':
      for name in state_dict:
        if 'attn.q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
        if 'attn.k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

    def ak(s, d=0): return kv.get(f'{arch}.{s}', d)
    key_length_mla = ak('attention.key_length_mla', ak('attention.key_length'))
    model = Transformer(num_blocks=ak('block_count'), dim=ak('embedding_length'),
                        hidden_dim=ak('feed_forward_length', ak('expert_feed_forward_length', 0)),
                        n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=ak('attention.layer_norm_rms_epsilon'),
                        vocab_size=len(kv['tokenizer.ggml.tokens']),
                        head_dim=ak('attention.key_length', ak('embedding_length') // n_heads),
                        rope_theta=ak('rope.freq_base'), max_context=max_context,
                        qk_norm=int(state_dict['blk.0.attn.q_norm.weight'].shape[0]) if 'blk.0.attn.q_norm.weight' in state_dict else 0,
                        num_experts=ak('expert_count', 0), num_experts_per_tok=ak('expert_used_count', 0),
                        expert_ggml_types=expert_ggml_types_list, leading_dense_blocks=ak('leading_dense_block_count'),
                        kv_lora_rank=ak('attention.kv_lora_rank'), q_lora_rank=ak('attention.q_lora_rank'),
                        qk_nope_head_dim=key_length_mla - ak('rope.dimension_count') if key_length_mla > 0 else 0,
                        qk_rope_head_dim=ak('rope.dimension_count'),
                        v_head_dim=ak('attention.value_length_mla', ak('attention.value_length')),
                        n_shared_experts=ak('expert_shared_count'), moe_hidden_dim=ak('expert_feed_forward_length'),
                        expert_weights_norm=ak('expert_weights_norm', False), expert_weights_scale=ak('expert_weights_scale', 1.0))
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())  # removing this would cut 5s warmup time for glm
    if realize: Tensor.realize(*params)
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

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
  "glm-4.7:flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_0.gguf",
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
<textarea id="input" rows="1" placeholder="Ask anything"></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }
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
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  in:{len(ids):5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{len(ids)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id in stop_tokens: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"out:{len(out):5d}  {colored('--', 'BLACK')}  gen: {len(out)/(time.perf_counter()-pt):4.0f} tok/s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens
      ids: list[int] = [bos_id] if bos_id is not None else []
      if tok.preset == 'glm4': ids += tok.encode("<sop>")
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
      if tok.preset == 'glm4': ids += tok.encode("<think>\n")

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
  parser.add_argument("--model", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(Tensor.from_url(models[args.model]), args.max_context, quantized=args.model.startswith("glm"))
  if DEBUG >= 1: print(f"using model {args.model}")

  # do benchmark
  if args.benchmark:
    param_bytes = sum(x.nbytes() for x in nn.state.get_parameters(model))
    gen = model.generate([0], 0)
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s, param {param_bytes/x:7.2f} GB/s"): next(gen)
    exit(0)

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']
  stop_tokens = {eos_id} | ({tok.encode("<|user|>")[0]} if tok.preset == 'glm4' else set())

  # start server
  if args.serve: TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  ids: list[int] = [bos_id] if bos_id is not None else []
  if tok.preset == 'glm4': ids += tok.encode("<sop>")
  while 1:
    start_pos = max(len(ids) - 1, 0)
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
      if tok.preset == 'glm4': ids += tok.encode("<think>\n")
    except EOFError:
      break
    for next_id in model.generate(ids, start_pos):
      sys.stdout.write(tok.decode([next_id]) if next_id not in stop_tokens else "\n\n")
      sys.stdout.flush()
      if next_id in stop_tokens: break
