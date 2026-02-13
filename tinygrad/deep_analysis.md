  Deep Analysis: What Should Be Core, +10 tok/s, and Warmup

  1. The Custom DSL Kernels — Why They Exist

  After studying all the code, the custom DSL kernels in quantized.py (~300 lines) exist
  because of 3 scheduler deficiencies:

  A. MV heuristic can't see through dequant chains (heuristic.py:72-73)
  The MV heuristic requires MUL(INDEX, INDEX). The native dequantize_q4_0(blocks) @ x
  creates MUL(dequant_chain(AND/SHR/CAST/BITCAST...(INDEX)), INDEX). The MUL's weight
  operand is NOT an INDEX, so MV never fires. The fused kernel gets NO GROUP, NO LOCAL —
  runs at 2-15 GB/s instead of 300-1100 GB/s.

  B. No block-aligned reduction factoring
  The custom kernel explicitly splits reduction into (bpr, pair_in_block) = (I//32, 8).
  Each GROUP thread processes one Q4_0 block (same scale factor). The native path's flat
  reduction axis I gets GROUP'd arbitrarily, straddling block boundaries — the scale must
   be re-loaded when GROUP chunks cross blocks.

  C. Gather→reduce fusion requires single-consumer tensors
  Pre-separation (scale vs packed into separate tensors) was discovered to enable
  gather→reduce fusion, not for coalescing. The combined blocks[:, :2] and blocks[:, 2:]
  slices create TWO consumers → scheduler can't fuse the gather.

  2. What Should Be Part of tinygrad Core

  Fix the MV heuristic (heuristic.py:72-78)
  Walk backward from the MUL through CAST/AND/SHR/BITCAST chains to find the underlying
  INDEX. ~20-line change. This would make the native dequant(blocks) @ x path get
  GROUP+LOCAL+UPCAST automatically. It's the single most important core change — enables
  the scheduler to handle ALL quant types (Q4_0, Q4_K, Q5_K, Q6_K) without per-type
  custom kernels.

  Pre-separation should live in the GGML loader (state.py)
  Currently _ensure_q4_0_separated in quantized.py does this lazily. Move it into
  GGML_QUANT_INFO or the loading path so the native dequant path starts from
  pre-separated tensors. This removes the multi-consumer gather problem at source.

  3. The Honest Truth About +10 tok/s

  I investigated pre-composing the attention weight matrices (ideas #4 and #5) and
  discovered they're actually WORSE for MLA:

  - q_nope @ k_b: Composing into q_b would expand output from n_heads × qk_nope_head_dim
  = 20×128 = 2560 to n_heads × kv_lora_rank = 20×512 = 10240. That's 2.45x MORE
  computation (7.2M → 17.7M MACs).
  - v_b @ attn_output: Same issue — kv_lora_rank (512) > v_head_dim (128), so composing
  undoes the latent compression. 3.2x MORE computation.

  MLA's whole design is that kv_lora_rank < n_heads × head_dim. The current absorbed
  attention is already the efficient path.

  The custom DSL kernels are hitting 300-1100 GB/s. The heuristic is already well-tuned
  for Q4_0. To get +10 tok/s from 30.5 purely through DSL without custom MSL, we'd need
  to find ~7.8ms of waste per token. The honest candidates:
  ┌──────────────────────────────────────────────────┬───────────────┬────────────┐
  │                      Source                      │    Saving     │ tok/s gain │
  ├──────────────────────────────────────────────────┼───────────────┼────────────┤
  │ MV heuristic fix → remove custom kernel overhead │ ~1-2ms        │ +1-2       │
  ├──────────────────────────────────────────────────┼───────────────┼────────────┤
  │ Fuse more ops (norm+linear, split+silu*up)       │ ~0.5ms        │ +0.5       │
  ├──────────────────────────────────────────────────┼───────────────┼────────────┤
  │ Better topk (replace O(n²) pairwise)             │ ~0.3ms        │ +0.3       │
  ├──────────────────────────────────────────────────┼───────────────┼────────────┤
  │ Reduce Python generate() overhead                │ ~1ms          │ +1         │
  ├──────────────────────────────────────────────────┼───────────────┼────────────┤
  │ Remove .contiguous() on block output (line 176)  │ needs testing │ ?          │
  └──────────────────────────────────────────────────┴───────────────┴────────────┘
  That's maybe +3-4 tok/s total from DSL improvements. Getting to +10 would require
  either custom MSL, speculative decoding, or deep scheduler restructuring.

  4. The Warmup Fix — This is the Real Win

  The warmup is 23s + 56s = 79 seconds because tokens 1 and 2 compile completely
  different kernel sets:

  Token 1 (cnt=0): Uses concrete start_pos=0 → compiles ~600 kernels for the concrete
  graph
  Token 2 (cnt=1, JIT capture): Uses symbolic v_start_pos.bind(1) → compiles ~600 NEW
  kernels because the symbolic graph is structurally different

  The reason token 1 uses concrete is mla.py:121:
  if not hasattr(self, "cache_k") or start_pos == 0:
      self.cache_k = Tensor.empty(...).contiguous().realize()

  When start_pos=0, this allocates the KV cache, which creates a different computation
  graph from the symbolic path where the cache already exists. So llm.py forces
  start_pos=0 to be concrete.

  The fix: Pre-allocate KV caches at model load time, not on first forward pass. Then:
  - Token 1 can use symbolic start_pos (same code path as token 2)
  - Token 2 (JIT capture) hits ALL schedule + compilation caches from token 1
  - Token 2 goes from 56s → ~2s (just JIT capture overhead, zero new compilation)
  - Total warmup: 79s → ~25s (3x improvement)

  This is a small, clean change:
  1. mla.py:__init__: pre-allocate self.cache_k with B=1, max_context, cache_dim
  2. Remove start_pos == 0 re-allocation check
  3. llm.py: always use symbolic start_pos (remove the start_pos != 0 guard)

  The remaining 23s (token 1) is serial Metal compilation. A second improvement would be
  to parallelize Metal compilation using a thread pool — each
  MTLCodeGenServiceBuildRequest is independent. But that's a deeper tinygrad core change.

  Summary of Recommendations
  Priority: 1
  Change: Pre-allocate KV cache to eliminate double compilation
  Location: mla.py, llm.py
  Impact: 79s → 25s warmup
  ────────────────────────────────────────
  Priority: 2
  Change: MV heuristic: walk through dequant chains
  Location: heuristic.py
  Impact: Enable removing custom kernels
  ────────────────────────────────────────
  Priority: 3
  Change: Move pre-separation to GGML loader
  Location: state.py
  Impact: Simplify quantized.py
  ────────────────────────────────────────
  Priority: 4
  Change: Parallel Metal compilation
  Location: ops_metal.py
  Impact: 25s → ~8s warmup
  Want me to implement the KV cache pre-allocation (warmup fix) first? It's the
  highest-impact, lowest-risk change and would make every development iteration faster.