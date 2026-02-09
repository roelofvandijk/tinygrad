Fold q_nope @ attn_k_b into Q projection weights (remove one per-token matmul)
File: mla.py
Change: during weight load/init, precompose the q_nope projection with attn_k_b.weight, so _attention directly produces KV-rank query features.
Why high impact: removes a hot per-token matmul and one launch from every MLA block.
Risk: medium (weight-shape surgery, needs careful parity check).
Precompute RoPE cos/sin tensors in target dtype/shape once
File: mla.py
Change: cache freqs_cos/freqs_sin as [1,1,max_context,rope_dim/2] in model dtype, stop doing per-call reshape(...).cast(...) in _rope_interleaved.
Why: kills repeated tiny cast/reshape kernels across all layers/tokens.
Risk: low.
Fold (attn_kv @ attn_v_b^T) -> attn_output into one projection
File: mla.py
Change: precompose attn_v_b with attn_output.weight so attention output path becomes one linear from KV-rank-per-head to model dim.
Why: removes one large per-token matmul and a transpose/reshape barrier.
Risk: medium-high (weight algebra + load mapping).
Replace _topk_pairwise with a decode-fast top-k path for T==1
File: mla.py
Change: keep _topk_pairwise for prefill, but add a T==1 router path using direct top-k primitive/indexing (no O(n²) compare graph).
Why: MoE routing runs every layer/token; this is launch-heavy today.
Risk: medium (depends on available stable top-k op behavior).
Remove MoE gate/up forced contiguity, but only behind a flag
File: mla.py
Change: gate gate.contiguous() / up.contiguous() with env flag (default current behavior), so we can test on/off safely.
Why: gives fusion opportunity without forcing permanent regression risk.
Risk: low-medium (you already saw regression in one configuration).
If you want, I’ll implement #2 first (lowest risk, likely clean win), then you benchmark.