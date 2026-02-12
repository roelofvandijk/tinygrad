# GLM-4.7-Flash Q4_0: Cleanup & Speed Unlock Tips

Status: DSL ~31 tok/s, MSL ~35 tok/s, llama.cpp ~35 tok/s.
Goal: surpass both with pure tinygrad DSL on Metal.

---

## 10 Cleanup Suggestions

### 1. Remove spec.py transitional workarounds
The 3 new patterns (`INDEX(VECTORIZE,_)`, `INDEX(CAST(DEFINE_REG),_)`, `STORE(LOAD,VECTORIZE)`)
were added to unblock upcasted custom kernels. Journal proves they don't work — the auto path
(`QL_DSL_Q4_LINEAR_AUTO=1`) still crashes at Metal compile after passing SPEC. These mask real
bugs and add tech debt for zero benefit.
**Evidence**: journal "re-test auto DSL schedule" (line 1783-1788): "still fails in Metal codegen
for upcasted/vectorized custom Q4 linear path".

### 2. Remove unused kernel variants from quantized.py
`custom_q4_0_linear_split`, `custom_q4_0_linear_vec2`, `custom_q4_0_mul_mat_id_split`, and
`custom_fp16_linear` are defined but never wired into any production path. No journal entry shows
them producing a benchmark win. They add 150+ lines of dead code and 4 dead `_opts` functions.

### 3. Collapse env-var switchboard into hardcoded winners
The uncommitted diff adds ~15 env vars (`QL_DSL_Q4_LINEAR_LOCAL`, `_AXIS`, `_GROUP`, `_UPCAST`,
`_GROUPTOP_2048_5120`, `_LOCAL_2048_5120`, `_LOCAL_5120_768`, `_LOCAL_768_2048`,
`QL_DSL_MOE_LOCAL_*`, `QL_DSL_MOE_GROUP_*`, `QL_DSL_MOE_SPLIT_*`, `QL_DSL_Q4_LINEAR_AUTO`).
Journal A/B tests already identified winners:
- Dense Q4: `LOCAL(0, 32)` for 2048x5120, `LOCAL(0, 16)` for 5120x768 and 768x2048.
- MoE: `LOCAL(1,8)+GROUP(1,16)` for 3072x2048, `LOCAL(1,16)+GROUP(1,16)` for 2048x1536.
Hardcode these. Keep at most one `QL_DSL_DEBUG_OPTS=1` escape hatch for future sweeps.

### 4. Remove cstyle.py VECTORIZE(ptr) rewrite
Journal says it "moved the compile error forward but did not fully resolve upcast compile
validity" (line 1819-1820). It's dead code for a broken path.

### 5. Squash WIP commits before merge
"29 tok", "faster?", "md", "this hit's 35 tok/s! (with MSL)", "wip faster dsl" are not
reviewable. Squash into 2-3 commits with messages describing what changed, why, and perf delta.

### 6. Separate MSL files into their own branch/feature flag
`q4_moe_msl.py`, `qk_linear_msl.py`, `q4_linear_msl.py` plus their `realize.py` dispatch are
useful as a performance reference and ceiling measurement, but they conflict with the pure-DSL
goal. Move them behind a single `QL_MSL=1` flag (already partially there) and keep them out of
the default code path so DSL regressions are immediately visible.

### 7. Remove bench_q4_linear_dsl.py TinyJit change from this diff
The JIT wrapping is a good improvement but belongs in a separate "benchmark tooling" commit,
not mixed with kernel optimization changes.

### 8. Clean up linearizer.py and cstyle.py trivial diffs
One is a missing newline, the other a blank line. These create noise in review. Either commit
them separately as formatting or drop them.

### 9. Remove _blocks_cache dead code paths
`_blocks_cache` and `_ensure_blocks_cache` in QuantizedLinear exist only for the MSL shared
expert path (`QL_SHEXP_MSL=1`). If MSL is being separated (suggestion 6), this goes with it.

### 10. Add unit tests for custom kernel correctness
No test file covers `custom_q4_0_linear` or `custom_q4_0_mul_mat_id` with the new 2-byte
reduction and pre-separated scale/packed inputs. One small test per kernel shape that compares
custom_kernel output against reference Tensor dequant+matmul would catch regressions from
future scheduler/codegen changes. Journal shows GROUPTOP already produced silently wrong results
(line 1689-1691) — a test would have caught that.

---

## 10 Avenues for Massive Speed Unlock

### 1. Fix GROUPTOP numerical correctness for dense Q4 (potential ~10x on hottest kernel)
**Journal evidence**: "Tried LOCAL+GROUP first: huge perf win but numerically wrong on microbench
(max_diff enormous)" (line 1689-1691). GROUPTOP is the single biggest known unlock — it would
give the DSL kernel the same cooperative-reduce structure that makes MSL fast. The MSL kernel
does `simd_sum(acc)` across 32 threads; GROUPTOP is the DSL equivalent (threadgroup shared memory
+ barrier + final reduce).
**Root cause to investigate**: GROUP splits a REDUCE axis into (r_local × group_size + r_group).
For the Q4 custom kernel, the reduction axes are already non-trivial expressions
(`br = r//8, jb = (r%8)*2`). The `fix_group_for_reduce` expander may be mis-extracting
GROUP_REDUCE ranges from these embedded expressions, producing a wrong final combine. Compare
the Metal source of the GROUPTOP kernel (DEBUG=5) against the MSL reference to find the
divergence point.
**Why this is #1**: MSL's Q4 dense kernel is 24.7x faster than the original serial DSL kernel.
LOCAL alone recovered ~2.5x. GROUPTOP would recover most of the remaining ~10x gap by enabling
parallel reduction across the 32-thread SIMD group.

### 2. Fix renderer/linearizer for upcasted custom-kernel codegen
**Journal evidence**: "Found malformed scalar/vector access in emitted C-style source
(`(*(acc0+0)).x/.y` on scalar float and pointer-vector expressions)" (line 1817-1818). The auto
scheduling path (`opts_to_apply=None`) crashes because the renderer emits invalid Metal when
UPCAST is applied to custom reduce kernels.
**Root cause**: `gpudims.py` assumes INDEX base is always PtrDType (partially fixed), but the
deeper issue is that VECTORIZE of register pointers and scalar accumulators aren't properly
lowered through the cstyle renderer. The renderer needs to handle the case where UPCAST creates
`float.vec(N)` accumulators with `DEFINE_REG` storage — these should emit as `float acc[N]`
arrays, not as vector types with `.x/.y` component access.
**Why this matters**: Enabling auto-scheduling would let BEAM find optimal schedules for custom
kernels instead of requiring hand-tuned `opts_to_apply` per shape.

### 3. Q5_K/Q6_K online dequant via DSL custom kernels (replace MSL shared expert path)
**Journal evidence**: MSL shared-expert online dequant gave +16.9% bench proxy, +8.9% full model
(lines 1595-1608). The existing non-MSL path dequantizes full weights to fp16 cache, which is
bandwidth-heavy.
**What to do**: Write `custom_q5_k_linear` and `custom_q6_k_linear` DSL kernels using the same
SINK/CALL primitive boundary as the Q4_0 kernel. The Q5_K format is 176 bytes/block with 5-bit
values + 6-bit scales; Q6_K is 210 bytes/block with 6-bit values. Pre-separate the component
tensors (like Q4_0 scale/packed separation) so the scheduler can fuse gather→dequant→reduce.
**Estimated impact**: Currently the shared expert linears (`ffn_gate_up_shexp` Q5_K 3072x2048,
`ffn_down_shexp` Q6_K 2048x1536) run 46 times per token. MSL shows ~92us and ~54us respectively.
The fp16-cache DSL path is much slower. Closing this gap with DSL kernels would recover most of
the +8.9% that currently requires MSL.

### 4. Reduce kernel count per token (1450 → target <500)
**Journal evidence**: "Kernel-count pressure is as important as kernel GFLOPS; any path that adds
barriers/materialization is likely a net loss" (line 465). Profile shows 1450 kernels/token across
6 ICBs with 184us average overhead per kernel.
**Where the count comes from**: 30 kernels per MoE block × 46 blocks = 1380 kernels just for MoE.
Many of these are small elementwise/norm kernels that could be fused.
**What to do**: (a) Fuse RMSNorm+first-linear into one kernel (currently 2-3 kernels per norm).
(b) Fuse router softmax+topk into one kernel (currently ~4 kernels). (c) Fuse silu+mul
elementwise into the preceding reduce kernel. Each fusion that removes one kernel per block saves
46 kernel launches per token. Even getting from 30→20 kernels per block would cut 460 launches
and ~85ms of overhead.

### 5. Strided-thread reduction pattern as a scheduler primitive
**Journal evidence**: The DEBUG=6 MSL vs DSL comparison (lines 1642-1651) identifies the core
structural gap: MSL uses `for (br = tid; br < BPR; br += THREADS)` where each of 32 threads
handles BPR/32 blocks independently, then `simd_sum` combines. DSL generates serial nested loops
(`for Ridx2_0 in 0..159 then for Ridx2_1 in 0..15`).
**What's missing in tinygrad**: GROUP/GROUPTOP split a reduce axis into two nested axes with
shared-memory materialization between them. What MSL does is different — it distributes the
*same* reduce axis across threads (like LOCAL for globals, but for reduces), then uses SIMD
intrinsics to combine. This is closer to a "LOCAL for reduce" or "PARALLEL_REDUCE" primitive.
**Prototype approach**: A new `OptOps.REDUCE_LOCAL` that distributes a REDUCE range across
threadgroup threads and emits `simd_sum` / shared-memory tree-reduce in the epilogue. This
would be the DSL-native equivalent of llama.cpp's matvec pattern.

### 6. Packed ushort loads for Q4_0 dequant (4 nibbles per load instead of 2)
**Journal evidence**: MSL uses `ushort q = *reinterpret_cast<const device ushort*>(pb + j)` to
load 2 bytes at once and extract 4 nibbles via bitmask (`q & 0x000F`, `q & 0x00F0`, etc.). DSL
loads individual bytes and extracts 2 nibbles each (`q & 0xF`, `q >> 4`).
**What to do**: The DSL 2-byte reduction step already processes pairs of bytes per iteration, but
the generated Metal still emits two separate `uchar` loads. If the renderer could emit a single
`ushort` load (via bitcast or explicit cast in the index expression), it would halve the number
of memory transactions in the inner loop. This could be expressed as loading
`packed.cast(dtypes.ushort)[o, br, jb//2]` and extracting 4 nibbles with shift+mask.

### 7. Scale constant hoisting outside inner loop
**Journal evidence**: "scale decode from two bytes to half appears inside the inner loop
expression, so scale work is repeated many times" (line 1646). MSL pre-computes per-block
constants once: `s = scale[br]`, then derives `s16 = s/16`, `s256 = s/256`, `md = -8*s`.
**What to do**: The pre-separation of scale from packed already helped (+25% from gather→reduce
fusion), but the generated kernel still reloads scale inside the j-loop (16 iterations per block).
The scale value is loop-invariant over j — it only changes with `br`. If the scheduler/linearizer
could hoist the scale load outside the inner packed-byte loop, it would eliminate 15/16 redundant
loads per block. This might require teaching the linearizer about loop-invariant code motion for
LOAD ops, or restructuring the DSL kernel to make the invariance explicit (e.g., separate
`br`-indexed and `j`-indexed ranges).

### 8. Multi-output-row (NR>1) for dense Q4 via proper DSL vectorization
**Journal evidence**: MSL Q5_K shared expert uses NR=2 as accepted default (line 313-318). The
`custom_q4_0_linear_vec2` attempt in the diff tried this but couldn't compile due to vectorized
pointer/store codegen issues (same blocker as suggestion #2).
**What to do**: Once the UPCAST codegen fix (#2) lands, apply `UPCAST` on the output axis to
compute 2-4 output rows per thread. This reuses the same activation loads across multiple output
rows, improving arithmetic intensity. For 2048x5120 with NR=2, each thread loads x once and
applies it to 2 weight rows — effectively doubling compute per memory load.

### 9. Two-phase split-K reduction for the hottest dense kernel
**Journal evidence**: "This mirrors the MSL lane-parallel idea using pure DSL: increase parallel
work-items over the reduction dimension (via chunk index g) instead of one long serial loop/thread"
(from `custom_q4_0_linear_split` docstring). The split variant was prototyped but never benchmarked
end-to-end.
**What to do**: For the 1x2048x5120 hotspot (160 Q4 blocks), split reduction into G=8 or G=16
chunks. Stage 1: G independent partial dot-products (parallelizable as separate workgroups).
Stage 2: sum G partials (cheap). This trades one slow serial kernel for G fast parallel kernels
+ one tiny reduce. The key is that stage 1 kernels can saturate memory bandwidth independently.
**Prerequisite**: The split kernel needs proper `opts_to_apply` tuning and the stage-2 sum must
JIT-fuse cleanly (not add ICB overhead that negates the gain — journal warns "micro faster, full
pipeline slower" is common, line 615).

### 10. MoE expert kernel fusion: merge gate_up + silu*mul + down into fewer launches
**Journal evidence**: Feed-forward steady-state uses 15 kernels per block (line 80-81). The hot
sequence is: gather_gate_up → dequant+matmul → silu*mul → gather_down → dequant+matmul → combine.
Currently gather+dequant+matmul is one kernel (good), but silu*mul is a separate elementwise
launch, and the combine reduction is another.
**What to do**: (a) Fuse silu*mul into the gate_up reduce kernel epilogue. The gate_up custom
kernel stores `half` output; if it instead computed `silu(gate) * up` before storing, that
eliminates one elementwise kernel per block (46 launches/token). (b) Fuse the combine
`(expert_out * probs).sum(axis=2)` into the down kernel epilogue — write the weighted-and-summed
output directly. Both require extending the custom kernel DSL to support multi-operation epilogues,
which is non-trivial but would remove 2 kernels per block = 92 fewer launches per token.
