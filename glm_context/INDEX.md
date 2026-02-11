# GLM LLM Optimization Documentation Index

**Last updated**: Feb 10, 2026
**Current performance**: GLM-4.7-Flash Q4_0 = **20.0 tok/s** (pure tinygrad DSL, no custom MSL)
**Target**: 35 tok/s (llama.cpp parity)

---

## Quick Start

**New to GLM optimization?** Start with **[START.md](START.md)** for overview and current status.

**Looking for specific information?** Use the sections below.

---

## Core Documentation

### Getting Started
- **[START.md](START.md)** - Overview, current performance, key findings, action plan
- **[tools.md](tools.md)** - Commands, profiling workflows, benchmarking rules, chat templates

### Understanding the Architecture
- **[architecture.md](architecture.md)** - MLA formulations, MoE routing, quantization formats
- **[llama_cpp.md](llama_cpp.md)** - Reference implementation patterns

### Performance Analysis
- **[bottlenecks.md](bottlenecks.md)** - 3 performance gaps, per-model budgets, heuristic failures
- **[performance.md](performance.md)** - Cross-model benchmarks and theoretical limits
- **[kernel_analysis.md](kernel_analysis.md)** - **NEW**: Per-kernel breakdown of GLM Q4_0, detailed bottleneck analysis

---

## Optimization Work

### Experiment Log
- **[experiments.md](experiments.md)** - **NEW**: Detailed log of all optimization attempts with results
  - What worked: merge_gate_up_experts (14→20 tok/s, +43%)
  - What failed: removing .contiguous() (-17% regression), merging RoPE (0% gain)
  - Key learnings: when manual merging helps vs when scheduler already fuses

### Optimization Ideas
- **[ideas.md](ideas.md)** - Code simplification and performance ideas (60+ items)
- **[advanced_ideas.md](advanced_ideas.md)** - **NEW**: Next-gen optimizations from recent PRs
  - E-graph for MoE (expected 2-3x speedup)
  - PARAM normalization for cache hits (1.5-2x speedup)
  - Atomic scatter for expert aggregation (1.3-1.5x speedup)
  - Combined potential: **40-60 tok/s**

### Historical Context
- **[optimization_ideas.md](optimization_ideas.md)** - Earlier optimization brainstorming
- **[optimization_ideas2.md](optimization_ideas2.md)** - Additional optimization thoughts
- **[opus_ideas.md](opus_ideas.md)** - Claude Opus analysis
- **[next.md](next.md)** - Next steps and priorities

---

## Specialized Topics

### Kernel-Level Analysis
- **[glm_hot_kernels.md](glm_hot_kernels.md)** - Analysis of hotspot kernels
- **[scheduler_fusion.md](scheduler_fusion.md)** - Scheduler fusion experiments and threadgroup memory fixes
- **[group_placeholder_fix.md](group_placeholder_fix.md)** - GROUP optimization placeholder handling

---

## Document Organization

### By Task

**I want to understand current performance:**
→ [START.md](START.md) → [kernel_analysis.md](kernel_analysis.md) → [bottlenecks.md](bottlenecks.md)

**I want to see what's been tried:**
→ [experiments.md](experiments.md) → [bottlenecks.md](bottlenecks.md) (complete experiment log)

**I want to know what to try next:**
→ [advanced_ideas.md](advanced_ideas.md) (E-graph, PARAM, atomics - highest priority)
→ [ideas.md](ideas.md) (60+ code/perf ideas)
→ [next.md](next.md) (immediate next steps)

**I want to understand the architecture:**
→ [architecture.md](architecture.md) → [llama_cpp.md](llama_cpp.md) (reference impl)

**I want to run benchmarks:**
→ [tools.md](tools.md) (profiling commands, testing protocols)

### By Role

**Researcher/Optimizer:**
1. [kernel_analysis.md](kernel_analysis.md) - Understand bottlenecks
2. [advanced_ideas.md](advanced_ideas.md) - Next-gen optimizations
3. [experiments.md](experiments.md) - Learn from past attempts

**Developer/Implementer:**
1. [architecture.md](architecture.md) - Understand the code
2. [ideas.md](ideas.md) - Code simplification opportunities
3. [tools.md](tools.md) - Testing and benchmarking

**Performance Engineer:**
1. [bottlenecks.md](bottlenecks.md) - Full gap analysis
2. [scheduler_fusion.md](scheduler_fusion.md) - Scheduler optimization
3. [llama_cpp.md](llama_cpp.md) - Reference patterns

---

## Key Metrics

### Current Status (GLM-4.7-Flash Q4_0)
- **Performance**: 20.0 tok/s (49ms/token)
- **Kernels**: 1358 kernels/token, 6 ICBs
- **Efficiency**: 25% of theoretical (80 tok/s bandwidth limit)

### Top Bottlenecks
1. **Shared expert gate+silu*up** (22%) - 370us, 34 GB/s
2. **Expert gather gate_up** (21%) - 356us, 160 GB/s (pure data copy waste)
3. **Expert Q4_0 matmul** (16%) - 278us, 51 GB/s
4. **MoE total**: 54% of per-token time

### Optimization Impact
- **merge_gate_up_experts**: 14 → 20 tok/s (+43%) ✅
- **E-graph (projected)**: 20 → 40-60 tok/s (2-3x)
- **PARAM normalization (projected)**: 1.5-2x speedup
- **Combined potential**: **Match or exceed llama.cpp's 35 tok/s**

---

## File Map

```
glm_context/
├── INDEX.md                    ← You are here
├── START.md                    ← Start here if new
│
├── Core/
│   ├── architecture.md         MLA, MoE, quantization
│   ├── bottlenecks.md          Performance gaps, experiments
│   ├── performance.md          Benchmark results
│   └── tools.md                Commands, workflows
│
├── Analysis/
│   ├── kernel_analysis.md      Per-kernel breakdown (NEW)
│   ├── experiments.md          Detailed experiment log (NEW)
│   └── glm_hot_kernels.md      Hotspot analysis
│
├── Ideas/
│   ├── advanced_ideas.md       E-graph, PARAM, atomics (NEW)
│   ├── ideas.md                60+ optimization ideas
│   ├── optimization_ideas.md   Earlier brainstorming
│   └── next.md                 Immediate priorities
│
└── Reference/
    ├── llama_cpp.md            Reference implementation
    ├── scheduler_fusion.md     Fusion experiments
    └── group_placeholder_fix.md GROUP optimization
```

---

## Archive

Original markdown files from root directory have been synthesized into glm_context and archived in `archive_md/`:
- experiment_plan.md → [experiments.md](experiments.md)
- glm_kernel_map.md → [kernel_analysis.md](kernel_analysis.md)
- glm_speedup_ideas.md → [advanced_ideas.md](advanced_ideas.md)
- optimization_log.md → [experiments.md](experiments.md)
- optimization_analysis.md → [bottlenecks.md](bottlenecks.md)
- optimization_hypotheses.md → [experiments.md](experiments.md)
- grouptop_results.md → [experiments.md](experiments.md)
- split_shexp_results.md → [experiments.md](experiments.md)

---

## Contributing

When adding new optimization results:
1. Update [experiments.md](experiments.md) with detailed log
2. Update [kernel_analysis.md](kernel_analysis.md) if kernel behavior changes
3. Update [START.md](START.md) current performance numbers
4. Add learnings to [advanced_ideas.md](advanced_ideas.md) or [ideas.md](ideas.md)

---

## Quick Reference

**Current branch**: glm9
**Model**: GLM-4.7-Flash Q4_0 (pure tinygrad DSL)
**Performance**: 20.0 tok/s
**Target**: 35 tok/s (llama.cpp parity)
**Next priority**: E-graph optimization for MoE (expected 2-3x speedup)
