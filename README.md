# KV Cache Optimization for Efficient LLM Serving

**Course:** CS595 Efficient Machine Learning Systems  
**Team:** Devarsh Kale & Dhruv Arora  
**Project Type:** Engineering / Measurement

---

## Overview

This project investigates how KV (Key-Value) cache growth affects LLM inference efficiency — specifically prefill latency, decode latency per token, and VRAM usage — across different model scales and hardware. We implement and evaluate a StreamingLLM-style KV eviction strategy (sink tokens + sliding window) and compare it against a full-cache baseline and a naive context truncation baseline.

**Key findings:**
- Prefill latency scales sharply with context length: +577% from 512→4096 tokens on Mistral-7B (A100)
- Decode latency is hardware-dependent: +26% on RTX 2060 (consumer GPU), flat on A100 (server GPU)
- VRAM grows +856 MB from 512→4096 tokens on Mistral-7B
- KV eviction preserves quality at budget ≥ 1024 tokens; sharp quality cliff below that threshold
- Eviction overhead on A100 adds +8–11% latency due to Python-level cache slicing

---

## Repository Structure

```
├── KV_Cache_Mistral.ipynb        # Primary notebook — Mistral-7B on Colab A100
├── KV_Cache_TinyLlama.ipynb      # Comparison notebook — TinyLlama-1.1B on RTX 2060
├── results/                      # CSV summaries from all experiments
│   ├── baseline_summary.csv
│   ├── baseline_results.csv
│   ├── context_compare_summary.csv
│   ├── context_compare_results.csv
│   ├── window_ablation_summary.csv
│   ├── window_ablation_results.csv
│   ├── actual_kv_optimization_summary.csv
│   ├── actual_kv_optimization_results.csv
│   ├── perplexity_results.csv
│   ├── tradeoff_table.csv
│   ├── bottleneck_table.csv
│   ├── realistic_benchmark_summary.csv
│   └── realistic_quality_outputs.csv
├── plots/                        # All generated figures
│   ├── latency_vs_prompt_length.png
│   ├── latency_per_token_vs_prompt_length.png
│   ├── vram_vs_prompt_length.png
│   ├── full_vs_reduced_context_latency.png
│   ├── full_vs_reduced_context_vram.png
│   ├── actual_kv_comparison.png
│   ├── pareto_tradeoff.png
│   ├── hardware_comparison.png
│   ├── window_size_vs_latency_per_token.png
│   └── window_size_vs_vram.png
└── README.md
```

---

## Hardware & Software Requirements

### Primary Experiments (Mistral-7B)
- **GPU:** NVIDIA A100 40GB (Google Colab)
- **VRAM required:** ~15 GB minimum
- **Recommended:** Google Colab with A100 runtime

### Comparison Experiments (TinyLlama)
- **GPU:** NVIDIA RTX 2060 6GB (or any CUDA-capable GPU with ≥ 6 GB VRAM)
- Can also be run on Colab T4 or L4

### Software Stack
- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers 4.36+
- CUDA 11.8+

---

## Setup Instructions

### Option A — Google Colab (Recommended for Mistral-7B)

1. Open `KV_Cache_Mistral.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → A100 GPU**
3. Run the first cell — it installs all dependencies automatically:
   ```python
   !pip install transformers torch accelerate pandas matplotlib
   ```
4. Run all cells top to bottom
5. At the end, the notebook downloads `kv_project_outputs.zip` containing all results and plots

### Option B — Local Setup (TinyLlama / smaller GPU)

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install torch transformers accelerate pandas matplotlib
   ```
3. Open `KV_Cache_TinyLlama.ipynb` in Jupyter:
   ```bash
   jupyter notebook KV_Cache_TinyLlama.ipynb
   ```
4. Run all cells top to bottom

---

## Experiment Description

Both notebooks follow the same experimental pipeline:

### 1. Baseline Prompt-Length Sweep
Runs the model across prompt lengths of 512, 1024, 2048, and 4096 tokens (3 runs per setting after warmup). Measures TTFT, prefill time, decode time per token, total latency, and peak VRAM.

### 2. Naive Context Reduction Baseline
Compares full 1024-token prompt vs. truncated 512-token prompt. Used to show that simply feeding less input is a weak baseline — it discards context before the model can use it and provides only marginal gains.

### 3. Window Size Ablation
Sweeps different input prompt lengths (512, 768, 1024 tokens) to isolate the effect of context length on latency and VRAM.

### 4. StreamingLLM-Style KV Eviction
The core optimization. Applied at a 4096-token prompt across four cache budget modes:

| Mode | Description |
|---|---|
| `full_cache` | No eviction — baseline |
| `kv_budget_1024` | Keep first 4 sink tokens + most recent 1020 tokens |
| `kv_budget_512` | Keep first 4 sink tokens + most recent 508 tokens |
| `kv_budget_256` | Keep first 4 sink tokens + most recent 252 tokens |

Eviction is applied to `past_key_values` at every decode step.

### 5. Perplexity Evaluation
Generates output under each KV mode, then scores the continuation using the full model in teacher-forced mode. Lower perplexity = better quality preservation.

### 6. Hardware Comparison Plot
Combines Mistral-7B A100 results with hardcoded TinyLlama RTX 2060 baseline numbers to produce a side-by-side comparison of decode latency and VRAM scaling across hardware classes.

---

## Key Custom Functions

The HuggingFace `DynamicCache` API does not natively support manual truncation during decoding. Three custom helper functions were written to enable this:

- **`cache_to_legacy_tuple(past_key_values)`** — converts HuggingFace DynamicCache to a standard tuple of (key, value) tensors per layer
- **`legacy_tuple_to_dynamic_cache(legacy_cache)`** — converts back to DynamicCache format after truncation
- **`truncate_past_key_values(past_key_values, kv_budget, sink_size=4)`** — implements StreamingLLM-style eviction: always keeps the first `sink_size` tokens (attention sinks) and the most recent `kv_budget - sink_size` tokens, evicts everything in between

> **Why sink tokens matter:** Without retaining the first few tokens, small models produce degenerate (repetitive or incoherent) output. Keeping 4 sink tokens restores stable generation. This was validated empirically during development.

---

## Expected Outputs

After running the Mistral notebook end-to-end, the `results/` directory will contain:

| File | Contents |
|---|---|
| `baseline_summary.csv` | Avg latency, VRAM per prompt length |
| `actual_kv_optimization_summary.csv` | Eviction results across 4 budget modes |
| `tradeoff_table.csv` | Latency %, VRAM %, perplexity % vs baseline |
| `perplexity_results.csv` | Perplexity per mode per prompt length |
| `bottleneck_table.csv` | Unified summary across all experiments |

And the `plots/` directory will contain all figures used in the report and presentation.

---

## Reproducing the Hardware Comparison

The RTX 2060 baseline numbers are hardcoded in the hardware comparison cell of `KV_Cache_Mistral.ipynb` (sourced from running `KV_Cache_TinyLlama.ipynb` locally). To fully reproduce the hardware comparison from scratch:

1. Run `KV_Cache_TinyLlama.ipynb` on a local RTX 2060 (or equivalent consumer GPU)
2. Copy the `baseline_summary.csv` output
3. Replace the hardcoded values in the hardware comparison cell of `KV_Cache_Mistral.ipynb` with your own measurements

---

## Notes on Reproducibility

- Absolute timing values will vary across GPU instances even within the same hardware class (e.g., different Colab A100 allocations). The relative comparisons and scaling trends are stable.
- The first run of each experiment setting may be slightly slower due to CUDA JIT compilation. Warmup passes are included in both notebooks to eliminate this artifact from reported measurements.
- Mistral-7B requires a Hugging Face account. The model (`mistralai/Mistral-7B-Instruct-v0.2`) is publicly available and downloads automatically on first run.

---

## Results Summary

### Baseline (Mistral-7B, A100)

| Prompt Tokens | Prefill (ms) | Decode (ms/tok) | Peak VRAM (MB) |
|---|---|---|---|
| 512 | 48 | 35.99 | 13,943 |
| 1024 | 86 | 36.19 | 14,065 |
| 2048 | 161 | 36.30 | 14,310 |
| 4096 | 325 | 35.39 | 14,799 |

### KV Eviction (Mistral-7B, A100, 4096-token prompt)

| Mode | Decode ms/tok | Perplexity | Latency Δ | VRAM Δ |
|---|---|---|---|---|
| full_cache | 35.26 | 1.68 | — | — |
| kv_budget_1024 | 38.18 | 1.96 | +8.3% | -0.6% |
| kv_budget_512 | 38.84 | 2.97 | +10.2% | -1.0% |
| kv_budget_256 | 39.11 | 2.97 | +10.9% | -1.2% |

### Hardware Comparison (Decode Latency per Token)

| GPU | 512 tokens | 1024 tokens | Change |
|---|---|---|---|
| RTX 2060 (TinyLlama) | 50.5 ms/tok | 66.3 ms/tok | +26% |
| A100 (Mistral-7B) | 35.99 ms/tok | 36.19 ms/tok | ~0% |

---

## References

1. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453, 2023
2. Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023
3. Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385, 2024
4. Hugging Face Transformers Documentation — https://huggingface.co/docs/transformers
5. PyTorch Documentation — https://pytorch.org/docs
