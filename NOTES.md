# TransBench Research Directive: Niche-Optimized Sequence Modeling

## Executive Summary
This document outlines the strategic pivot for the TransBench benchmarking project. Having validated that standard Mutual Information Gating (MIG) independently converges on industry-standard Mixture-of-Depths (MoD) and Residual Linear Attention (RLA) paradigms, we are shifting focus away from general-purpose sequence modeling. 

We will no longer attempt to beat models like Mamba2 or MiniMax on continuous, generalized language tasks. Instead, we are pivoting to **Architectural Inductive Biases**—designing highly specialized Transformer variants engineered to mathematically outperform general state-of-the-art models in two specific, adversarial environments: **Ultra-High Noise Streams** and **Deterministic Logical Branching**.

---

## The Strategic Pivot: Exploiting Generalist Vulnerabilities

General models (SSMs, standard Transformers, Lightning Attention) are designed as "Jacks of all trades." This generalism introduces fundamental mathematical vulnerabilities:
1. **Context Dilution:** Recurrent models (like Mamba) must update their hidden state with every token. In environments with 99% noise, the state becomes irreparably diluted.
2. **Latent Blurring:** Softmax-based attention is a continuous function. When faced with mutually exclusive logical rules, standard models blend them, resulting in hallucinations.

Our research will exploit these vulnerabilities using two specialized architectures.

---

## Track 1: Asymmetric Mutual Information Gating (A-MIG)
**Codename:** *The Skimmer*
**Target Domain:** Edge AI, IoT Telemetry, Financial Tick Data, Cyber Log Parsing.

### 1. Architectural Concept
Unlike standard MoD which distributes sparsity evenly, A-MIG acts as an aggressive filter. The early layers are structurally forbidden from deep reasoning; their singular objective is to classify and drop noise. By physically removing redundant tokens early in the residual stream, the deeper reasoning layers are protected from context dilution.

### 2. Structural Design
* **Layers 0-2 (The Skimmers):** Hard Top-K routing with extreme sparsity ($K \approx 0.05$). Drops 95% of the input sequence.
* **Layers 3-7 (The Deep Reasoners):** Standard Dense Attention, operating *only* on the highly concentrated 5% signal that survived the skimmers.

### 3. TransBench Implementation Protocol
* Modify `mig_module.py` to accept a `layer_keep_ratios` list rather than a static float. 
* E.g., `keep_ratios = [0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 1.0, 1.0]`.

### 4. Custom Benchmark: The "Poisoned Needle"
Standard `TinyStories` will not prove A-MIG's value. We must construct an adversarial dataset.
* **Dataset Generation:** Take standard `TinyStories` sequences and dynamically inject 80-90% random string literals or repetitive noise blocks in the center of the context window.
* **Success Metric:** Mamba2's validation loss should spike (hidden state corruption), while A-MIG maintains a loss $< 3.0$ by successfully gating the poison in Layer 0.

---

## Track 2: Stochastic Induction Layer (SIL)
**Codename:** *The Logic Switch*
**Target Domain:** Theorem Proving, Legal Tech, Deterministic Code Execution.

### 1. Architectural Concept
SIL Abandons continuous attention in favor of discrete rule-selection. By introducing a Gumbel-Softmax bottleneck, the model is forced to make a hard, discrete choice between $K$ latent logical rules before generating output. This prevents "concept blurring" and provides structural explainability.

### 2. Structural Design
* **The Deterministic Path:** Standard Grouped Query Attention (GQA).
* **The Probabilistic Path:** A parallel branch where the input $x_t$ is projected into $K$ rule logits. 
* **The Gumbel Bottleneck:**
  $$z = \text{Softmax}\left(\frac{l + g}{\tau}\right)$$
  During the forward pass with `hard=True`, this yields a pure one-hot vector, forcing the architecture to commit to a single discrete rule embedding to add back into the residual stream.

### 3. TransBench Implementation Protocol
* Isolate `sil_module.py` and ensure the Gumbel temperature $\tau$ is strictly annealed during the training loop. High $\tau$ initially for exploration, decaying to $< 0.1$ to force strict discretization.

### 4. Custom Benchmark: "Bifurcated Grammar"
* **Dataset Generation:** Create a synthetic task containing two mutually exclusive rules. 
  * *Rule A:* "Reverse the input sequence."
  * *Rule B:* "Sort the input sequence alphabetically."
* **Success Metric:** Measure the "Mode Collapse/Hallucination Rate." Standard Transformers will frequently output partially sorted, partially reversed strings. A successful SIL architecture must demonstrate $0\%$ rule-mixing due to its discrete latent bottleneck.

---

## Immediate Next Steps for the Engineering Agent

1. ~~**Halt General Benchmarking:** Suspend standard `TinyStories` runs against Mamba2.~~ ✅ Done
2. ~~**Select the Primary Track:** Choose **A-MIG** (Track 1).~~ ✅ Done — A-MIG selected
3. ~~**Draft the Adversarial Dataset:** Update `src/transbench/datasets.py` to generate the *Poisoned Needle* dataset.~~ ✅ Done — `poisoned_needle` dataset implemented with configurable `poison_ratio`
4. ~~**Deploy the Niche Architecture:** Wire A-MIG through the full stack.~~ ✅ Done — `mig_module.py` (Top-K + per-layer keep ratios), `benchmark.py`, `cli.py`, `suite.py`, `benchmarks/benchmarks.amig.toml`

### Implementation Summary (A-MIG — Track 1)
- **`mig_module.py`**: Added `layer_keep_ratios` param, `_effective_keep_ratio()`, `_apply_topk_mask()` with hard Top-K selection via `torch.topk` + scatter mask
- **`datasets.py`**: Added `poisoned_needle` dataset — injects 80-90% random noise in center tokens, preserving edge "needles"
- **`cli.py`**: Added `--mig-layer-keep-ratios` (comma-separated) and `--poison-ratio` arguments
- **`benchmark.py`**: Added `mig_layer_keep_ratios` and `poison_ratio` to `BenchmarkConfig`; wired through `_build_model()` and `make_sampler()`
- **`suite.py`**: Added TOML support for `mig_layer_keep_ratios` and `poison_ratio`
- **`benchmarks/benchmarks.amig.toml`**: 3-way comparison: A-MIG Skimmer vs standard MIG vs GQA on Poisoned Needle

### Remaining Work
- **Track 2 (SIL):** Temperature annealing in training loop + Bifurcated Grammar dataset
- **Validation:** Full benchmark run of `benchmarks.amig.toml` to confirm A-MIG's loss advantage

This directive transitions the project from trying to build a better engine for *everything*, to building an insurmountable engine for *specific, high-value edge cases*.