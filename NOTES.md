Okay, if you want to keep both tracks alive in your research (the Skimmer for noise resiliency and the Logic Switch for deterministic reasoning), we need to establish a highly rigorous, split-track benchmarking protocol.

This is the comprehensive ArchiFactory Research Guideline. It defines exactly how to build the adversarial datasets, how to configure the architecture variants, and how to execute the benchmarks to prove your specific architectural advantages.

---

# ArchiFactory Research Protocol: Niche-Optimized Sequence Modeling

**Objective:** To demonstrate that specialized Transformer architectures (A-MIG and SIL) mathematically outperform general sequence models (Mamba2) in specifically designed adversarial environments: High-Noise Streams and Deterministic Logical Branching.

---

## Track 1: The Skimmer (A-MIG) vs. Context Dilution

**Hypothesis:** In environments with $>80\%$ sequence noise, continuous recurrent models (Mamba2) suffer from context dilution and catastrophic forgetting. An Asymmetric Mutual Information Gated (A-MIG) architecture will maintain high accuracy by physically dropping noise in early layers, shielding the reasoning capacity of deeper layers.

### Phase 1: Architecture Configuration (A-MIG)
The architecture must be configured as a "funnel."

1.  **Modify `mig_module.py`:** Ensure the module can accept a list or schedule of keep ratios per layer, rather than a global constant.
2.  **Configuration (8-Layer Model):**
    * **Layer 0 (Extreme Filter):** `keep_ratio = 0.1` (Drops 90% of tokens).
    * **Layer 1 (Secondary Filter):** `keep_ratio = 0.5` (Drops 50% of remaining tokens).
    * **Layers 2-7 (Deep Reasoners):** `keep_ratio = 1.0` (Standard Dense Attention).
3.  **Mechanism:** The early layers must use a Load-Balancing / Z-Loss to quickly learn which tokens to drop without destroying the gradient flow.

### Phase 2: Adversarial Dataset Generation (The "Poisoned Needle")
We will weaponize the `TinyStories` dataset.

1.  **The Base:** Load a standard `TinyStories` sample (e.g., 200 tokens).
2.  **The Split:** Identify the midpoint of the sequence (token index 100).
3.  **The Poison:** Inject 800 tokens of high-entropy "poison."
    * *Type A (Static Poison):* A repeating string of irrelevant text (e.g., "The system is functioning normally. No errors detected.")
    * *Type B (High Entropy Poison):* Randomly sampled tokens from the vocabulary.
4.  **The Task:** The model must read the first half of the story, survive the 800-token poison block, and correctly generate the ending of the story.

### Phase 3: The Benchmark Protocol
1.  **Baseline:** Train Mamba2 on the "Poisoned Needle" dataset for 2,000 steps.
2.  **Challenger:** Train A-MIG on the "Poisoned Needle" dataset for 2,000 steps.
3.  **Success Criteria:**
    * **Validation Loss:** A-MIG must achieve a significantly lower validation loss ($< 3.5$) than Mamba2.
    * **Sparsity Metric:** Verify that A-MIG's Layer 0 gate is successfully assigning low scores ($< 0.1$) to the "poison" tokens and high scores to the "story" tokens.

---

## Track 2: The Logic Switch (SIL) vs. Latent Blurring

**Hypothesis:** When faced with mutually exclusive logical operations within the same context, continuous models (Mamba2, standard Transformers) suffer from "mode collapse" (blending the rules). The Stochastic Induction Layer (SIL) will prevent this hallucination by forcing a discrete latent bottleneck via Gumbel-Softmax.

### Phase 1: Architecture Configuration (SIL)
The architecture must feature a discrete bottleneck.

1.  **Modify `sil_module.py`:** Ensure the Gumbel-Softmax sampling is correctly implemented and differentiable.
2.  **Configuration (8-Layer Model):**
    * **Layers 0-3:** Standard Dense Attention (Context Gathering).
    * **Layer 4 (The Logic Bottleneck):** Replace standard attention with the SIL module. Set $K=4$ (number of latent rules).
    * **Layers 5-7:** Standard Dense Attention (Decoding).
3.  **Temperature Annealing:** The training loop must decay the Gumbel temperature ($\tau$) from $1.0$ to $0.1$ over the first 500 steps to force hard discretization.

### Phase 2: Adversarial Dataset Generation (Bifurcated Algorithmic)
We need a synthetic dataset that demands strict rule adherence.

1.  **Task A (Reverse):**
    * *Prompt:* `[TASK_REV] seq: A B C D -> ans:`
    * *Target:* `D C B A`
2.  **Task B (Sort):**
    * *Prompt:* `[TASK_SRT] seq: D A C B -> ans:`
    * *Target:* `A B C D`
3.  **The Dataset:** Generate 100,000 random sequences (length 4-8). Label 50% with `[TASK_REV]` and 50% with `[TASK_SRT]`.

### Phase 3: The Benchmark Protocol
1.  **Baseline:** Train a standard small Transformer (or Mamba2) on the Bifurcated dataset.
2.  **Challenger:** Train the SIL model on the Bifurcated dataset.
3.  **Evaluation Metric: "Hallucination Rate"**
    * During validation, check for *rule blending*. If the prompt is `[TASK_REV] c a b` and the output is `b c a` (partially sorted, partially reversed), log this as a hallucination.
4.  **Success Criteria:**
    * The SIL architecture must demonstrate a $0\%$ (or near-zero) rule-blending hallucination rate due to its discrete bottleneck, outperforming the continuous baseline.

---

## Implementation Roadmap for the Engineering Agent

To execute this research, instruct your coding agent to perform the following steps sequentially:

1.  **Dataset Construction:**
    * Write `scripts/generate_poisoned_tinystories.py` (Track 1).
    * Write `scripts/generate_bifurcated_algo.py` (Track 2).
2.  **Architecture Validation:**
    * Update `mig_module.py` to support `layer_keep_ratios`.
    * Update `sil_module.py` to ensure temperature annealing is connected to the training loop.
3.  **Benchmarking Execution:**
    * Run Track 1 (A-MIG vs. Mamba2) on the poisoned dataset. Collect loss and speed metrics.
    * Run Track 2 (SIL vs. Standard Transformer) on the bifurcated dataset. Collect the hallucination rate metric.