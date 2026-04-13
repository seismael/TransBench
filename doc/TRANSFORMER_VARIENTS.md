
# **Architecture Technical Specification**

**Version:** 2.0 (Experimental)
**Target Framework:** PyTorch / TransBench

---

## **1. Mutual Information Gated Transformer (MIG-Transformer)**
**Codename:** *The Skimmer*
**Design Philosophy:** Extreme Context Protection via Asymmetric Sparsity

### **1.1 The Flaw in Standard Architectures**

Standard sequence models operate on an assumption of **Information Density Equivalency** — they mathematically assume every token in the sequence has the potential to be important.

* **In Transformers (GQA):** The attention matrix computes the similarity of every token against every other token ($O(N^2)$), regardless of whether a token is a critical noun or a repetitive stop-word.
* **In State-Space Models (Mamba2):** The recurrent state is updated sequentially. If a model ingests 10,000 tokens of noise, the continuous hidden state is mathematically diluted by 10,000 minor updates, washing out the signal of the few critical tokens.

### **1.2 Executive Summary**

The **MIG-Transformer** addresses this inefficiency by introducing a learnable "Information Gate" prior to the attention block. The model dynamically filters out tokens with low predicted utility (e.g., redundant determiners, repetitive patterns) before the computationally expensive attention operation.

### **1.3 Core Architectural Components**

#### **A. The Information Gate (IG)**

A lightweight, pointwise Multi-Layer Perceptron (MLP) acts as a "bouncer" for the attention layer.

* **Input:** Token hidden state .
* **Operation:** Projects the high-dimensional state to a scalar "importance score" .
* **Activation:** Sigmoid, to ensure a smooth probability-like gating factor.

#### **B. Gated Attention Mechanism**

Unlike standard attention where , MIG modifies the input stream itself.

* The query/key/value projections effectively receive scaled inputs: .
* If , the token contributes a zero-vector to the attention weighted sum, effectively removing it from the context window of other tokens.

### **1.4 Mathematical Formulation**

Let  be the hidden state of token .

1. **Gate Calculation:**


2. **Gated Attention:**


3. **Sparsity Objective (Loss Function):**
To prevent the gate from staying open () permanently, we add an L1 penalty to the loss function:



### **1.5 Data Flow Diagram**

1. **Input Tensor** `[Batch, Seq, Hidden]`
2. **Branch A:** Compute Main Attention (Standard Path).
3. **Branch B:** Compute Gate Score (MLP -> Sigmoid).
4. **Fusion:** Element-wise multiplication of Input × Gate Score.
5. **Attention:** Run standard GQA on fused input.
6. **Residual:** Add output to original stream.

### **1.6 Asymmetric MIG (A-MIG) — Per-Layer Sparsity**

A-MIG extends MIG by applying **different keep ratios per layer**, creating an asymmetric funnel:

| Layer Position | Role | Keep Ratio | Purpose |
|:---:|:---:|:---:|:---|
| Layers 0–2 | **Skimmer** | 0.05 (5%) | High-speed noise classification; physically drops 95% of tokens |
| Layers 3–5 | **Transition** | 0.10–0.50 | Gradual widening as signal concentration increases |
| Layers 6+ | **Reasoner** | 1.00 (100%) | Dense attention on concentrated signal only |

**Configuration:** `mig_layer_keep_ratios = [0.05, 0.05, 0.10, 0.50, 1.0, 1.0]`

Unlike standard Mixture-of-Depths which distributes sparsity evenly, A-MIG structurally forbids early layers from deep reasoning — their sole purpose is noise classification and removal.

### **1.7 Target Domains & Commercial Use Cases**

MIG/A-MIG is engineered for **High-Noise, Low-Signal Streams** — contexts where the window is large but information density is near zero.

* **Cybersecurity & Threat Detection:** 99.9% of server log lines are routine status checks. A-MIG skims past routine lines, passing only anomalous payload tokens to deep reasoning layers to detect novel intrusion vectors.
* **Genomic Sequencing (Bioinformatics):** Large portions of the human genome consist of non-coding "junk" DNA. A-MIG gates out repetitive non-coding sequences, focusing capacity on exons and regulatory regions.
* **High-Frequency Trading (HFT):** Thousands of micro-transactions occur before a meaningful trend forms. A-MIG drops the noise to focus on aggregate momentum shifts.

---

## **2. Stochastic Induction Transformer (SIL-Transformer)**
**Codename:** *The Logic Switch*
**Design Philosophy:** Anti-Hallucination via Discrete Latent Bottlenecks

### **2.1 The Flaw in Standard Architectures**

Standard Transformers utilize the Softmax function to distribute attention. Because Softmax is a continuous, differentiable function, it almost never assigns a probability of exactly `1.0` or `0.0`.

When a standard model encounters a prompt requiring a strict logical choice between two mutually exclusive rules (e.g., "Sort this list" vs. "Reverse this list"), the continuous attention mechanism "blurs" the concepts. It activates the "Sort" circuit at 60% and the "Reverse" circuit at 40%, resulting in an output that is a hallucinated hybrid of both rules. This is known as **Latent Blurring**.

### **2.2 Executive Summary**

The **SIL-Transformer** introduces a structural safeguard against blurring. It adds a discrete latent variable that forces the model to categorize the current context into one of $K$ distinct "rules" or "hypotheses." At a critical juncture, the continuous residual stream is forced through a discrete bottleneck using the Gumbel-Softmax estimator. This improves systematic generalization by disentangling *rule selection* from *token matching*.

### **2.3 Core Architectural Components**

#### **A. The Rule Encoder (Hypothesis Generator)**

A projection layer that maps the current context to a set of logits representing possible logical rules.

* **Input:** Hidden State .
* **Output:** Logits for  latent rules.

#### **B. Gumbel-Softmax Sampler**

A differentiable sampling mechanism that allows backpropagation through discrete choices.

* **Training:** Soft approximation (allows gradients to flow).
* **Inference:** Hard selection (picks exactly one rule).
* **Temperature ():** Controls the "confidence" of the sampling. High  = random exploration; Low  = greedy selection.

#### **C. The Induction Embedding**

A learnable embedding matrix  representing the vector form of each abstract rule.

### **2.4 Mathematical Formulation**

Let  be the number of latent rules.

1. **Rule Logits:**


2. **Stochastic Sampling (Gumbel-Max):**



*Where  are i.i.d samples drawn from Gumbel(0, 1).*
3. **Rule Injection:**



### **2.5 Data Flow Diagram**

1. **Input Tensor** `[Batch, Seq, Hidden]`
2. **Path 1 (Deterministic):** Standard Multi-Head Attention.
3. **Path 2 (Stochastic):**
* Linear Projection to  logits.
* **Sample**  via Gumbel-Softmax.
* **Decode**  back to hidden dimension.


4. **Aggregation:** Add Path 1 Output + Path 2 Output.

### **2.6 Target Domains & Commercial Use Cases**

SIL is engineered for **High-Determinism, Rule-Bound Environments** — where hallucination is not just annoying but legally, medically, or mathematically fatal.

* **Formal Theorem Proving:** Mathematical transformations are discrete (e.g., applying the transitive property). You cannot "partially" apply a mathematical law. SIL forces strict adherence to logical operations.
* **Legal Contract Analysis:** Legal logic operates on strict IF/THEN branching. SIL ensures the model commits to a single interpretation of a clause rather than generating a hallucinated compromise between two conflicting legal precedents.
* **Deterministic Code Execution / RPA:** Agentic workflows where the model must decide which API tool to call. SIL acts as a definitive "Router," ensuring the model commits to `Tool_A` rather than hallucinating a hybrid function call.

---

## **3. Attentional Symmetry Regularized Transformer (ASR-Transformer)**
**Codename:** *The Stabilizer*
**Design Philosophy:** Spatial Robustness via Geometric Consistency

### **3.1 The Flaw in Standard Architectures**

Standard sequence models are highly susceptible to **Spatial Overfitting and Prompt Fragility**.

Because standard attention mechanisms (GQA) rely purely on dot-product similarity, they tend to overfit to the exact position and sequencing of the training data. If you prompt a standard model with a question, and then prompt it with the *exact same question* but include a minor typo, a synonym swap, or shuffle the order of independent bullet points, the internal attention matrix shifts drastically. This fragility causes the model to output a completely different (often hallucinated) answer simply because the "spatial coordinates" of the prompt changed.

Furthermore, they struggle with **Out-of-Distribution (OOD) length generalization** — training on 128 tokens but failing catastrophically when inferencing on 512 tokens.

### **3.2 Executive Summary**

The **ASR-Transformer** does not change the inference architecture (it looks like a standard Llama/Mistral model at runtime). Instead, it modifies the **training paradigm** by enforcing a geometric constraint: the model's internal attention map must remain stable (invariant) even when the input is subjected to semantic noise. This forces the model to learn robust, generalized features rather than memorizing exact token positions or noise patterns.

### **3.3 Core Architectural Components**

#### **A. The Siamese Forward Pass**

During training, the attention block is executed twice for every batch:

1. **Clean Pass:** 
2. **Noised Pass:** , where .

#### **B. The Consistency Regulator**

An auxiliary loss term that penalizes the divergence between the Clean and Noised outputs. This effectively smooths the loss landscape, creating "wide valleys" of low loss (better generalization) rather than "sharp minima" (overfitting).

### **3.4 Mathematical Formulation**

1. **Perturbation:**


2. **Consistency Loss (Cosine Similarity approach):**



*Note: The target  is detached from the computation graph to prevent the model from collapsing to zero.*
3. **Total Objective:**



### **3.5 Data Flow Diagram**

1. **Input Tensor** `[Batch, Seq, Hidden]`
2. **Step 1:** Calculate Standard Attention Output .
3. **Step 2 (Training Only):**
* Generate noise tensor .
* Calculate .


4. **Step 3:** Compute consistency loss between  and .
5. **Output:** Return  for the next layer.

### **3.6 Target Domains & Commercial Use Cases**

ASR is engineered for **High-Variance, Unstructured Input Environments** — where inputs are messy, unpredictable, or structurally flawed but the model must maintain strict factual accuracy.

* **OCR / Document AI Pipelines:** OCR engines frequently introduce spelling errors, artifacts, and noise. ASR models are inherently trained to resist this exact type of $\epsilon$-perturbation, maintaining attention stability where standard models fail.
* **Customer-Facing Chatbots & Copilots:** End-users use slang, make typos, and formulate fragmented sentences. ASR provides "Prompt Robustness," ensuring the LLM does not derail its reasoning chain due to poor syntax.
* **Long-Context Information Retrieval:** When extending the context window during inference beyond the trained sequence length, ASR's smoothed attention landscape prevents the catastrophic "attention collapse" that standard GQA suffers from, allowing for safer zero-shot length extrapolation.

---

### **Comparison & Benchmark Strategy**

| Feature | MIG / A-MIG | SIL | ASR |
| --- | --- | --- | --- |
| **Codename** | *The Skimmer* | *The Logic Switch* | *The Stabilizer* |
| **Vulnerability Fixed** | Context Dilution | Latent Blurring | Spatial Fragility |
| **Primary Goal** | Efficiency (FLOPs reduction) | Reasoning (Rule adherence) | Robustness (Generalization) |
| **New Parameters** | Minimal (Gate MLP) | Moderate (Rule Embeddings) | None (Training only) |
| **Training Complexity** | Medium (Requires sparsity tuning) | High (Unstable gradients) | Low (Just add loss) |
| **Inference Cost** | **Lower** (skip gated-out tokens) | Slightly Higher (Logic path) | **Identical** to Baseline |
| **Target Domain** | Cybersecurity, Genomics, HFT | Theorem Proving, Legal Tech, RPA | OCR, Chatbots, Long-Context |

### The Trifecta

These three architectures form a complete defensive toolkit against the failure modes of generalist models:

1. **A-MIG** fixes *Context Dilution* — for massive noise streams.
2. **SIL** fixes *Latent Blurring* — for strict logical branching.
3. **ASR** fixes *Spatial Fragility* — for messy, unstructured inputs.

**Recommended Testing Order:**

1. **ASR:** Easiest to implement, high probability of improving your baseline `TinyStories` loss.
2. **MIG:** Good for analyzing which tokens the model thinks are "important."
3. **SIL:** Most experimental; save for last as it requires careful hyperparameter tuning (temperature, number of rules).