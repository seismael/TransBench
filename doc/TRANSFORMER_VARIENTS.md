
# **Architecture Technical Specification**

**Version:** 1.0 (Experimental)
**Target Framework:** PyTorch / TransBench

## **1. Mutual Information Gated Transformer (MIG-Transformer)**

### **1.1 Executive Summary**

The **MIG-Transformer** addresses the inefficiency of standard Self-Attention, which treats every token interaction as potentially valuable. By introducing a learnable "Information Gate" prior to the attention block, the model dynamically filters out tokens with low predicted utility (e.g., redundant determiners, repetitive patterns) before the computationally expensive attention operation.

### **1.2 Core Architectural Components**

#### **A. The Information Gate (IG)**

A lightweight, pointwise Multi-Layer Perceptron (MLP) acts as a "bouncer" for the attention layer.

* **Input:** Token hidden state .
* **Operation:** Projects the high-dimensional state to a scalar "importance score" .
* **Activation:** Sigmoid, to ensure a smooth probability-like gating factor.

#### **B. Gated Attention Mechanism**

Unlike standard attention where , MIG modifies the input stream itself.

* The query/key/value projections effectively receive scaled inputs: .
* If , the token contributes a zero-vector to the attention weighted sum, effectively removing it from the context window of other tokens.

### **1.3 Mathematical Formulation**

Let  be the hidden state of token .

1. **Gate Calculation:**


2. **Gated Attention:**


3. **Sparsity Objective (Loss Function):**
To prevent the gate from staying open () permanently, we add an L1 penalty to the loss function:



### **1.4 Data Flow Diagram**

1. **Input Tensor** `[Batch, Seq, Hidden]`
2. **Branch A:** Compute Main Attention (Standard Path).
3. **Branch B:** Compute Gate Score (MLP -> Sigmoid).
4. **Fusion:** Element-wise multiplication of Input  Gate Score.
5. **Attention:** Run standard GQA on fused input.
6. **Residual:** Add output to original stream.

---

## **2. Stochastic Induction Transformer (SIL-Transformer)**

### **2.1 Executive Summary**

The **SIL-Transformer** introduces a probabilistic path to the architecture. Standard Transformers are purely deterministic "pattern matchers." SIL adds a discrete latent variable  that forces the model to categorize the current context into one of  distinct "rules" or "hypotheses." This improves systematic generalization by disentangling *rule selection* from *token matching*.

### **2.2 Core Architectural Components**

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

### **2.3 Mathematical Formulation**

Let  be the number of latent rules.

1. **Rule Logits:**


2. **Stochastic Sampling (Gumbel-Max):**



*Where  are i.i.d samples drawn from Gumbel(0, 1).*
3. **Rule Injection:**



### **2.4 Data Flow Diagram**

1. **Input Tensor** `[Batch, Seq, Hidden]`
2. **Path 1 (Deterministic):** Standard Multi-Head Attention.
3. **Path 2 (Stochastic):**
* Linear Projection to  logits.
* **Sample**  via Gumbel-Softmax.
* **Decode**  back to hidden dimension.


4. **Aggregation:** Add Path 1 Output + Path 2 Output.

---

## **3. Attentional Symmetry Regularized Transformer (ASR-Transformer)**

### **3.1 Executive Summary**

The **ASR-Transformer** does not change the inference architecture (it looks like a standard Llama/Mistral model at runtime). Instead, it modifies the **training paradigm** by enforcing a geometric constraint: the model's internal attention map must remain stable (invariant) even when the input is subjected to semantic noise. This forces the model to learn robust, generalized features rather than memorizing exact token positions or noise patterns.

### **3.2 Core Architectural Components**

#### **A. The Siamese Forward Pass**

During training, the attention block is executed twice for every batch:

1. **Clean Pass:** 
2. **Noised Pass:** , where .

#### **B. The Consistency Regulator**

An auxiliary loss term that penalizes the divergence between the Clean and Noised outputs. This effectively smooths the loss landscape, creating "wide valleys" of low loss (better generalization) rather than "sharp minima" (overfitting).

### **3.3 Mathematical Formulation**

1. **Perturbation:**


2. **Consistency Loss (MSE approach):**



*Note: The target  is detached from the computation graph to prevent the model from collapsing to zero.*
3. **Total Objective:**



### **3.4 Data Flow Diagram**

1. **Input Tensor** `[Batch, Seq, Hidden]`
2. **Step 1:** Calculate Standard Attention Output .
3. **Step 2 (Training Only):**
* Generate noise tensor .
* Calculate .


4. **Step 3:** Compute MSE Loss between  and .
5. **Output:** Return  for the next layer.

---

### **Comparison & Benchmark Strategy**

| Feature | MIG-Transformer | SIL-Transformer | ASR-Transformer |
| --- | --- | --- | --- |
| **Primary Goal** | Efficiency (FLOPs reduction) | Reasoning (Rule adherence) | Robustness (Generalization) |
| **New Parameters** | Minimal (Gate MLP) | Moderate (Rule Embeddings) | None (Training only) |
| **Training Complexity** | Medium (Requires sparsity tuning) | High (Unstable gradients) | Low (Just add loss) |
| **Inference Cost** | **Lower** (can skip computation) | Slightly Higher (Logic path) | **Identical** to Baseline |

**Recommended Testing Order:**

1. **ASR:** Easiest to implement, high probability of improving your baseline `TinyStories` loss.
2. **MIG:** Good for analyzing which tokens the model thinks are "important."
3. **SIL:** Most experimental; save for last as it requires careful hyperparameter tuning (temperature, number of rules).