# TransBench

**Transformer benchmarking framework for specialized architecture research.**

TransBench is a research benchmarking framework for designing and validating specialized Transformer variants. Rather than competing with Mamba2 or standard Transformers on general language modeling, we engineer architectures with structural inductive biases that provably outperform generalist models in specific adversarial environments.

---

## Research Thesis

General sequence models (SSMs, standard Transformers, Lightning Attention) are Jacks of all trades. This generalism creates two exploitable vulnerabilities:

1. **Context Dilution** — Recurrent models must update their hidden state with every token. In streams containing 95%+ noise, the state becomes irreparably corrupted.
2. **Latent Blurring** — Softmax attention is a continuous function. When faced with mutually exclusive logical rules, standard models blend them, producing hallucinations.

TransBench exploits these vulnerabilities through two research tracks:

### Track 1: Asymmetric Mutual Information Gating (A-MIG) — *The Skimmer*
**Status: Implemented** | Target: Edge AI, IoT Telemetry, Financial Tick Data, Cyber Log Parsing

A-MIG applies aggressive, asymmetric token filtering across the layer stack. Early layers (0–2) use hard Top-K routing at extreme sparsity ($K \approx 0.05$), dropping 95% of the input sequence. Deeper layers (3–7) operate as standard dense attention on only the concentrated 5% signal that survived.

Unlike standard Mixture-of-Depths which distributes sparsity evenly, A-MIG structurally forbids early layers from deep reasoning — their sole purpose is noise classification and removal.

### Track 2: Stochastic Induction Layer (SIL) — *The Logic Switch*
**Status: Planned** | Target: Theorem Proving, Legal Tech, Deterministic Code Execution

SIL replaces continuous attention with discrete rule-selection via a Gumbel-Softmax bottleneck, forcing the model to commit to a single latent logical rule before generating output. This prevents concept blurring and enables structural explainability.

---

## Quick Start

Requires Python ≥ 3.10. Optimized for [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/seismael/TransBench.git
cd TransBench

# Core install (CPU benchmarks, synthetic datasets)
uv pip install -e ".[dev]"

# + TinyStories / Poisoned Needle dataset support
uv pip install -e ".[dev,train]"

# + Mamba2, RWKV6, RetNet mixins (Linux/CUDA only)
uv pip install -e ".[dev,train,fla]"
```

---

## Running Benchmarks

### A-MIG vs Baselines (Poisoned Needle)

The primary experiment: A-MIG Skimmer vs standard MIG vs dense GQA on the adversarial Poisoned Needle dataset.

```bash
transbench suite --config benchmarks/benchmarks.amig.toml
```

This runs three configurations:

| Run | Architecture | Keep Ratios | Dataset |
|-----|-------------|------------|---------|
| `amig-skimmer-poisoned` | MIG (A-MIG) | `[0.05, 0.05, 0.05, 1, 1, 1, 1, 1]` | poisoned_needle |
| `mig-baseline-poisoned` | MIG (uniform) | `0.7` uniform | poisoned_needle |
| `gqa-baseline-poisoned` | GQA (dense) | N/A | poisoned_needle |

**Expected outcome:** A-MIG maintains lower validation loss than both baselines because the skimmer layers gate out the 85% poison before it reaches the reasoning layers.

### Single Architecture Run

```bash
# A-MIG with per-layer keep ratios on Poisoned Needle
transbench benchmark \
  --arch mig \
  --dataset poisoned_needle \
  --poison-ratio 0.85 \
  --mig-layer-keep-ratios 0.05,0.05,0.05,1,1,1,1,1 \
  --num-layers 8 --hidden-size 128 --num-heads 4 --num-kv-heads 2 \
  --seq-len 128 --batch-size 4 --steps 100 --device cpu --dtype float32

# Standard MIG (uniform sparsity)
transbench benchmark --arch mig --dataset poisoned_needle --mig-keep-ratio 0.7

# Dense GQA baseline
transbench benchmark --arch gqa --dataset poisoned_needle

# SIL with Gumbel-Softmax
transbench benchmark --arch sil --sil-num-rules 64 --sil-temperature 0.5

# Any architecture on TinyStories
transbench benchmark --arch gqa --dataset tinystories --steps 500
```

### All Architectures (Quick Validation)

```bash
transbench benchmark --all --steps 10 --device cpu --dtype float32
```

### Datasets

| Dataset | Description | Requires `[train]` |
|---------|------------|-------------------|
| `tinystories` | Real language data (GPT2 tokenized) | Yes |
| `poisoned_needle` | TinyStories with 80–90% center-injected noise | Yes |
| `synthetic` | Random token IDs | No |
| `zeros` | All-zero sequences | No |
| `ramp` | Monotonic ascending IDs | No |

### Prepare Dataset for Offline Use

```bash
transbench prepare --dataset tinystories
```

---

## Visualizing Results

```bash
transbench serve-dashboard
# Open http://127.0.0.1:8000/dashboard/index.html
```

Reports are JSON files written to `reports/`. The dashboard renders loss curves, timing data, and system telemetry for side-by-side comparison.

Rebuild the report manifest after manual edits:

```bash
transbench make-manifest
```

---

## Project Structure

```
src/transbench/
  cli.py                    CLI entrypoint (benchmark, suite, serve-dashboard)
  benchmark.py              BenchmarkConfig, training loop, model building
  datasets.py               Dataset generation (synthetic, TinyStories, Poisoned Needle)
  suite.py                  TOML suite loader
  reporting.py              JSON report writing, system telemetry
  clean.py                  Cache/artifact cleanup
  modules/
    mig_module.py           A-MIG: per-layer Top-K gating (Track 1)
    sil_module.py           SIL: Gumbel-Softmax discrete rule selection (Track 2)
    asr_module.py           ASR: adversarial stability regularization
    mixin_modules.py        GQA, MHLA, Mamba2, RWKV6, RetNet mixins
    archi_modules.py        ArchiTransformerStack (model assembly)
    ffn_modules.py          Feed-forward networks
    positionnal_modules.py  Positional embeddings
benchmarks/
  benchmarks.amig.toml      A-MIG vs baselines on Poisoned Needle
  benchmarks.example.toml   Minimal example config
reports/                    Generated JSON benchmark reports
dashboard/                  Static HTML/JS dashboard
tests/                      pytest suite
```

---

## Supported Architectures

**Always available (CPU + CUDA):**

| Architecture | CLI name | Description |
|-------------|----------|------------|
| Grouped Query Attention | `gqa` | Dense attention baseline |
| Mutual Information Gating | `mig` | Sparse attention with optional per-layer A-MIG Top-K |
| Stochastic Induction Layer | `sil` | Gumbel-Softmax discrete rule selection |
| Adversarial Stability Reg. | `asr` | Noise-injection invariance regularization |
| Multi-Head Latent Attention | `mhla` | DeepSeek-V2 style compressed KV |

**Optional (requires `[fla]`, Linux + CUDA):**

| Architecture | CLI name | Description |
|-------------|----------|------------|
| Mamba2 | `mamba2` | State-space model via flash-linear-attention |
| RWKV v6 | `rwkv6` | Receptance Weighted Key Value v6 |
| RetNet | `retnet` | Retentive Network |

---

## Testing

```bash
# Smoke tests (all architectures, fast)
uv run pytest tests/test_benchmark_smoke.py

# Full test suite
uv run pytest

# Verify clean build
uv run pytest tests/test_clean.py
```

---

## Methodology

All benchmarks enforce controlled conditions:

- **Same tokenizer** — GPT2 (default)
- **Same training schedule** — Warmup + cosine decay to `min_lr`
- **Same hardware** — Reports capture OS, CPU, GPU, RAM, PyTorch version
- **Deterministic seeding** — Reproducible via `--seed`

Reports include system telemetry, per-step loss curves, timing breakdowns, and model parameter counts.

---

## Attribution

This project originated from [ArchiFactory](https://github.com/gabrielolympie/ArchiFactory) (MIT License) and has diverged significantly to focus on specialized Transformer benchmarking research.

---

## License

MIT — see [LICENSE](LICENSE).
