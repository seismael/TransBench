from __future__ import annotations

from pathlib import Path

import pytest

from transbench.benchmark import BenchmarkConfig, run_benchmark


@pytest.mark.parametrize("arch", ["gqa", "mig", "sil", "asr", "mhla", "rnn", "lstm"])
def test_run_benchmark_smoke_cpu(arch: str):
    cfg = BenchmarkConfig(
        arch=arch,
        num_layers=1,
        hidden_size=32,
        ffn_mult=2.0,
        num_heads=4,
        num_kv_heads=None,
        vocab_size=128,
        initializer_range=0.02,
        seq_len=16,
        batch_size=2,
        warmup=0,
        steps=1,
        learning_rate=1e-4,
        min_lr=1e-6,
        dataset="synthetic",
        seed=123,
        cache_dir=None,
        offline=False,
        device="cpu",
        dtype="float32",
        tokenizer_model="gpt2",
        mig_gate_dim=64,
        mig_lambda=1e-3,
        sil_num_latent_rules=64,
        sil_temperature=1.0,
        sil_hard_train=True,
        sil_hard_eval=True,
        asr_noise_std=0.01,
        asr_lambda=1e-3,
    )

    result = run_benchmark(cfg)
    assert result.config["arch"] == arch
    assert result.model.total_parameters > 0
    assert result.metrics["tokens_per_s"] > 0
    assert result.metrics["train_step_ms_mean"] > 0
