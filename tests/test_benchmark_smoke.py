from __future__ import annotations

from pathlib import Path

import pytest

from transbench.benchmark import BenchmarkConfig, run_benchmark


@pytest.mark.parametrize("arch", ["gqa", "mig", "sil", "asr", "mhla"])
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


def test_sil_gate_series_collected():
    """SIL should produce gate series when trained with sil_lambda > 0."""
    cfg = BenchmarkConfig(
        arch="sil",
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
        steps=3,
        learning_rate=1e-4,
        min_lr=1e-6,
        dataset="synthetic",
        seed=42,
        cache_dir=None,
        offline=False,
        device="cpu",
        dtype="float32",
        tokenizer_model="gpt2",
        mig_gate_dim=64,
        mig_lambda=0.0,
        sil_num_latent_rules=16,
        sil_temperature=1.0,
        sil_hard_train=False,
        sil_hard_eval=True,
        asr_noise_std=0.01,
        asr_lambda=0.0,
        sil_lambda=0.005,
    )
    result = run_benchmark(cfg)
    gate_series = result.metrics.get("sil_gate_series")
    assert gate_series is not None
    assert len(gate_series) == 3
    # With bias=-1 init, gate should start near sigmoid(-1) ≈ 0.27
    assert 0.05 < gate_series[0] < 0.6


def test_mig_gate_series_collected():
    """MIG should produce gate series when trained with mig_lambda > 0."""
    cfg = BenchmarkConfig(
        arch="mig",
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
        steps=3,
        learning_rate=1e-4,
        min_lr=1e-6,
        dataset="synthetic",
        seed=42,
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
        asr_lambda=0.0,
    )
    result = run_benchmark(cfg)
    gate_series = result.metrics.get("mig_gate_series")
    assert gate_series is not None
    assert len(gate_series) == 3
    # With bias=-2 init, gate should start near sigmoid(-2) ≈ 0.12
    assert 0.01 < gate_series[0] < 0.4


def test_aux_warmup_ramps_lambda():
    """With aux_warmup_steps, aux loss should be smaller at step 0 than at convergence."""
    cfg = BenchmarkConfig(
        arch="mig",
        num_layers=1,
        hidden_size=32,
        ffn_mult=2.0,
        num_heads=4,
        num_kv_heads=None,
        vocab_size=128,
        initializer_range=0.02,
        seq_len=16,
        batch_size=2,
        warmup=2,
        steps=4,
        learning_rate=1e-4,
        min_lr=1e-6,
        dataset="synthetic",
        seed=42,
        cache_dir=None,
        offline=False,
        device="cpu",
        dtype="float32",
        tokenizer_model="gpt2",
        mig_gate_dim=64,
        mig_lambda=1e-2,
        sil_num_latent_rules=64,
        sil_temperature=1.0,
        sil_hard_train=True,
        sil_hard_eval=True,
        asr_noise_std=0.01,
        asr_lambda=0.0,
        aux_warmup_steps=4,
    )
    result = run_benchmark(cfg)
    assert result.config["aux_warmup_steps"] == 4
    assert result.metrics.get("mig_gate_series") is not None
