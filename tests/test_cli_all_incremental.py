from __future__ import annotations

import json
from pathlib import Path

import transbench.cli as cli
from transbench.reporting import BenchmarkResult, ModelBreakdown, SystemInfo


def _fake_result(cfg) -> BenchmarkResult:
    system = SystemInfo(
        os="test",
        python="3.x",
        torch="test",
        cuda_available=False,
        cuda_version=None,
        gpu_name=None,
        cpu_brand="test",
        cpu_count=1,
        ram_gb=1.0,
        device=str(cfg.device or "cpu"),
    )
    model = ModelBreakdown(
        total_parameters=1234,
        embedding_parameters=100,
        mixin_parameters=200,
        ffn_parameters=300,
        lm_head_parameters=400,
    )
    config = {
        "arch": cfg.arch,
        "num_layers": cfg.num_layers,
        "hidden_size": cfg.hidden_size,
        "ffn_mult": cfg.ffn_mult,
        "num_heads": cfg.num_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "vocab_size": cfg.vocab_size,
        "initializer_range": getattr(cfg, "initializer_range", 0.02),
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "warmup": cfg.warmup,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "min_lr": getattr(cfg, "min_lr", 1e-6),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "tokenizer_model": getattr(cfg, "tokenizer_model", "gpt2"),
        "cache_dir": str(getattr(cfg, "cache_dir", "")),
        "offline": bool(getattr(cfg, "offline", False)),
    }
    metrics = {
        "tokens_per_s": 1000.0,
        "train_step_ms_mean": 1.0,
        "forward_ms_mean": 0.5,
        "loss_mean": 2.0,
        "peak_mem_mb": 10.0,
        "loss_series": [2.1, 2.0, 1.9],
        "lr_series": [cfg.learning_rate] * 3,
    }
    return BenchmarkResult(system=system, model=model, config=config, metrics=metrics)


def test_benchmark_all_is_incremental_by_default(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_run_benchmark(cfg):
        calls.append(cfg.arch)
        return _fake_result(cfg)

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    reports_dir = tmp_path / "reports"

    base_args = [
        "benchmark",
        "--all",
        "--reports-dir",
        str(reports_dir),
        "--dataset",
        "synthetic",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--num-layers",
        "1",
        "--hidden-size",
        "32",
        "--seq-len",
        "16",
        "--batch-size",
        "2",
        "--vocab-size",
        "128",
        "--warmup",
        "0",
        "--steps",
        "1",
    ]

    # First run should generate 3 stable reports.
    assert cli.main(base_args) == 0
    assert sorted(calls) == ["asr", "gqa", "mhla", "mig", "sil"]

    files1 = sorted(p.name for p in reports_dir.glob("*.json") if p.name != "manifest.json")
    assert len(files1) == 5

    # Second run should skip all (no new benchmark calls) and keep file count stable.
    calls.clear()
    assert cli.main(base_args) == 0
    assert calls == []

    files2 = sorted(p.name for p in reports_dir.glob("*.json") if p.name != "manifest.json")
    assert files2 == files1


def test_benchmark_all_replace_regenerates_all(tmp_path, monkeypatch):
    def fake_run_benchmark(cfg):
        return _fake_result(cfg)

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    reports_dir = tmp_path / "reports"

    args = [
        "benchmark",
        "--all",
        "--reports-dir",
        str(reports_dir),
        "--dataset",
        "synthetic",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--num-layers",
        "1",
        "--hidden-size",
        "32",
        "--seq-len",
        "16",
        "--batch-size",
        "2",
        "--vocab-size",
        "128",
        "--warmup",
        "0",
        "--steps",
        "1",
    ]

    assert cli.main(args) == 0

    reports = [p for p in reports_dir.glob("*.json") if p.name != "manifest.json"]
    assert len(reports) == 5

    before_ids = {}
    for p in reports:
        before_ids[p.name] = json.loads(p.read_text(encoding="utf-8"))["run"]["id"]

    assert cli.main(args + ["--replace"]) == 0

    after_ids = {}
    for p in reports_dir.glob("*.json"):
        if p.name == "manifest.json":
            continue
        after_ids[p.name] = json.loads(p.read_text(encoding="utf-8"))["run"]["id"]

    # Same filenames, new content.
    assert set(after_ids.keys()) == set(before_ids.keys())
    assert any(after_ids[k] != before_ids[k] for k in after_ids)
