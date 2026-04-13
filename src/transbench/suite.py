from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from transbench.benchmark import BenchmarkConfig
from transbench.reporting import BenchmarkResult, ReportsConfig, RunMeta, write_report_with_meta
from transbench.datasets import default_cache_dir


@dataclass(frozen=True)
class SuiteEntry:
    bench: BenchmarkConfig
    meta: RunMeta


def load_suite_toml(path: Path) -> list[SuiteEntry]:
    import tomllib

    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    benches = payload.get("bench")
    if not isinstance(benches, list) or not benches:
        raise ValueError("Suite config must contain one or more [[bench]] tables")

    entries: list[SuiteEntry] = []
    for i, b in enumerate(benches, start=1):
        if not isinstance(b, dict):
            continue

        bench_cfg = BenchmarkConfig(
            arch=b.get("arch", "gqa"),
            num_layers=int(b.get("num_layers", 8)),
            hidden_size=int(b.get("hidden_size", 512)),
            ffn_mult=float(b.get("ffn_mult", 4.0)),
            num_heads=int(b.get("num_heads", 8)),
            num_kv_heads=b.get("num_kv_heads", None),
            vocab_size=int(b.get("vocab_size", 32000)),
            initializer_range=float(b.get("initializer_range", 0.02)),
            seq_len=int(b.get("seq_len", 256)),
            batch_size=int(b.get("batch_size", 2)),
            warmup=int(b.get("warmup", 0)),
            steps=int(b.get("steps", 50)),
            learning_rate=float(b.get("learning_rate", b.get("lr", 2e-4))),
            min_lr=float(b.get("min_lr", 1e-6)),
            dataset=str(b.get("dataset", "synthetic")),
            seed=b.get("seed", None),
            device=b.get("device", None),
            dtype=b.get("dtype", None),
            tokenizer_model=str(b.get("tokenizer_model", b.get("tokenizer", "gpt2"))),
            cache_dir=Path(str(b.get("cache_dir", default_cache_dir()))),
            offline=bool(b.get("offline", False)),
            mig_gate_dim=int(b.get("mig_gate_dim", b.get("gate_dim", 64))),
            mig_lambda=float(b.get("mig_lambda", b.get("lambda_sparsity", 0.01))),
            mig_keep_ratio=float(b.get("mig_keep_ratio", b.get("keep_ratio", 0.7))),
            mig_layer_keep_ratios=tuple(float(x) for x in b["mig_layer_keep_ratios"]) if "mig_layer_keep_ratios" in b else None,
            poison_ratio=float(b.get("poison_ratio", 0.85)),
            sil_num_latent_rules=int(b.get("sil_num_latent_rules", b.get("sil_num_rules", b.get("num_latent_rules", 64)))),
            sil_temperature=float(b.get("sil_temperature", b.get("temperature", 0.5))),
            sil_hard_train=bool(b.get("sil_hard_train", b.get("hard_train", False))),
            sil_hard_eval=bool(b.get("sil_hard_eval", b.get("hard_eval", True))),
            asr_noise_std=float(b.get("asr_noise_std", b.get("noise_std", 0.3))),
            asr_lambda=float(b.get("asr_lambda", b.get("lambda_asr", 0.1))),
            sil_lambda=float(b.get("sil_lambda", b.get("lambda_sil", 0.01))),
        )

        meta = RunMeta(
            name=b.get("run_name") or f"suite:{path.stem}#{i}",
            tags=list(b.get("tags") or []),
            notes=b.get("notes"),
            include_raw=bool(b.get("raw", False)),
        )

        entries.append(SuiteEntry(bench=bench_cfg, meta=meta))

    if not entries:
        raise ValueError("Suite config parsed zero benchmark entries")

    return entries


def run_suite(
    entries: Iterable[SuiteEntry],
    reports_cfg: ReportsConfig,
    runner: Callable[[BenchmarkConfig], BenchmarkResult],
) -> list[Path]:
    paths: list[Path] = []
    for entry in entries:
        result = runner(entry.bench)
        report_path = write_report_with_meta(reports_cfg, entry.bench, result, entry.meta)
        paths.append(report_path)
    return paths
