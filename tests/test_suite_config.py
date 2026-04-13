from __future__ import annotations

from pathlib import Path

from transbench.reporting import BenchmarkResult, ModelBreakdown, ReportsConfig, SystemInfo
from transbench.suite import load_suite_toml, run_suite


def test_suite_runs_from_example_toml(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    suite = repo_root / "benchmarks.example.toml"

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    entries = load_suite_toml(suite)
    cfg = ReportsConfig(reports_dir=reports_dir)

    def dummy_runner(_bench):
        return BenchmarkResult(
            system=SystemInfo(
                os="test",
                python="3.x",
                torch="2.x",
                cuda_available=False,
                cuda_version=None,
                gpu_name=None,
                cpu_brand=None,
                cpu_count=1,
                ram_gb=1.0,
                device="cpu",
            ),
            model=ModelBreakdown(
                total_parameters=10,
                embedding_parameters=2,
                mixin_parameters=3,
                ffn_parameters=4,
                lm_head_parameters=1,
            ),
            config={"arch": "gqa", "num_layers": 1, "hidden_size": 64},
            metrics={"tokens_per_s": 1.0, "train_step_ms_mean": 1.0, "loss_mean": 1.0, "peak_mem_mb": 1.0},
        )

    paths = run_suite(entries, cfg, dummy_runner)
    assert len(paths) >= 1
    assert (reports_dir / "manifest.json").exists()
