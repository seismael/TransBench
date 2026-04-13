from __future__ import annotations

import json
import tempfile
from pathlib import Path

from transbench.reporting import (
    BenchmarkResult,
    ModelBreakdown,
    ReportsConfig,
    RunMeta,
    SystemInfo,
    write_report_with_meta,
)


def test_write_report_creates_report_and_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    local_tmp_root = repo_root / ".pytest_tmp"
    local_tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=local_tmp_root) as td:
        tmp_path = Path(td)
        cfg = ReportsConfig(reports_dir=tmp_path)

        result = BenchmarkResult(
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
            config={"arch": "gqa", "num_layers": 2, "hidden_size": 16},
            metrics={
                "tokens_per_s": 123.4,
                "train_step_ms_mean": 8.0,
                "loss_mean": 2.0,
                "peak_mem_mb": None,
            },
        )

        class DummyCfg:
            arch = "gqa"

        report_path = write_report_with_meta(cfg, DummyCfg(), result, RunMeta())
        assert report_path.exists()

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert len(manifest["reports"]) == 1
        assert manifest["reports"][0]["file"] == report_path.name


def test_write_report_replace_overwrites_stable_filename():
    repo_root = Path(__file__).resolve().parents[1]
    local_tmp_root = repo_root / ".pytest_tmp"
    local_tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=local_tmp_root) as td:
        tmp_path = Path(td)
        cfg = ReportsConfig(reports_dir=tmp_path)

        result = BenchmarkResult(
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
            config={"arch": "gqa", "num_layers": 2, "hidden_size": 16},
            metrics={"tokens_per_s": 1.0, "train_step_ms_mean": 1.0, "loss_mean": 1.0, "peak_mem_mb": 1.0},
        )

        class DummyCfg:
            arch = "gqa"

        fixed = "stable.json"
        p1 = write_report_with_meta(cfg, DummyCfg(), result, RunMeta(name="one"), fixed_filename=fixed)
        p2 = write_report_with_meta(cfg, DummyCfg(), result, RunMeta(name="two"), fixed_filename=fixed)

        assert p1.name == fixed
        assert p2.name == fixed
        assert p2.exists()

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        # only one entry for that file
        files = [r["file"] for r in manifest["reports"]]
        assert files.count(fixed) == 1
