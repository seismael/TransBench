from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import uuid
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    importlib_metadata = None  # type: ignore


@dataclass(frozen=True)
class ReportsConfig:
    reports_dir: Path


@dataclass(frozen=True)
class RunMeta:
    name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    include_raw: bool = False


@dataclass(frozen=True)
class SystemInfo:
    os: str
    python: str
    torch: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    cpu_brand: str | None
    cpu_count: int
    ram_gb: float
    device: str


def _package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    if importlib_metadata is None:
        return versions
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            continue
    return versions


@dataclass(frozen=True)
class ModelBreakdown:
    total_parameters: int
    embedding_parameters: int
    mixin_parameters: int
    ffn_parameters: int
    lm_head_parameters: int


@dataclass(frozen=True)
class BenchmarkResult:
    system: SystemInfo
    model: ModelBreakdown
    config: dict[str, Any]
    metrics: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _manifest_path(cfg: ReportsConfig) -> Path:
    return cfg.reports_dir / "manifest.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_json(payload: Any) -> Any:
    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            return None
        return payload
    if isinstance(payload, dict):
        return {k: _sanitize_json(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize_json(v) for v in payload]
    return payload


def _write_json(path: Path, payload: Any) -> None:
    safe = _sanitize_json(payload)
    path.write_text(
        json.dumps(safe, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_report(cfg: ReportsConfig, bench_cfg: Any, result: BenchmarkResult) -> Path:
    return write_report_with_meta(cfg, bench_cfg, result, RunMeta())


def write_report_with_meta(
    cfg: ReportsConfig,
    bench_cfg: Any,
    result: BenchmarkResult,
    meta: RunMeta,
    *,
    fixed_filename: str | None = None,
) -> Path:
    _ensure_dir(cfg.reports_dir)

    now = _utc_now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:10]

    arch = _safe_slug(getattr(bench_cfg, "arch", "arch"))
    filename = fixed_filename or f"{ts}_{arch}_{run_id}.json"
    report_path = cfg.reports_dir / filename

    tags = meta.tags or []

    report = {
        "schema_version": 2,
        "run": {
            "id": run_id,
            "timestamp_utc": now.isoformat(),
            "name": meta.name,
            "tags": tags,
            "notes": meta.notes,
            "command": "transbench benchmark",
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "hostname": platform.node(),
            "git_commit": _git_commit(),
        },
        "system": asdict(result.system),
        "packages": _package_versions(["torch", "numpy", "einops", "psutil"]),
        "model": asdict(result.model),
        "config": result.config,
        "metrics": result.metrics,
    }

    # Derived metrics useful for dashboards (keep stable names)
    try:
        tps = float(result.metrics.get("tokens_per_s") or 0.0)
        params_m = float(result.model.total_parameters) / 1e6
        report["derived"] = {
            "params_m": params_m,
            "tokens_per_s_per_mparam": (tps / params_m) if params_m > 0 else None,
        }
    except Exception:
        report["derived"] = {}

    # If using a fixed filename, overwrite atomically-ish (best-effort).
    _write_json(report_path, report)
    _update_manifest(cfg, report_path, report)
    return report_path


def _update_manifest(cfg: ReportsConfig, report_path: Path, report_payload: dict[str, Any]) -> None:
    manifest_path = _manifest_path(cfg)
    rel = report_path.as_posix().split("/", 1)[-1] if report_path.is_absolute() else report_path.as_posix()

    entry = {
        "file": report_path.name,
        "timestamp_utc": report_payload.get("run", {}).get("timestamp_utc"),
        "name": report_payload.get("run", {}).get("name"),
        "tags": report_payload.get("run", {}).get("tags"),
        "arch": report_payload.get("config", {}).get("arch"),
        "dataset": report_payload.get("config", {}).get("dataset"),
        "device": report_payload.get("config", {}).get("device", "cpu"),
        "dtype": report_payload.get("config", {}).get("dtype"),
        "hidden_size": report_payload.get("config", {}).get("hidden_size"),
        "num_layers": report_payload.get("config", {}).get("num_layers"),
        "learning_rate": report_payload.get("config", {}).get("learning_rate"),
        "params": report_payload.get("model", {}).get("total_parameters"),
        "tokens_per_s": report_payload.get("metrics", {}).get("tokens_per_s"),
        "tokens_per_s_per_mparam": report_payload.get("derived", {}).get("tokens_per_s_per_mparam"),
        "train_step_ms_mean": report_payload.get("metrics", {}).get("train_step_ms_mean"),
        "forward_ms_mean": report_payload.get("metrics", {}).get("forward_ms_mean"),
        "loss_mean": report_payload.get("metrics", {}).get("loss_mean"),
        "peak_mem_mb": report_payload.get("metrics", {}).get("peak_mem_mb"),
    }

    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {"schema_version": 1, "reports": []}

    reports = [r for r in manifest.get("reports", []) if r.get("file") != report_path.name]
    reports.append(entry)

    def _sort_key(r: dict[str, Any]) -> str:
        return r.get("timestamp_utc") or ""

    reports.sort(key=_sort_key, reverse=True)

    manifest["reports"] = reports
    _write_json(manifest_path, manifest)


def rebuild_manifest(cfg: ReportsConfig) -> Path:
    _ensure_dir(cfg.reports_dir)
    manifest_path = _manifest_path(cfg)

    reports = []
    for path in sorted(cfg.reports_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        reports.append(
            {
                "file": path.name,
                "timestamp_utc": payload.get("run", {}).get("timestamp_utc"),
                "name": payload.get("run", {}).get("name"),
                "tags": payload.get("run", {}).get("tags"),
                "arch": payload.get("config", {}).get("arch"),
                "dataset": payload.get("config", {}).get("dataset"),
                "device": payload.get("config", {}).get("device", "cpu"),
                "dtype": payload.get("config", {}).get("dtype"),
                "hidden_size": payload.get("config", {}).get("hidden_size"),
                "num_layers": payload.get("config", {}).get("num_layers"),
                "learning_rate": payload.get("config", {}).get("learning_rate"),
                "params": payload.get("model", {}).get("total_parameters"),
                "tokens_per_s": payload.get("metrics", {}).get("tokens_per_s"),
                "tokens_per_s_per_mparam": payload.get("derived", {}).get("tokens_per_s_per_mparam"),
                "train_step_ms_mean": payload.get("metrics", {}).get("train_step_ms_mean"),
                "forward_ms_mean": payload.get("metrics", {}).get("forward_ms_mean"),
                "loss_mean": payload.get("metrics", {}).get("loss_mean"),
                "peak_mem_mb": payload.get("metrics", {}).get("peak_mem_mb"),
            }
        )

    reports.sort(key=lambda r: r.get("timestamp_utc") or "", reverse=True)
    _write_json(manifest_path, {"schema_version": 1, "reports": reports})
    return manifest_path
