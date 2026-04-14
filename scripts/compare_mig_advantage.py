#!/usr/bin/env python3
"""Analyse MIG noise-sweep results: sparse_signal and poisoned_needle reports.

Reads all JSON reports from `reports/` whose tags overlap with the MIG sweep
tags and produces:

  1. Noise sweep table   – Signal% × Arch → train loss, eval loss, tok/s
  2. Crossover analysis  – At which noise level does MIG beat GQA?
  3. Gate selectivity     – signal vs noise gate means (sparse_signal only)
  4. Verdict summary

Usage:
    python scripts/compare_mig_advantage.py [--reports-dir reports]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ──────────────────── helpers ────────────────────

def _load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for p in sorted(reports_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["_file"] = p.name
        reports.append(data)
    return reports


def _get_tags(report: dict[str, Any]) -> list[str]:
    """Get tags from run.tags (schema v2) or config.tags (legacy)."""
    tags = report.get("run", {}).get("tags") or report.get("config", {}).get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return tags


def _tag_match(report: dict[str, Any], prefix: str) -> bool:
    return any(t.startswith(prefix) for t in _get_tags(report))


def _arch_label(report: dict[str, Any]) -> str:
    cfg = report.get("config", {})
    arch = cfg.get("arch", "?")
    tags = _get_tags(report)
    if "amig" in tags:
        return "A-MIG"
    return arch.upper()


def _noise_pct(report: dict[str, Any]) -> float | None:
    """Return noise percentage (0-100) for unified comparison."""
    cfg = report.get("config", {})
    ds = cfg.get("dataset", "")
    if ds == "sparse_signal":
        sr = cfg.get("signal_ratio", 0.15)
        return round((1.0 - sr) * 100, 1)
    if ds == "poisoned_needle":
        pr = cfg.get("poison_ratio")
        if pr is not None:
            return round(float(pr) * 100, 1)
        # Fallback: extract from tags like "p50", "p70", "p85", "p95"
        tags = _get_tags(report)
        for t in tags:
            if t.startswith("p") and t[1:].isdigit():
                return float(t[1:])
        return round(0.85 * 100, 1)
    return None


def _fmt(x: object, width: int = 10) -> str:
    if x is None:
        return "-".center(width)
    if isinstance(x, float):
        return f"{x:.5f}".rjust(width)
    return str(x).rjust(width)


# ──────────────────── tables ────────────────────

def _print_sweep_table(reports: list[dict[str, Any]], dataset: str) -> None:
    """Print the noise-sweep table for one dataset."""
    rows: list[dict[str, Any]] = []
    for r in reports:
        cfg = r.get("config", {})
        if cfg.get("dataset") != dataset:
            continue
        metrics = r.get("metrics", {}) or {}
        loss_series = metrics.get("loss_series") or []
        loss_last = float(loss_series[-1]) if loss_series else None
        rows.append({
            "arch": _arch_label(r),
            "noise_pct": _noise_pct(r),
            "loss_last": loss_last,
            "eval_loss": metrics.get("eval_loss"),
            "tokens_per_s": metrics.get("tokens_per_s"),
            "gate_selectivity": metrics.get("mig_gate_selectivity"),
            "gate_signal": metrics.get("mig_gate_signal_mean"),
            "gate_noise": metrics.get("mig_gate_noise_mean"),
            "file": r.get("_file", ""),
        })

    if not rows:
        print(f"  (no reports for dataset={dataset})")
        return

    rows.sort(key=lambda r: (r["noise_pct"] or 0, r["arch"]))

    # Header
    hdr = (
        f"{'Noise%':>8}  {'Arch':>6}  {'TrainLoss':>10}  {'EvalLoss':>10}  "
        f"{'tok/s':>10}  {'GateSel':>10}  {'File'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(
            f"{_fmt(row['noise_pct'], 8)}  {row['arch']:>6}  "
            f"{_fmt(row['loss_last'])}  {_fmt(row['eval_loss'])}  "
            f"{_fmt(row['tokens_per_s'])}  {_fmt(row['gate_selectivity'])}  "
            f"{row['file']}"
        )


def _crossover_analysis(reports: list[dict[str, Any]], dataset: str) -> None:
    """Find the noise level at which MIG/A-MIG beat GQA."""
    # Bucket by noise_pct → arch → eval_loss (prefer eval_loss, fallback to loss_last)
    buckets: dict[float, dict[str, float]] = {}
    for r in reports:
        cfg = r.get("config", {})
        if cfg.get("dataset") != dataset:
            continue
        np_ = _noise_pct(r)
        if np_ is None:
            continue
        metrics = r.get("metrics", {}) or {}
        loss_series = metrics.get("loss_series") or []
        loss = metrics.get("eval_loss")
        if loss is None and loss_series:
            loss = float(loss_series[-1])
        if loss is None:
            continue
        arch = _arch_label(r)
        buckets.setdefault(np_, {})[arch] = loss

    if not buckets:
        print("  (no data for crossover analysis)")
        return

    print(f"{'Noise%':>8}  {'GQA':>10}  {'MIG':>10}  {'A-MIG':>10}  {'MIG win?':>9}  {'A-MIG win?':>10}")
    print("-" * 70)
    for np_ in sorted(buckets):
        b = buckets[np_]
        gqa = b.get("GQA")
        mig = b.get("MIG")
        amig = b.get("A-MIG")
        mig_win = "YES" if (gqa is not None and mig is not None and mig < gqa) else "no"
        amig_win = "YES" if (gqa is not None and amig is not None and amig < gqa) else "no"
        print(
            f"{np_:8.1f}  {_fmt(gqa)}  {_fmt(mig)}  {_fmt(amig)}  "
            f"{mig_win:>9}  {amig_win:>10}"
        )


def _gate_table(reports: list[dict[str, Any]]) -> None:
    """Print gate selectivity details (sparse_signal only)."""
    rows = []
    for r in reports:
        cfg = r.get("config", {})
        if cfg.get("dataset") != "sparse_signal":
            continue
        if _arch_label(r) not in ("MIG", "A-MIG"):
            continue
        metrics = r.get("metrics", {}) or {}
        gs = metrics.get("mig_gate_signal_mean")
        gn = metrics.get("mig_gate_noise_mean")
        sel = metrics.get("mig_gate_selectivity")
        if gs is None and gn is None:
            continue
        rows.append({
            "arch": _arch_label(r),
            "noise_pct": _noise_pct(r),
            "gate_signal": gs,
            "gate_noise": gn,
            "selectivity": sel,
        })
    if not rows:
        print("  (no gate selectivity data)")
        return

    rows.sort(key=lambda r: (r["noise_pct"] or 0, r["arch"]))
    print(f"{'Noise%':>8}  {'Arch':>6}  {'GateSignal':>11}  {'GateNoise':>11}  {'Selectivity':>12}")
    print("-" * 58)
    for row in rows:
        print(
            f"{_fmt(row['noise_pct'], 8)}  {row['arch']:>6}  "
            f"{_fmt(row['gate_signal'], 11)}  {_fmt(row['gate_noise'], 11)}  "
            f"{_fmt(row['selectivity'], 12)}"
        )


def _verdict(reports: list[dict[str, Any]]) -> None:
    """Print a high-level verdict."""
    wins = {"sparse_signal": {"MIG": 0, "A-MIG": 0}, "poisoned_needle": {"MIG": 0, "A-MIG": 0}}
    totals = {"sparse_signal": 0, "poisoned_needle": 0}

    for ds in ("sparse_signal", "poisoned_needle"):
        buckets: dict[float, dict[str, float]] = {}
        for r in reports:
            cfg = r.get("config", {})
            if cfg.get("dataset") != ds:
                continue
            np_ = _noise_pct(r)
            if np_ is None:
                continue
            metrics = r.get("metrics", {}) or {}
            loss_series = metrics.get("loss_series") or []
            loss = metrics.get("eval_loss")
            if loss is None and loss_series:
                loss = float(loss_series[-1])
            if loss is None:
                continue
            arch = _arch_label(r)
            buckets.setdefault(np_, {})[arch] = loss

        for np_ in buckets:
            b = buckets[np_]
            gqa = b.get("GQA")
            if gqa is None:
                continue
            totals[ds] += 1
            for a in ("MIG", "A-MIG"):
                if a in b and b[a] < gqa:
                    wins[ds][a] += 1

    print("Dataset            MIG wins   A-MIG wins   Total noise pts")
    print("-" * 60)
    for ds in ("sparse_signal", "poisoned_needle"):
        t = totals[ds]
        mig_w = wins[ds]["MIG"]
        amig_w = wins[ds]["A-MIG"]
        print(f"{ds:<20} {mig_w}/{t:<10} {amig_w}/{t:<12}")


# ──────────────────── main ────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing JSON reports (default: reports/)",
    )
    args = ap.parse_args()

    reports = _load_reports(args.reports_dir)
    # Filter to MIG sweep reports only (tags starting with "mig-")
    sweep = [r for r in reports if _tag_match(r, "mig-")]

    if not sweep:
        print("No MIG sweep reports found. Run the benchmarks first.")
        print(f"  Searched: {args.reports_dir.resolve()}")
        return 1

    print(f"Found {len(sweep)} MIG sweep report(s) in {args.reports_dir.resolve()}\n")

    # ── sparse_signal ──
    print("=" * 72)
    print("  SPARSE SIGNAL SWEEP")
    print("=" * 72)
    _print_sweep_table(sweep, "sparse_signal")
    print()
    _crossover_analysis(sweep, "sparse_signal")
    print()

    # ── poisoned_needle ──
    print("=" * 72)
    print("  POISONED NEEDLE SWEEP")
    print("=" * 72)
    _print_sweep_table(sweep, "poisoned_needle")
    print()
    _crossover_analysis(sweep, "poisoned_needle")
    print()

    # ── Gate selectivity ──
    print("=" * 72)
    print("  GATE SELECTIVITY (sparse_signal only)")
    print("=" * 72)
    _gate_table(sweep)
    print()

    # ── Verdict ──
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    _verdict(sweep)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
