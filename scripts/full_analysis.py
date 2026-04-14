#!/usr/bin/env python3
"""Full analysis of all TransBench results across all three tracks.

Reads all JSON reports from reports/ and produces:
  1. TinyStories baseline table (all architectures)
  2. Track 1: MIG noise sweep (sparse_signal + poisoned_needle)
  3. Track 2: SIL diagnostics (gate dynamics, aux loss)
  4. Track 3: ASR diagnostics (consistency loss, training overhead)
  5. Cross-track summary

Usage:
    python scripts/full_analysis.py [--reports-dir reports]
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


def _get(report: dict[str, Any], *keys: str) -> Any:
    """Walk nested keys: _get(r, 'metrics', 'eval_loss')."""
    val: Any = report
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def _tags(report: dict[str, Any]) -> list[str]:
    t = _get(report, "run", "tags") or _get(report, "config", "tags") or []
    return [t] if isinstance(t, str) else list(t)


def _arch(r: dict[str, Any]) -> str:
    return (_get(r, "config", "arch") or "?").lower()


def _arch_label(r: dict[str, Any]) -> str:
    a = _arch(r)
    if "amig" in _tags(r):
        return "A-MIG"
    return a.upper()


def _dataset(r: dict[str, Any]) -> str:
    return _get(r, "config", "dataset") or "?"


def _device(r: dict[str, Any]) -> str:
    return (_get(r, "config", "device") or "?").lower()


def _steps(r: dict[str, Any]) -> int:
    return int(_get(r, "config", "steps") or 0)


def _loss_last(r: dict[str, Any]) -> float | None:
    series = _get(r, "metrics", "loss_series") or []
    nums = [x for x in series if isinstance(x, (int, float))]
    return float(nums[-1]) if nums else None


def _loss_best(r: dict[str, Any]) -> float | None:
    series = _get(r, "metrics", "loss_series") or []
    nums = [x for x in series if isinstance(x, (int, float))]
    return float(min(nums)) if nums else None


def _eval_loss(r: dict[str, Any]) -> float | None:
    v = _get(r, "metrics", "eval_loss")
    return float(v) if v is not None else None


def _tokens_per_s(r: dict[str, Any]) -> float | None:
    v = _get(r, "metrics", "tokens_per_s")
    return float(v) if v is not None else None


def _step_ms(r: dict[str, Any]) -> float | None:
    v = _get(r, "metrics", "train_step_ms_mean")
    return float(v) if v is not None else None


def _fwd_ms(r: dict[str, Any]) -> float | None:
    v = _get(r, "metrics", "forward_ms_p50")
    return float(v) if v is not None else None


def _peak_mem(r: dict[str, Any]) -> float | None:
    v = _get(r, "metrics", "peak_mem_mb")
    return float(v) if v is not None else None


def _params(r: dict[str, Any]) -> int | None:
    v = _get(r, "model", "total_parameters")
    return int(v) if v is not None else None


def _noise_pct(r: dict[str, Any]) -> float | None:
    ds = _dataset(r)
    if ds == "sparse_signal":
        sr = float(_get(r, "config", "signal_ratio") or 0.15)
        return round((1.0 - sr) * 100, 1)
    if ds == "poisoned_needle":
        # First try config.poison_ratio
        pr = _get(r, "config", "poison_ratio")
        if pr is not None:
            return round(float(pr) * 100, 1)
        # Fallback: extract from tags like "p50", "p70", "p85", "p95"
        for t in _tags(r):
            if t.startswith("p") and t[1:].isdigit():
                return float(t[1:])
        return round(0.85 * 100, 1)  # default
    return None


def _fmt(x: object, width: int = 10, decimals: int = 4) -> str:
    if x is None:
        return "-".center(width)
    if isinstance(x, float):
        return f"{x:.{decimals}f}".rjust(width)
    if isinstance(x, int):
        return f"{x:,}".rjust(width)
    return str(x).rjust(width)


def _bar(label: str) -> None:
    print(f"\n{'=' * 76}")
    print(f"  {label}")
    print(f"{'=' * 76}\n")


# ──────────────────── Sections ────────────────────

def section_tinystories_baselines(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Print baseline comparison for TinyStories CUDA runs and return best data."""
    _bar("TINYSTORIES BASELINES (8 Architectures, CUDA)")

    runs = [r for r in reports
            if _dataset(r) == "tinystories" and _device(r) == "cuda" and _steps(r) >= 2000]

    if not runs:
        print("  (no TinyStories CUDA ≥2000-step reports found)")
        return {}

    # Deduplicate: keep latest per arch
    by_arch: dict[str, dict[str, Any]] = {}
    for r in runs:
        a = _arch(r)
        if a not in by_arch or r["_file"] > by_arch[a]["_file"]:
            by_arch[a] = r

    archs_sorted = sorted(by_arch, key=lambda a: _loss_last(by_arch[a]) or 999)

    header = (
        f"{'Rank':>4}  {'Arch':>8}  {'Track':>10}  {'Loss(last)':>11}  {'Loss(best)':>11}  "
        f"{'EvalLoss':>11}  {'tok/s':>8}  {'Params':>9}  {'Step ms':>8}  {'VRAM MB':>8}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, Any]] = {}
    for rank, a in enumerate(archs_sorted, 1):
        r = by_arch[a]
        track = {"mig": "Track 1", "sil": "Track 2", "asr": "Track 3"}.get(a, "Baseline")
        row = {
            "loss_last": _loss_last(r),
            "loss_best": _loss_best(r),
            "eval_loss": _eval_loss(r),
            "tokens_per_s": _tokens_per_s(r),
            "params": _params(r),
            "step_ms": _step_ms(r),
            "peak_mem": _peak_mem(r),
            "fwd_ms": _fwd_ms(r),
        }
        results[a] = row
        tok_s_str = f"{int(row['tokens_per_s']):,}" if row["tokens_per_s"] else "-"
        par_str = f"{row['params']/1e6:.2f}M" if row["params"] else "-"
        print(
            f"{rank:>4}  {a.upper():>8}  {track:>10}  "
            f"{_fmt(row['loss_last'], 11, 4)}  {_fmt(row['loss_best'], 11, 4)}  "
            f"{_fmt(row['eval_loss'], 11, 4)}  {tok_s_str:>8}  {par_str:>9}  "
            f"{_fmt(row['step_ms'], 8, 0)}  {_fmt(row['peak_mem'], 8, 0)}"
        )

    # Compute deltas vs GQA
    gqa_data = results.get("gqa", {})
    gqa_loss = gqa_data.get("eval_loss") or gqa_data.get("loss_last")
    if gqa_loss:
        print(f"\n  GQA baseline loss: {gqa_loss:.4f}")
        print(f"  {'Arch':>8}  {'Delta':>8}  {'Relative':>8}")
        print(f"  {'-'*30}")
        for a in archs_sorted:
            if a == "gqa":
                continue
            d = results[a]
            loss = d.get("eval_loss") or d.get("loss_last")
            if loss:
                delta = loss - gqa_loss
                rel = delta / gqa_loss * 100
                print(f"  {a.upper():>8}  {delta:>+8.4f}  {rel:>+7.2f}%")

    return results


def section_mig_sweep(reports: list[dict[str, Any]], dataset: str) -> None:
    """Print MIG noise sweep for a given dataset."""
    ds_label = dataset.replace("_", " ").title()
    _bar(f"TRACK 1: MIG NOISE SWEEP — {ds_label.upper()}")

    sweep = [r for r in reports
             if _dataset(r) == dataset and _device(r) == "cuda" and _steps(r) >= 2000]

    if not sweep:
        print(f"  (no CUDA ≥2000-step reports for {dataset})")
        return

    # Group by noise_pct → arch_label → best report
    buckets: dict[float, dict[str, dict[str, Any]]] = {}
    for r in sweep:
        np_ = _noise_pct(r)
        if np_ is None:
            continue
        label = _arch_label(r)
        bucket = buckets.setdefault(np_, {})
        if label not in bucket or r["_file"] > bucket[label]["_file"]:
            bucket[label] = r

    # Print main table
    noise_levels = sorted(buckets)
    arch_order = ["GQA", "MIG", "A-MIG"]

    header = f"{'Noise%':>8}"
    for a in arch_order:
        header += f"  {a+' eval':>12}  {a+' train':>12}"
    header += f"  {'MIG win?':>9}  {'A-MIG win?':>11}"
    print(header)
    print("-" * len(header))

    for np_ in noise_levels:
        b = buckets[np_]
        row = f"{np_:>8.1f}"
        losses: dict[str, float | None] = {}
        for a in arch_order:
            r = b.get(a)
            el = _eval_loss(r) if r else None
            tl = _loss_last(r) if r else None
            losses[a] = el or tl
            row += f"  {_fmt(el, 12, 5)}  {_fmt(tl, 12, 5)}"
        gqa_l = losses.get("GQA")
        mig_l = losses.get("MIG")
        amig_l = losses.get("A-MIG")
        mig_w = "**YES**" if (gqa_l and mig_l and mig_l < gqa_l) else "no"
        amig_w = "**YES**" if (gqa_l and amig_l and amig_l < gqa_l) else "no"
        row += f"  {mig_w:>9}  {amig_w:>11}"
        print(row)

    # Gate selectivity sub-table
    gate_rows = []
    for r in sweep:
        if _arch(r) != "mig":
            continue
        gs = _get(r, "metrics", "mig_gate_signal_mean")
        gn = _get(r, "metrics", "mig_gate_noise_mean")
        sel = _get(r, "metrics", "mig_gate_selectivity")
        if gs is None and gn is None:
            continue
        gate_rows.append({
            "label": _arch_label(r),
            "noise_pct": _noise_pct(r),
            "gate_signal": gs,
            "gate_noise": gn,
            "selectivity": sel,
        })

    if gate_rows:
        gate_rows.sort(key=lambda x: (x["noise_pct"] or 0, x["label"]))
        print(f"\n  Gate Selectivity ({dataset}):")
        print(f"  {'Noise%':>8}  {'Config':>6}  {'GateSignal':>11}  {'GateNoise':>11}  {'Selectivity':>12}")
        print(f"  {'-'*56}")
        for gr in gate_rows:
            print(
                f"  {_fmt(gr['noise_pct'], 8)}  {gr['label']:>6}  "
                f"{_fmt(gr['gate_signal'], 11, 5)}  {_fmt(gr['gate_noise'], 11, 5)}  "
                f"{_fmt(gr['selectivity'], 12, 4)}"
            )


def section_sil_diagnostics(reports: list[dict[str, Any]]) -> None:
    """Print SIL diagnostic details."""
    _bar("TRACK 2: SIL DIAGNOSTICS")

    sil_runs = [r for r in reports
                if _arch(r) == "sil" and _device(r) == "cuda" and _steps(r) >= 2000]

    if not sil_runs:
        print("  (no SIL CUDA ≥2000-step reports found)")
        return

    for r in sil_runs:
        ds = _dataset(r)
        print(f"  Dataset: {ds}")
        print(f"  File: {r['_file']}")
        print(f"  Loss (last): {_fmt(_loss_last(r), 10, 4)}")
        print(f"  Loss (best): {_fmt(_loss_best(r), 10, 4)}")
        print(f"  Eval loss:   {_fmt(_eval_loss(r), 10, 4)}")
        print(f"  Tokens/s:    {_fmt(_tokens_per_s(r), 10, 0)}")
        print(f"  Step ms:     {_fmt(_step_ms(r), 10, 0)}")
        print(f"  Peak VRAM:   {_fmt(_peak_mem(r), 10, 0)} MB")

        # Gate series
        gate_series = _get(r, "metrics", "sil_gate_series") or []
        if gate_series:
            milestones = [0, len(gate_series)//4, len(gate_series)//2, 3*len(gate_series)//4, len(gate_series)-1]
            milestones = [m for m in milestones if 0 <= m < len(gate_series)]
            print(f"\n  SIL Gate Evolution:")
            print(f"  {'Step':>8}  {'Gate Mean':>10}")
            print(f"  {'-'*22}")
            for m in milestones:
                print(f"  {m+1:>8}  {gate_series[m]:>10.5f}")

        # Aux loss series (derive from total_loss_series - loss_series if not stored directly)
        aux_series = _get(r, "metrics", "sil_aux_loss_series") or []
        if not aux_series:
            tl = _get(r, "metrics", "total_loss_series") or []
            ll = _get(r, "metrics", "loss_series") or []
            if len(tl) == len(ll) and len(tl) > 0:
                aux_series = [t - l for t, l in zip(tl, ll)]
        if aux_series:
            milestones = [0, len(aux_series)//4, len(aux_series)//2, 3*len(aux_series)//4, len(aux_series)-1]
            milestones = [m for m in milestones if 0 <= m < len(aux_series)]
            print(f"\n  SIL Aux Loss Evolution:")
            print(f"  {'Step':>8}  {'AuxLoss':>10}")
            print(f"  {'-'*22}")
            for m in milestones:
                print(f"  {m+1:>8}  {aux_series[m]:>10.6f}")

        print()


def section_asr_diagnostics(reports: list[dict[str, Any]]) -> None:
    """Print ASR diagnostic details."""
    _bar("TRACK 3: ASR DIAGNOSTICS")

    asr_runs = [r for r in reports
                if _arch(r) == "asr" and _device(r) == "cuda" and _steps(r) >= 2000]

    if not asr_runs:
        print("  (no ASR CUDA ≥2000-step reports found)")
        return

    for r in asr_runs:
        ds = _dataset(r)
        print(f"  Dataset: {ds}")
        print(f"  File: {r['_file']}")
        print(f"  Loss (last): {_fmt(_loss_last(r), 10, 4)}")
        print(f"  Loss (best): {_fmt(_loss_best(r), 10, 4)}")
        print(f"  Eval loss:   {_fmt(_eval_loss(r), 10, 4)}")
        print(f"  Tokens/s:    {_fmt(_tokens_per_s(r), 10, 0)}")
        print(f"  Step ms:     {_fmt(_step_ms(r), 10, 0)}")
        print(f"  Fwd ms:      {_fmt(_fwd_ms(r), 10, 0)}")
        print(f"  Peak VRAM:   {_fmt(_peak_mem(r), 10, 0)} MB")

        # ASR aux loss series (derive from total_loss_series - loss_series if not stored directly)
        aux_series = _get(r, "metrics", "asr_aux_loss_series") or []
        if not aux_series:
            tl = _get(r, "metrics", "total_loss_series") or []
            ll = _get(r, "metrics", "loss_series") or []
            if len(tl) == len(ll) and len(tl) > 0:
                aux_series = [t - l for t, l in zip(tl, ll)]
        if aux_series:
            milestones = [0, len(aux_series)//4, len(aux_series)//2, 3*aux_series.__len__()//4, len(aux_series)-1]
            milestones = [m for m in milestones if 0 <= m < len(aux_series)]
            print(f"\n  ASR Consistency Loss Evolution:")
            print(f"  {'Step':>8}  {'ConsistencyLoss':>16}")
            print(f"  {'-'*28}")
            for m in milestones:
                print(f"  {m+1:>8}  {aux_series[m]:>16.6f}")

        print()


def section_cross_track(baselines: dict[str, dict[str, Any]]) -> None:
    """Print cross-track comparison."""
    _bar("CROSS-TRACK COMPARISON")

    if not baselines:
        print("  (no baseline data available)")
        return

    gqa = baselines.get("gqa", {})
    gqa_loss = gqa.get("eval_loss") or gqa.get("loss_last")
    gqa_toks = gqa.get("tokens_per_s")
    gqa_step = gqa.get("step_ms")

    tracks = [
        ("MIG", "Track 1", "mig", "Context Dilution"),
        ("SIL", "Track 2", "sil", "Latent Blurring"),
        ("ASR", "Track 3", "asr", "Spatial Fragility"),
    ]

    header = (
        f"{'Architecture':>14}  {'Track':>8}  {'Fixes':>20}  "
        f"{'Loss gap':>9}  {'Loss gap%':>10}  {'Speed vs GQA':>12}  "
        f"{'Train overhead':>15}  {'Infer overhead':>15}"
    )
    print(header)
    print("-" * len(header))

    for name, track, key, fixes in tracks:
        d = baselines.get(key, {})
        loss = d.get("eval_loss") or d.get("loss_last")
        toks = d.get("tokens_per_s")
        step = d.get("step_ms")

        gap = (loss - gqa_loss) if (loss and gqa_loss) else None
        gap_pct = (gap / gqa_loss * 100) if (gap is not None and gqa_loss) else None
        speed_ratio = (toks / gqa_toks) if (toks and gqa_toks) else None
        train_oh = (step / gqa_step) if (step and gqa_step) else None

        infer = {
            "mig": "~1x (gate cheap)",
            "sil": "~1.4x (rule path)",
            "asr": "0x (identical)",
        }.get(key, "?")

        print(
            f"{name:>14}  {track:>8}  {fixes:>20}  "
            f"{_fmt(gap, 9, 4) if gap else '-':>9}  "
            f"{f'{gap_pct:+.2f}%' if gap_pct else '-':>10}  "
            f"{f'{speed_ratio:.2f}x' if speed_ratio else '-':>12}  "
            f"{f'{train_oh:.2f}x' if train_oh else '-':>15}  "
            f"{infer:>15}"
        )


def section_report_inventory(reports: list[dict[str, Any]]) -> None:
    """Print summary inventory of all reports."""
    _bar("REPORT INVENTORY")

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        ds = _dataset(r)
        by_dataset.setdefault(ds, []).append(r)

    print(f"  Total reports: {len(reports)}")
    print()
    for ds in sorted(by_dataset):
        runs = by_dataset[ds]
        cuda_runs = [r for r in runs if _device(r) == "cuda"]
        cpu_runs = [r for r in runs if _device(r) == "cpu"]
        with_eval = [r for r in runs if _eval_loss(r) is not None]
        print(f"  {ds}: {len(runs)} total ({len(cuda_runs)} CUDA, {len(cpu_runs)} CPU, {len(with_eval)} with eval_loss)")


# ──────────────────── main ────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    reports = _load_reports(args.reports_dir)
    if not reports:
        print(f"No reports found in {args.reports_dir.resolve()}")
        return 1

    print(f"Loaded {len(reports)} reports from {args.reports_dir.resolve()}")

    section_report_inventory(reports)
    baselines = section_tinystories_baselines(reports)
    section_mig_sweep(reports, "sparse_signal")
    section_mig_sweep(reports, "poisoned_needle")
    section_sil_diagnostics(reports)
    section_asr_diagnostics(reports)
    section_cross_track(baselines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
