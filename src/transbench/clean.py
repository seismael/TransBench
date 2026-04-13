from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CleanPlan:
    files: list[Path]
    dirs: list[Path]


def _unique_existing(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            out.append(p)
    return out


def build_clean_plan(
    root: Path,
    *,
    reports: bool,
    caches: bool,
    runs: bool,
) -> CleanPlan:
    root = root.resolve()
    files: list[Path] = []
    dirs: list[Path] = []

    if reports:
        reports_dir = root / "reports"
        if reports_dir.exists():
            for p in reports_dir.glob("*.json"):
                files.append(p)
            # keep .gitkeep if present

    if caches:
        dirs.extend(
            [
                root / ".pytest_tmp",
                root / ".pytest_cache",
                root / ".ruff_cache",
                root / ".ipynb_checkpoints",
                root / "modules" / "__pycache__",
                root / "utils" / "__pycache__",
            ]
        )

        # Any __pycache__ under src/
        src_dir = root / "src"
        if src_dir.exists():
            for p in src_dir.rglob("__pycache__"):
                dirs.append(p)

    if runs:
        dirs.append(root / "runs")

    return CleanPlan(files=_unique_existing(files), dirs=_unique_existing(dirs))


def apply_clean_plan(plan: CleanPlan) -> None:
    for f in plan.files:
        f.unlink(missing_ok=True)

    # Remove dirs after files; rmdir if empty else recursive
    for d in plan.dirs:
        if not d.exists():
            continue
        # Prefer fast recursive delete when supported
        for child in sorted(d.rglob("*"), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            except Exception:
                # best-effort; continue
                pass
        try:
            d.rmdir()
        except Exception:
            # best-effort
            pass
