from __future__ import annotations

from pathlib import Path

from transbench.clean import apply_clean_plan, build_clean_plan


def test_clean_reports_removes_json(tmp_path: Path):
    root = tmp_path
    (root / "reports").mkdir(parents=True)
    (root / "reports" / ".gitkeep").write_text("")
    (root / "reports" / "a.json").write_text("{}")
    (root / "reports" / "manifest.json").write_text("{}")

    plan = build_clean_plan(root, reports=True, caches=False, runs=False)
    assert any(p.name == "a.json" for p in plan.files)
    assert any(p.name == "manifest.json" for p in plan.files)

    apply_clean_plan(plan)

    assert (root / "reports" / ".gitkeep").exists()
    assert not (root / "reports" / "a.json").exists()
    assert not (root / "reports" / "manifest.json").exists()


def test_clean_caches_removes_pycache(tmp_path: Path):
    root = tmp_path
    d = root / "modules" / "__pycache__"
    d.mkdir(parents=True)
    (d / "x.pyc").write_text("x")

    plan = build_clean_plan(root, reports=False, caches=True, runs=False)
    assert d in plan.dirs
    apply_clean_plan(plan)
    assert not d.exists()
