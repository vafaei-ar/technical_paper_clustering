"""Verify Figure 1 renders without requiring any private cohort datasets."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def test_fig1_renders_with_synthetic_cohort_counts(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "paper_figures"
    # The diagram uses counts only; its data loader need not read private CSVs.
    monkeypatch.setitem(sys.modules, "clustro_data", SimpleNamespace(
        N_STROKE=9835, N_SEPSIS=15842,
    ))
    monkeypatch.syspath_prepend(str(figure_dir))
    spec = importlib.util.spec_from_file_location(
        "fig1_layout_smoke", figure_dir / "fig1_workflow.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = Path(module.build(str(tmp_path)))
    assert output.is_file()
    with Image.open(output) as img:
        assert img.width >= 1500
        assert img.height >= 1000
        assert img.mode in {"RGB", "RGBA"}
    assert output.stat().st_size > 50_000

    assert [row[2] for row in module.DECISION_CELLS] == [
        "Proceed to robustness checks", "Interpret cautiously",
        "Unstable partition", "Weak evidence",
    ]
