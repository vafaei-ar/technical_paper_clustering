"""Keep manuscript figure titles outside their exported artwork."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper_figures"


@pytest.mark.parametrize("number,stem", [
    (1, "fig1_workflow"),
    (2, "fig2_model_selection"),
    (3, "fig3_null_reference"),
    (4, "fig4_phenotypes"),
    (5, "fig5_robustness"),
])
def test_figure_exports_have_no_super_title(number, stem):
    """Panels remain titled, but Figure N/title belongs to manuscript caption."""
    tree = ast.parse((FIG_DIR / f"{stem}.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "figure_title"
    ]
    assert not calls, f"Figure {number} must not render a manuscript-style supertitle"
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "_crop_top_inches"
        for node in ast.walk(tree)
    ), f"Figure {number} must crop the now-empty top band"


def test_shared_figure_export_trims_only_top_margin(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(FIG_DIR))
    import clustro_style as style

    fig, ax = style.canvas(3.0, 2.0)
    # Put content inside the retained portion of the canvas.
    ax.text(1.0, 1.0, "Panel A")
    fig._crop_top_inches = 0.48
    output = tmp_path / "cropped.png"
    style.save(fig, str(output), dpi=100)
    with Image.open(output) as image:
        assert abs(image.width - 300) <= 4
        assert abs(image.height - 152) <= 4
