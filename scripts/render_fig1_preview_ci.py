"""Render Figure 1 with verified aggregate cohort counts and no private inputs.

This workflow-only preview renders the shared-style Python figure with the
already published cohort totals; it never loads EHR files.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("fig1_preview"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1 uses these aggregate totals only. Bypass unrelated figure-data imports.
    data = ModuleType("clustro_data")
    data.N_STROKE = 9835
    data.N_SEPSIS = 15842
    sys.modules["clustro_data"] = data

    fig_dir = ROOT / "paper_figures"
    sys.path.insert(0, str(fig_dir))
    spec = importlib.util.spec_from_file_location("fig1_preview_renderer",
                                                  fig_dir / "fig1_workflow.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Figure 1")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("PNG:", mod.build(str(args.outdir)))

    # Save the exact same source scene with a vector backend for publication.
    import clustro_style as style
    original = mod.save

    def save_pdf(fig, requested_path, dpi=300):
        path = os.path.splitext(requested_path)[0] + ".pdf"
        return style.save(fig, path, dpi=dpi)

    mod.save = save_pdf
    try:
        print("PDF:", mod.build(str(args.outdir)))
    finally:
        mod.save = original


if __name__ == "__main__":
    main()
