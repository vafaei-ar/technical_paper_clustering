from __future__ import annotations

import argparse
from pathlib import Path

from tpcluster.paper_heatmaps import make_combined_heatmap_figure
from tpcluster.reporting_figures import (
    make_combined_pca_figure,
    make_effect_size_figure,
    make_workflow_figure,
    write_figure_manifest,
)


def latest_complete_run(root: str | Path) -> Path:
    root = Path(root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "manuscript_outputs" / "analysis_manifest.json").exists()
        and (path / "manuscript_outputs" / "tables" / "table_cluster_sizes.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed manuscript run found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact publication figures for the stroke and sepsis paper."
    )
    parser.add_argument("--stroke-config", default="configs/stroke.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--stroke-run")
    parser.add_argument("--sepsis-run")
    parser.add_argument("--output-dir", default="results/paper_figures/latest")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    stroke_run = Path(args.stroke_run) if args.stroke_run else latest_complete_run("results/stroke")
    sepsis_run = Path(args.sepsis_run) if args.sepsis_run else latest_complete_run("results/sepsis")

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.extend(make_workflow_figure(figure_dir, args.formats))
    generated.extend(
        make_combined_pca_figure(
            args.stroke_config,
            args.sepsis_config,
            stroke_run,
            sepsis_run,
            figure_dir,
            args.formats,
        )
    )
    generated.extend(
        make_combined_heatmap_figure(
            stroke_run,
            sepsis_run,
            figure_dir,
            args.formats,
        )
    )
    generated.extend(
        make_effect_size_figure(
            stroke_run,
            sepsis_run,
            figure_dir,
            args.formats,
        )
    )
    manifest = write_figure_manifest(
        output_dir,
        stroke_run,
        sepsis_run,
        generated,
    )
    print(f"Stroke run: {stroke_run}")
    print(f"Sepsis run: {sepsis_run}")
    print(f"Generated {len(generated)} publication figure files under {figure_dir}")
    print(f"Figure manifest: {manifest}")


if __name__ == "__main__":
    main()
