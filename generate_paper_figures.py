from __future__ import annotations

import argparse
from pathlib import Path

from tpcluster.reporting_figures import (
    make_combined_heatmap_figure,
    make_combined_pca_figure,
    make_effect_size_figure,
    make_workflow_figure,
    write_figure_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact publication figures for the stroke and sepsis paper."
    )
    parser.add_argument("--stroke-config", default="configs/stroke.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--stroke-run", required=True)
    parser.add_argument("--sepsis-run", required=True)
    parser.add_argument("--output-dir", default="results/paper_figures/latest")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.extend(make_workflow_figure(figure_dir, args.formats))
    generated.extend(
        make_combined_pca_figure(
            args.stroke_config,
            args.sepsis_config,
            args.stroke_run,
            args.sepsis_run,
            figure_dir,
            args.formats,
        )
    )
    generated.extend(
        make_combined_heatmap_figure(
            args.stroke_run,
            args.sepsis_run,
            figure_dir,
            args.formats,
        )
    )
    generated.extend(
        make_effect_size_figure(
            args.stroke_run,
            args.sepsis_run,
            figure_dir,
            args.formats,
        )
    )
    manifest = write_figure_manifest(
        output_dir,
        args.stroke_run,
        args.sepsis_run,
        generated,
    )
    print(f"Generated {len(generated)} publication figure files under {figure_dir}")
    print(f"Figure manifest: {manifest}")


if __name__ == "__main__":
    main()
