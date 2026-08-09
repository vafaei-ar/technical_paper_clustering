from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_author_revision_materials import (
    _heatmap_matrix,
    _latest_run,
    _load_yaml,
    _profile_table,
    make_compact_supplement_tables,
    make_demographic_table,
    make_model_selection_figure,
)


def make_demographic_code_audit(stroke_cfg: dict, sepsis_cfg: dict, output_dir: Path) -> Path:
    """Small raw-value audit used to decode demographic variables from upstream codebooks."""
    candidates = [
        "SEX",
        "RACE",
        "HISPANIC",
        "PAT_PREF_LANGUAGE_SPOKEN",
        "RUCA_CODE",
        "RURAL_URBAN",
        "RUCA",
    ]
    rows: list[dict] = []
    for cohort, cfg in [("Stroke", stroke_cfg), ("Sepsis", sepsis_cfg)]:
        frame = pd.read_parquet(cfg["input_path"])
        for column in candidates:
            if column not in frame.columns:
                continue
            counts = frame[column].fillna("Missing").value_counts(dropna=False)
            for raw_value, count in counts.items():
                rows.append(
                    {
                        "cohort": cohort,
                        "variable": column,
                        "raw_value": str(raw_value),
                        "n": int(count),
                        "percent": round(100.0 * count / len(frame), 1),
                    }
                )
    out = output_dir / "demographic_code_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def make_heatmap_figure_v2(stroke_run: Path, sepsis_run: Path, output_dir: Path) -> list[Path]:
    """Create phenotype heatmaps with dedicated colorbar axes and no label overlap."""
    specs = [
        ("A", "Stroke: continuous features", _profile_table(stroke_run, "continuous"), 12),
        ("B", "Stroke: binary features", _profile_table(stroke_run, "binary"), 12),
        ("C", "Sepsis: continuous features", _profile_table(sepsis_run, "continuous"), 16),
        ("D", "Sepsis: binary features", _profile_table(sepsis_run, "binary"), 10),
    ]

    fig = plt.figure(figsize=(11.5, 10.0))
    # Each plot receives its own narrow colorbar column. A deliberately generous
    # gap separates the left colorbar from the right panel's y-axis labels.
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 0.035, 1.0, 0.035],
        left=0.11,
        right=0.96,
        top=0.95,
        bottom=0.08,
        wspace=0.55,
        hspace=0.36,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 2]),
    ]
    caxes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 3]),
    ]

    for ax, cax, (panel, title, profile, top_n) in zip(axes, caxes, specs):
        matrix, value_col = _heatmap_matrix(profile, top_n)
        limit = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy()))))
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)

        ax.set_title(f"{panel}   {title}", loc="left", fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Cluster 0", "Cluster 1", "Cluster 2"], fontsize=8.5)
        ax.tick_params(length=0)

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7.5, length=2)
        cbar.set_label(
            "Median difference / IQR"
            if value_col == "median_difference_iqr"
            else "Standardized prevalence difference",
            fontsize=7.5,
            labelpad=4,
        )

    fig.text(
        0.5,
        0.025,
        "Color denotes cohort-relative enrichment or depletion. Clinical interpretation is based on original-scale profiles; processed-scale values support reproducibility.",
        ha="center",
        fontsize=8.5,
    )

    outputs: list[Path] = []
    for ext in ("png", "pdf"):
        path = output_dir / f"figure3_phenotype_heatmaps_compact_v2.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final pre-manuscript author-revision products without refitting clustering models."
    )
    parser.add_argument("--stroke-config", default="configs/stroke.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--stroke-run")
    parser.add_argument("--sepsis-run")
    parser.add_argument("--output-dir", default="results/author_revision_materials_v2")
    args = parser.parse_args()

    stroke_cfg = _load_yaml(Path(args.stroke_config))
    sepsis_cfg = _load_yaml(Path(args.sepsis_config))
    stroke_run = Path(args.stroke_run) if args.stroke_run else _latest_run(stroke_cfg["output_dir"])
    sepsis_run = Path(args.sepsis_run) if args.sepsis_run else _latest_run(sepsis_cfg["output_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    created.append(make_demographic_table(stroke_cfg, sepsis_cfg, output_dir))
    created.append(make_demographic_code_audit(stroke_cfg, sepsis_cfg, output_dir))
    created.extend(make_model_selection_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_heatmap_figure_v2(stroke_run, sepsis_run, output_dir))
    created.extend(make_compact_supplement_tables(stroke_run, sepsis_run, output_dir))

    manifest = {
        "stroke_run": str(stroke_run),
        "sepsis_run": str(sepsis_run),
        "created": [str(path) for path in created],
        "note": "These products summarize frozen cohort inputs and existing analysis outputs. Clustering models are not refit.",
    }
    manifest_path = output_dir / "author_revision_materials_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(created)} products to {output_dir}")
    print(manifest_path)


if __name__ == "__main__":
    main()
