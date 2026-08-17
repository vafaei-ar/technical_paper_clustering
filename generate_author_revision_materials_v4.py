from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_author_revision_materials import _latest_run, _load_yaml, make_compact_supplement_tables
from generate_author_revision_materials_v3 import make_concise_methods_evidence, make_manuscript_demographic_table
from generate_publication_figures import make_model_selection_figure, make_phenotype_heatmap_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready manuscript revision materials without rerunning clustering."
    )
    parser.add_argument("--stroke-config", default="configs/stroke.yaml")
    parser.add_argument("--sepsis-config", default="configs/sepsis.yaml")
    parser.add_argument("--stroke-run")
    parser.add_argument("--sepsis-run")
    parser.add_argument("--provenance-csv", required=True)
    parser.add_argument("--output-dir", default="results/author_revision_materials_v4")
    args = parser.parse_args()

    stroke_cfg = _load_yaml(Path(args.stroke_config))
    sepsis_cfg = _load_yaml(Path(args.sepsis_config))
    stroke_run = Path(args.stroke_run) if args.stroke_run else _latest_run(stroke_cfg["output_dir"])
    sepsis_run = Path(args.sepsis_run) if args.sepsis_run else _latest_run(sepsis_cfg["output_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    created.append(make_manuscript_demographic_table(stroke_cfg, sepsis_cfg, output_dir))
    created.extend(make_model_selection_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_phenotype_heatmap_figure(stroke_run, sepsis_run, output_dir))
    created.extend(make_compact_supplement_tables(stroke_run, sepsis_run, output_dir))
    created.extend(make_concise_methods_evidence(Path(args.provenance_csv), output_dir))

    manifest = {
        "stroke_run": str(stroke_run),
        "sepsis_run": str(sepsis_run),
        "provenance_csv": str(args.provenance_csv),
        "created": [str(path) for path in created],
        "note": (
            "Publication revision products generated from existing cohort and manuscript-output tables. "
            "Clustering models and preprocessing are not refit."
        ),
    }
    manifest_path = output_dir / "author_revision_materials_v4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(created)} products to {output_dir}")
    print(manifest_path)


if __name__ == "__main__":
    main()
