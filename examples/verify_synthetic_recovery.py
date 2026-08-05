from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score


def latest_run(output_root: str | Path) -> Path:
    root = Path(output_root)
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return runs[-1]


def verify_recovery(
    config_path: str | Path,
    run_dir: str | Path | None = None,
    reduction: str = "pca",
    clusterer: str = "kmeans",
    k: int = 3,
    seed: int = 11,
    minimum_ari: float = 0.80,
) -> dict:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    run_path = Path(run_dir) if run_dir is not None else latest_run(config["output_dir"])
    truth = pd.read_parquet(config["input_path"])[
        [config["id_column"], "cluster_truth"]
    ]
    assignments = pd.read_parquet(run_path / "cluster_assignments.parquet")
    selected = assignments[
        (assignments["reduction"] == reduction)
        & (assignments["clusterer"] == clusterer)
        & (assignments["k"] == int(k))
        & (assignments["seed"] == int(seed))
    ][[config["id_column"], "cluster"]]

    if selected.empty:
        raise ValueError(
            "No assignments matched the requested model: "
            f"reduction={reduction}, clusterer={clusterer}, k={k}, seed={seed}"
        )
    if selected[config["id_column"]].duplicated().any():
        raise ValueError("Selected assignments contain duplicate identifiers")

    merged = truth.merge(
        selected,
        on=config["id_column"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise ValueError(
            f"Only {len(merged)} of {len(truth)} synthetic rows matched assignments"
        )

    ari = float(adjusted_rand_score(merged["cluster_truth"], merged["cluster"]))
    passed = bool(ari >= float(minimum_ari))
    report = {
        "dataset": config["dataset_name"],
        "run_dir": str(run_path),
        "model": {
            "reduction": reduction,
            "clusterer": clusterer,
            "k": int(k),
            "seed": int(seed),
        },
        "n_rows": int(len(merged)),
        "adjusted_rand_index": ari,
        "minimum_acceptable_ari": float(minimum_ari),
        "passes_verification": passed,
        "truth_excluded_from_clustering_features": bool(
            "cluster_truth" not in config["features"]["primary"]
        ),
    }

    output_dir = run_path / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "reduction": reduction,
                "clusterer": clusterer,
                "k": int(k),
                "seed": int(seed),
                "n_rows": len(merged),
                "adjusted_rand_index": ari,
                "minimum_acceptable_ari": float(minimum_ari),
                "passes_verification": passed,
            }
        ]
    ).to_csv(output_dir / "synthetic_recovery_verification.csv", index=False)
    (output_dir / "synthetic_recovery_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if not passed:
        raise SystemExit(
            f"Synthetic recovery verification failed: ARI={ari:.3f} "
            f"< minimum {minimum_ari:.3f}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify recovery of the known synthetic latent groups."
    )
    parser.add_argument("--config", default="examples/synthetic_config.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--reduction", default="pca")
    parser.add_argument("--clusterer", default="kmeans")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--minimum-ari", type=float, default=0.80)
    args = parser.parse_args()

    report = verify_recovery(
        config_path=args.config,
        run_dir=args.run_dir,
        reduction=args.reduction,
        clusterer=args.clusterer,
        k=args.k,
        seed=args.seed,
        minimum_ari=args.minimum_ari,
    )
    print(
        f"Synthetic recovery passed: ARI={report['adjusted_rand_index']:.3f} "
        f"(minimum={report['minimum_acceptable_ari']:.3f})"
    )
    print(Path(report["run_dir"]) / "verification")


if __name__ == "__main__":
    main()
