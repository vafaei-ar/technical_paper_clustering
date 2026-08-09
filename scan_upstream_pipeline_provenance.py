from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Deliberately exclude CSV/parquet/data exports and notebooks. The previous broad
# scanner could ingest generated data and create multi-GB output. This scanner is
# intended to inspect source/configuration/documentation only.
TEXT_SUFFIXES = {".py", ".r", ".R", ".sql", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh"}
SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "env", "node_modules", "data", "datasets", "results",
    "outputs", "output", "logs", "artifacts", "dist", "build", "__pycache__",
}
MAX_FILE_BYTES = 2_000_000
MAX_MATCHES_PER_TOPIC_PER_FILE = 8
CONTEXT_RADIUS = 2

TOPICS = {
    "study_period": [
        r"study[_ -]?period", r"date[_ -]?range", r"start[_ -]?date", r"end[_ -]?date",
        r"admit.*date", r"encounter.*date", r"between\s+['\"]?20\d\d", r">=\s*['\"]?20\d\d",
        r"<=\s*['\"]?20\d\d",
    ],
    "index_event": [
        r"first[_ -]?(encounter|visit|admission|event|sepsis|stroke)", r"first qualifying",
        r"index[_ -]?(encounter|visit|event|admission)", r"drop_duplicates",
        r"duplicated\(", r"groupby\(.*(first|head)", r"sort_values.*(date|time|encounter)",
    ],
    "identifier_semantics": [
        r"\bPATID\b", r"\bENCOUNTERID\b", r"patient[_ -]?id", r"encounter[_ -]?id",
    ],
    "missingness_imputation": [
        r"imput", r"fillna", r"SimpleImputer", r"KNNImputer", r"IterativeImputer",
        r"missing[_ -]?(value|data|rate|ness)", r"isna\(", r"notna\(", r"median\(",
    ],
    "duration_los": [
        r"enc_duration", r"length[_ -]?of[_ -]?stay", r"\bLOS\b", r"prolonged_los",
        r"discharg.*-.*admi", r"discharg.*admi", r"admi.*discharg", r"timedelta",
    ],
    "ed_inpatient": [
        r"emergency department", r"\bED\b", r"inpatient", r"encounter[_ -]?type",
        r"merge.*encounter", r"admission", r"hospitalization",
    ],
    "discharge_codes": [
        r"DISCHARGE_STATUS", r"discharge[_ -]?status", r"disposition",
        r"['\"]OT['\"]", r"['\"]IP['\"]", r"['\"]AM['\"]", r"['\"]NH['\"]", r"['\"]SH['\"]",
    ],
    "inclusion_exclusion": [
        r"inclusion[_ -]?criteria", r"exclusion[_ -]?criteria", r"eligib",
        r"exclude[_ -]?(patient|encounter|record|row)", r"include[_ -]?(patient|encounter|record)",
        r"cohort[_ -]?(definition|selection|criteria)",
    ],
    "demographic_coding": [
        r"\bSEX\b", r"\bRACE\b", r"\bHISPANIC\b", r"RUCA", r"rural.*urban",
        r"preferred[_ -]?language", r"PAT_PREF_LANGUAGE_SPOKEN", r"mapping", r"codebook",
    ],
    "dimensionality_reduction": [
        r"\bUMAP\b", r"t[-_ ]?SNE", r"TSNE", r"Isomap", r"LocallyLinearEmbedding",
        r"KernelPCA", r"TruncatedSVD", r"autoencoder", r"dimensionality[_ -]?reduction",
    ],
}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in SKIP_DIR_NAMES for part in relative.parts[:-1]):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"Dockerfile", "Makefile"}:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _compact(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def scan_repo(label: str, root: Path) -> list[dict]:
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")

    rows: list[dict] = []
    compiled_topics = {
        topic: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for topic, patterns in TOPICS.items()
    }

    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = text.splitlines()
        per_topic_count: Counter[str] = Counter()

        for line_no, line in enumerate(lines, start=1):
            for topic, patterns in compiled_topics.items():
                if per_topic_count[topic] >= MAX_MATCHES_PER_TOPIC_PER_FILE:
                    continue
                if not any(pattern.search(line) for pattern in patterns):
                    continue

                start = max(1, line_no - CONTEXT_RADIUS)
                end = min(len(lines), line_no + CONTEXT_RADIUS)
                context = " | ".join(lines[start - 1 : end])
                rows.append(
                    {
                        "repository": label,
                        "topic": topic,
                        "file": str(path.relative_to(root)),
                        "line": line_no,
                        "matched_text": _compact(line, 400),
                        "context": _compact(context, 1200),
                    }
                )
                per_topic_count[topic] += 1

    return rows


def write_summary(rows: list[dict], output: Path) -> Path:
    grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"matches": 0, "files": set()})
    for row in rows:
        key = f"{row['repository']}::{row['topic']}"
        grouped[key]["matches"] = int(grouped[key]["matches"]) + 1
        grouped[key]["files"].add(row["file"])

    summary = []
    for key, info in sorted(grouped.items()):
        repository, topic = key.split("::", 1)
        summary.append(
            {
                "repository": repository,
                "topic": topic,
                "matches": info["matches"],
                "files": sorted(info["files"]),
            }
        )

    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused, size-bounded scan of source repositories for manuscript-relevant provenance evidence."
    )
    parser.add_argument("--sepsis-repo", required=True)
    parser.add_argument("--stroke-repo", required=True)
    parser.add_argument(
        "--clustering-repo",
        default=".",
        help="Current clustering repository; scanned for earlier dimensionality-reduction experiments and related evidence.",
    )
    parser.add_argument(
        "--output",
        default="results/author_revision_materials/upstream_pipeline_provenance_focused.csv",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    rows.extend(scan_repo("sepsis", Path(args.sepsis_repo)))
    rows.extend(scan_repo("stroke", Path(args.stroke_repo)))

    clustering_rows = scan_repo("clustering", Path(args.clustering_repo))
    rows.extend(row for row in clustering_rows if row["topic"] == "dimensionality_reduction")

    # Deterministic ordering makes review/diffs straightforward.
    rows.sort(key=lambda row: (row["repository"], row["topic"], row["file"], int(row["line"])))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repository", "topic", "file", "line", "matched_text", "context"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = write_summary(rows, output)
    print(f"Wrote {len(rows):,} focused provenance matches to {output}")
    print(f"Wrote scan summary to {summary_path}")
    print(f"Output size: {output.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
