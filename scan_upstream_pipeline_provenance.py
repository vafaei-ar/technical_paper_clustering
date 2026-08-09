from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

TEXT_SUFFIXES = {".py", ".ipynb", ".r", ".R", ".sql", ".md", ".txt", ".yaml", ".yml", ".json", ".csv"}

TOPICS = {
    "study_period": [r"study period", r"date range", r"start_date", r"end_date", r"201\d", r"202\d"],
    "index_event": [r"first encounter", r"first qualifying", r"index encounter", r"index event", r"drop_duplicates", r"sort_values"],
    "identifier_semantics": [r"PATID", r"ENCOUNTERID", r"patient id", r"encounter id"],
    "missingness_imputation": [r"imput", r"missing", r"fillna", r"SimpleImputer", r"KNNImputer", r"IterativeImputer", r"median"],
    "duration_los": [r"enc_duration", r"length of stay", r"length_of_stay", r"LOS", r"discharge.*admit", r"admit.*discharge"],
    "ed_inpatient": [r"emergency", r"\bED\b", r"inpatient", r"IP encounter", r"merge.*encounter"],
    "discharge_codes": [r"DISCHARGE_STATUS", r"discharge status", r"\bOT\b", r"\bAM\b", r"\bNH\b", r"\bSH\b"],
    "inclusion_exclusion": [r"inclusion", r"exclusion", r"eligible", r"exclude", r"filter", r"cohort"],
}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {"Dockerfile", "Makefile"}:
            yield path


def scan_repo(label: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = text.splitlines()
        for topic, patterns in TOPICS.items():
            compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
            for line_no, line in enumerate(lines, start=1):
                if any(p.search(line) for p in compiled):
                    start = max(1, line_no - 2)
                    end = min(len(lines), line_no + 2)
                    context = " | ".join(lines[start - 1 : end]).strip()
                    rows.append(
                        {
                            "repository": label,
                            "topic": topic,
                            "file": str(path.relative_to(root)),
                            "line": line_no,
                            "matched_text": line.strip()[:500],
                            "context": context[:1500],
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan upstream cohort-construction repositories for manuscript-relevant provenance evidence.")
    parser.add_argument("--sepsis-repo", required=True)
    parser.add_argument("--stroke-repo", required=True)
    parser.add_argument("--output", default="results/author_revision_materials/upstream_pipeline_provenance.csv")
    args = parser.parse_args()

    rows = []
    rows.extend(scan_repo("sepsis", Path(args.sepsis_repo)))
    rows.extend(scan_repo("stroke", Path(args.stroke_repo)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["repository", "topic", "file", "line", "matched_text", "context"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} provenance matches to {output}")


if __name__ == "__main__":
    main()
