from __future__ import annotations

import generate_manuscript_outputs as manuscript_outputs


REVISED_SEPSIS_LABEL = "Neutrophil-predominant, lower organ-dysfunction burden"


def apply_revised_labels() -> None:
    """Apply the reviewer-driven sepsis label revision before output generation."""
    manuscript_outputs.LONG_LABELS["sepsis"][0] = REVISED_SEPSIS_LABEL


def main() -> None:
    apply_revised_labels()
    manuscript_outputs.main()


if __name__ == "__main__":
    main()
