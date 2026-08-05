from __future__ import annotations

import generate_manuscript_outputs as manuscript_outputs

from regenerate_manuscript_outputs import REVISED_SEPSIS_LABEL, apply_revised_labels
from tpcluster.figure_style import LONG_LABELS


def test_revised_sepsis_label_is_consistent() -> None:
    apply_revised_labels()
    assert manuscript_outputs.LONG_LABELS["sepsis"][0] == REVISED_SEPSIS_LABEL
    assert LONG_LABELS["sepsis"][0] == REVISED_SEPSIS_LABEL
    assert "lower-acuity" not in REVISED_SEPSIS_LABEL
