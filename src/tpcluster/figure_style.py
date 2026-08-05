from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

DPI = 600
FORMATS = ("png", "pdf")

LONG_LABELS = {
    "stroke": {
        0: "Preserved-haematologic, lower-comorbidity",
        1: "Renal-anaemic multimorbidity",
        2: "Hyperglycaemic diabetes",
    },
    "sepsis": {
        0: "Neutrophil-predominant, lower organ-dysfunction burden",
        1: "Immature-granulocyte-high organ dysfunction",
        2: "Eosinophil-lymphocyte-enriched",
    },
}

SHORT_LABELS = {
    "stroke": {0: "Preserved", 1: "Renal–anaemic", 2: "Hyperglycaemic"},
    "sepsis": {
        0: "Neutrophil-predominant",
        1: "IG-high dysfunction",
        2: "Eosinophil–lymphocyte",
    },
}


def apply_publication_style() -> None:
    """Apply compact, journal-ready matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig: plt.Figure,
    stem: str | Path,
    formats: Iterable[str] = FORMATS,
    dpi: int = DPI,
) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        path = stem.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if extension in {"png", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs
