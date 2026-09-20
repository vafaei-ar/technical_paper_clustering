"""
Figure 5 -- Robustness differs across clustering specifications.

Panel A: silhouette and full-refit ARI across k.
Panel B: raw-feature vs PCA representation.
Panel C: balanced-block weighting.
Panel D: WBC redundancy sensitivity in sepsis.
"""
from __future__ import annotations

import os

import numpy as np

from clustro_style import *
import clustro_data as D

W, H = 15.0, 10.55
PW = W - 0.80
HALF = (PW - 0.36) / 2

BAR_H = 0.60


def _subhead(ax, x, y, w, label, h=0.38):
    rbox(ax, x, y, w, h, fc="#E8F0F9", ec="none", r=0.07, z=2)
    txt(ax, x + w / 2, y + h / 2, label, size=9.6, weight="bold", color=INK,
        ha="center", z=5)
    return y + h


def _note(ax, x, y, w, h, heading, body):
    rbox(ax, x, y, w, h, fc="#F2F7FC", ec="#D9E6F2", lw=0.9, r=0.10, z=2)
    end = para(ax, x + 0.16, y + 0.14, heading, w - 0.32, size=8.8,
               color=INK, weight="bold")
    para(ax, x + 0.16, end + 0.12, body, w - 0.32, size=8.3, color=TEXT)


def _bar_panel(fig, ax, x0, y0, w, h, letter, title, subtitle, header,
               items, note=None, ylabel_w=1.30):
    panel(ax, x0, y0, w, h, letter=letter, title=title, subtitle=subtitle,
          sub_size=9.2)

    note_w = 2.05 if note else 0.0
    chart_x = x0 + 0.34
    chart_w = w - 0.68 - (note_w + 0.18 if note else 0.0)

    top = _subhead(ax, chart_x, y0 + 1.30, chart_w, header)

    bx, by = chart_x + ylabel_w, top + 0.30
    bw, bh = chart_w - ylabel_w - 0.55, h - (by - y0) - 0.88
    axb = axes_in(fig, bx, by, bw, bh)
    tidy_axes(axb, grid="x", spines=("bottom",))
    axb.set_xlim(0, 1.0)
    axb.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axb.set_ylim(len(items) - 0.5, -0.5)
    axb.set_yticks([])
    axb.set_xlabel("ARI (1 = identical)", fontsize=14.0, color=TEXT, labelpad=5)

    for i, (label, value, color) in enumerate(items):
        axb.barh(i, value, height=BAR_H, color=color, zorder=4)
        axb.text(value + 0.022, i, f"{value:.3f}", va="center", ha="left",
                 fontsize=14.8, color=TEXT, zorder=5)
        axb.text(-0.028, i, label, va="center", ha="right", fontsize=14.1,
                 color=TEXT, transform=axb.get_yaxis_transform(), zorder=5)

    if note:
        nx = x0 + w - 0.34 - note_w
        _note(ax, nx, top + 0.18, note_w, h - (top - y0) - 0.60, *note)
    return axb


def panel_a(fig, ax):
    x0, y0, w, h = 0.40, 0.88, HALF, 4.60
    panel(ax, x0, y0, w, h, letter="A", title="$k$ sensitivity",
          subtitle="Cluster count changes the separation-stability tradeoff "
                   "differently in stroke and sepsis", sub_size=9.2)

    sw = (w - 0.64 - 0.22) / 2
    max_sil = max(max(D.SIL_BY_K["Stroke"]), max(D.SIL_BY_K["Sepsis"]))
    sil_top = max(0.20, np.ceil((max_sil + 0.01) * 20) / 20)
    specs = [
        (x0 + 0.32, "Silhouette score", D.SIL_BY_K, (0.0, sil_top),
         np.linspace(0, sil_top, 6), "Silhouette score", "%.3f"),
        (x0 + 0.32 + sw + 0.22, "Mean full-refit ARI", D.ARI_BY_K, (0.0, 1.05),
         [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "Mean full-refit ARI", "%.3f"),
    ]

    for sx, header, data, ylim, yticks, ylab, fmt in specs:
        top = _subhead(ax, sx, y0 + 1.42, sw, header)
        axl = axes_in(fig, sx + 0.62, top + 0.24, sw - 0.78,
                      h - (top - y0) - 1.14)
        tidy_axes(axl, grid="none", spines=("left", "bottom"))
        axl.set_xlim(1.62, 4.38)
        axl.set_ylim(*ylim)
        axl.set_xticks(D.K_GRID)
        axl.set_yticks(yticks)
        axl.set_xlabel("Number of clusters ($k$)", fontsize=13.8,
                       color=TEXT, labelpad=4)
        axl.set_ylabel(ylab, fontsize=13.8, color=TEXT, labelpad=4)
        axl.tick_params(labelsize=13.1)

        axl.axvspan(D.K_SELECTED - 0.28, D.K_SELECTED + 0.28,
                    color="#EAF2FA", zorder=1)
        axl.axvline(D.K_SELECTED, color="#9AA9B8", ls=(0, (4, 3)),
                    lw=1.0, zorder=2)

        for cohort, color in (("Stroke", BLUE), ("Sepsis", ORANGE)):
            ys = data[cohort]
            axl.plot(D.K_GRID, ys, "-o", color=color, lw=1.8, ms=7,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            for k, v in zip(D.K_GRID, ys):
                off = 0.055 * (ylim[1] - ylim[0]) if cohort == "Stroke" \
                    else -0.075 * (ylim[1] - ylim[0])
                axl.annotate(fmt % v, (k, v + off), ha="center",
                             va="bottom" if cohort == "Stroke" else "top",
                             fontsize=13.3, color=color, weight="bold", zorder=6)

    ly = y0 + h - 0.16
    lx = x0 + (w - 4.90) / 2
    for cohort, color, n in (("Stroke", BLUE, D.N_STROKE),
                             ("Sepsis", ORANGE, D.N_SEPSIS)):
        ax.plot([lx, lx + 0.46], [ly, ly], color=color, lw=1.8, zorder=4)
        ax.plot([lx + 0.23], [ly], marker="o", ms=6.5, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        txt(ax, lx + 0.60, ly, f"{cohort} ($n$ = {n:,})", size=9.4,
            color=TEXT, va="center")
        lx += 2.55


def build(outdir="."):
    use_style()
    fig, ax = canvas(W, H)
    figure_title(
        ax, 0.42, 0.48,
        "Figure 5. Robustness differs across clustering specifications"
    )

    panel_a(fig, ax)

    _bar_panel(
        fig, ax, 0.40 + HALF + 0.36, 0.88, HALF, 4.60,
        "B", "Representation sensitivity",
        "Raw features vs. PCA give near-identical solutions",
        "ARI vs. primary PCA $k$ = 3 solution",
        [("Stroke", D.ARI_RAW_VS_PCA["Stroke"], BLUE),
         ("Sepsis", D.ARI_RAW_VS_PCA["Sepsis"], ORANGE)],
        note=("Minimal effect of representation",
              "Raw features give nearly identical assignments in both cohorts."),
    )

    _bar_panel(
        fig, ax, 0.40, 5.68, HALF, 4.60,
        "C", "Balanced-block weighting",
        "Moderate effect in stroke, substantial in sepsis",
        "ARI vs. primary (unweighted) solution",
        [("Stroke", D.ARI_BALANCED_BLOCK["Stroke"], BLUE),
         ("Sepsis", D.ARI_BALANCED_BLOCK["Sepsis"], ORANGE)],
        note=("Much higher sensitivity in sepsis",
              "Weighting substantially changes sepsis assignments; stroke "
              "is less affected."),
    )

    _bar_panel(
        fig, ax, 0.40 + HALF + 0.36, 5.68, HALF, 4.60,
        "D", "WBC redundancy sensitivity in sepsis",
        "Sepsis is sensitive to WBC feature coding",
        "ARI vs. primary sepsis solution ($k$ = 3)",
        [(k, v, ORANGE) for k, v in D.ARI_WBC_REDUNDANCY.items()],
        note=("Sepsis depends on WBC specification",
              "Dropping WBC counts or percentages substantially changes the "
              "sepsis clusters."),
        ylabel_w=1.55,
    )

    return save(fig, os.path.join(outdir, "fig5_robustness.png"))


if __name__ == "__main__":
    print(build())
