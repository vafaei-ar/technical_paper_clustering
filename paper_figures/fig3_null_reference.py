"""
Figure 3 -- Null-reference benchmarking distinguishes robust structure from
plausible artifacts.

Panels A/B: observed silhouette against the Gaussian-copula null distribution.
Panel C: decision map (separation vs reproducibility) + key takeaways.
"""
from __future__ import annotations

import os

import numpy as np

from clustro_style import *
import clustro_data as D

W, H = 15.0, 11.60
PW = W - 0.80
HALF = (PW - 0.36) / 2

NULL_FILL = "#C8D6E4"
NULL_LINE = "#7C8A99"


def _null_panel(fig, ax_c, ax_top, x0, y0, w, h, letter, cohort, subtitle, spec,
                obs_color, verdict, verdict_fill, verdict_edge, verdict_icon,
                verdict_color, xticks, note_x=0.60, obs_dx=0.0):
    panel(ax_c, x0, y0, w, h, letter=letter, title=cohort, subtitle=subtitle)

    px, py, pw, ph = x0 + 1.00, y0 + 1.05, w - 1.45, h - 2.05
    ax = axes_in(fig, px, py, pw, ph)
    tidy_axes(ax, grid="none", spines=("left", "bottom"))
    ax.tick_params(labelsize=14.0)

    x, dens = D.null_density(spec)
    ax.fill_between(x, dens, color=NULL_FILL, alpha=0.85, lw=0, zorder=2)
    ax.plot(x, dens, color=NULL_FILL, lw=1.0, zorder=3)

    ymax = dens.max() * 1.32
    step = xticks[1] - xticks[0]
    ax.set_xlim(min(xticks[0] - step * 0.55, x.min()),
                max(xticks[-1] + step * 0.42, x.max()))
    ax.set_ylim(0, ymax)
    ax.set_xticks(xticks)
    ax.set_xlabel("Silhouette score", fontsize=15.6, color=TEXT, labelpad=6)
    ax.set_ylabel("Density", fontsize=15.6, color=TEXT, labelpad=6)

    ax.plot([spec["mean"]] * 2, [0, ymax * 0.80], color=NULL_LINE,
            ls=(0, (5, 3)), lw=1.8, zorder=4)
    ax.plot([spec["observed"]] * 2, [0, ymax * 0.80], color=obs_color,
            ls=(0, (5, 3)), lw=2.0, zorder=5)

    ax.text(spec["mean"], ymax * 0.845, f"Null mean = {spec['mean']:.3f}",
            ha="center", va="bottom", fontsize=13.8, color=TEXT, zorder=6)
    ax.text(spec["mean"], ymax * 0.835, "(Gaussian-copula)", ha="center",
            va="top", fontsize=13.8, color=TEXT, zorder=6)
    ax.text(spec["observed"] + obs_dx, ymax * 0.845,
            f"Observed = {spec['observed']:.3f}", ha="center", va="bottom",
            fontsize=14.1, color=obs_color, weight="bold", zorder=6)

    nx, ny = px + note_x * pw, py + 0.80
    segments = [
        [(f"{spec['n_ge']}/{spec['n_rep']}", "bold"), (" null replicates", "normal")],
        [(">= observed", "normal")],
        [("empirical p = ", "normal"), (f"{spec['p']:.3f}", "bold")],
    ]
    for line in segments:
        cx = nx
        for piece, wt in line:
            txt(ax_top, cx, ny, piece, size=9.2, color=TEXT, weight=wt)
            cx += measure(fig, piece, 9.2, wt)
        ny += 0.215

    bw, bh = 2.40, 0.95
    bx, by = x0 + w - bw - 0.34, y0 + 2.42
    rbox(ax_top, bx, by, bw, bh, fc=verdict_fill, ec=verdict_edge, lw=1.1,
         r=0.10, z=2)
    verdict_icon(ax_top, bx + 0.32, by + bh / 2, 0.32)
    block_h = text_height(verdict, bw - 0.76, 8.8)
    para(ax_top, bx + 0.60, by + (bh - block_h) / 2, verdict, bw - 0.76,
         size=8.8, color=verdict_color, weight="bold", z=7)

    ly = y0 + h - 0.38
    lsize = 8.6
    labels = ["Null distribution (Gaussian-copula)", "Null mean", "Observed"]
    for _ in range(8):
        need = sum(measure(fig, t, lsize) for t in labels) + 0.30 + 0.36 * 2 + 1.35
        if need <= w - 1.1:
            break
        lsize *= 0.94
    lx = x0 + 0.70
    sbox(ax_c, lx, ly - 0.10, 0.32, 0.20, fc=NULL_FILL, ec="none", z=4)
    txt(ax_c, lx + 0.44, ly, labels[0], size=lsize, color=MUTED, va="center")
    lx += 0.44 + measure(fig, labels[0], lsize) + 0.45
    ax_c.plot([lx, lx + 0.38], [ly, ly], color=NULL_LINE, ls=(0, (4, 2.5)),
              lw=2.0, zorder=4)
    txt(ax_c, lx + 0.50, ly, labels[1], size=lsize, color=MUTED, va="center")
    lx += 0.50 + measure(fig, labels[1], lsize) + 0.45
    ax_c.plot([lx, lx + 0.38], [ly, ly], color=obs_color, ls=(0, (4, 2.5)),
              lw=2.0, zorder=4)
    txt(ax_c, lx + 0.50, ly, labels[2], size=lsize, color=MUTED, va="center")
    return ax


def panel_c(fig, ax):
    x0, y0, w, h = 0.40, 5.72, PW, 5.48
    panel(ax, x0, y0, w, h, letter="C",
          title="Decision map: separation vs reproducibility")

    grid_x = x0 + 2.05
    grid_w = w - 2.60
    gap = 0.20
    cw = (grid_w - gap) / 2
    row_y = [y0 + 1.42, y0 + 3.18]
    row_h = [1.55, 1.40]
    col_x = [grid_x, grid_x + cw + gap]

    rbox(ax, grid_x, y0 + 0.82, grid_w, 0.36,
         fc="#DCEAF7", ec="none", r=0.07, z=2)
    txt(ax, grid_x + grid_w / 2, y0 + 1.00,
        "Cluster separation (vs. null reference)",
        size=12.5, color=INK, ha="center", z=5)

    for x, label in zip(
        col_x,
        ("High (better than null)", "Null-like (no better than null)")
    ):
        txt(ax, x + cw / 2, y0 + 1.28, label,
            size=11.5, weight="bold", color=TEXT, ha="center", z=5)

    ax.text(
        x0 + 0.78,
        (row_y[0] + row_y[1] + row_h[1]) / 2,
        "Reproducibility\n(full-refit stability)",
        rotation=90, ha="center", va="center",
        size=11.5, color=TEXT, zorder=5, linespacing=1.35
    )

    for y, hh, top, bot in zip(
        row_y, row_h, ("Higher", "Lower"), ("(more stable)", "(less stable)")
    ):
        txt(ax, x0 + 1.56, y + hh / 2 - 0.12, top,
            size=11.0, weight="bold", color=TEXT, ha="right", z=5)
        txt(ax, x0 + 1.56, y + hh / 2 + 0.13, bot,
            size=10.0, color=MUTED, ha="right", z=5)

    cells = [
        (0, 0, "Promising", "Higher than null\nand stable",
         FILL_GREEN, "#9AD3B7", "#1B7A50", ("Stroke", BLUE)),
        (0, 1, "Caution", "Null-like separation\nbut stable",
         FILL_ORANGE, "#F0C089", "#D9821B", ("Sepsis", ORANGE)),
        (1, 0, "Unstable", "Higher than null\nbut unstable",
         "#FDE7E9", "#F3BFC5", RED, None),
        (1, 1, "Reject", "Null-like separation\nand unstable",
         FILL_RED, EDGE_RED, RED, None),
    ]

    for r_, c_, title, sub, fill, edge, col, marker in cells:
        x = col_x[c_]
        y = row_y[r_]
        hh = row_h[r_]
        rbox(ax, x, y, cw, hh, fc=fill, ec=edge,
             lw=1.4 if r_ == 0 else 1.0, r=0.11, z=2)
        txt(ax, x + cw / 2, y + 0.34, title,
            size=17.0, weight="bold", color=col, ha="center", z=5)
        ax.text(x + cw / 2, y + 0.78, sub,
                ha="center", va="center", size=12.0,
                color=TEXT, linespacing=1.35, zorder=5)
        if marker:
            label, mcol = marker
            ax.add_patch(Circle((x + cw / 2 - 0.52, y + hh - 0.30),
                                0.13, facecolor=mcol, edgecolor="none", zorder=5))
            txt(ax, x + cw / 2 - 0.30, y + hh - 0.30, label,
                size=13.0, weight="bold", color=mcol, z=5)

    txt(ax, grid_x, y0 + h - 0.36, "Higher than null",
        size=10.5, weight="bold", color=TEXT, ha="left")
    txt(ax, grid_x + grid_w, y0 + h - 0.36, "Null-like",
        size=10.5, weight="bold", color=TEXT, ha="right")
    txt(ax, grid_x + grid_w / 2, y0 + h - 0.10,
        "Separation vs. null reference",
        size=12.5, weight="bold", color=INK, ha="center")


def build(outdir="."):
    use_style()
    fig, ax = canvas(W, H)
    top = overlay(fig)
    figure_title(
        ax, 0.42, 0.48,
        "Figure 3. Null-reference benchmarking distinguishes robust structure "
        "from plausible artifacts.", size=16)

    _null_panel(
        fig, ax, top, 0.40, 0.88, HALF, 4.70, "A", "Stroke",
        "Observed separation exceeds null reference", D.NULL_STROKE,
        obs_color="#1C7BD4",
        verdict="Observed separation exceeds null reference",
        verdict_fill=FILL_GREEN, verdict_edge="#9AD3B7",
        verdict_icon=lambda a, cx, cy, s: ico_check_circle(a, cx, cy, s, "#1E8E5A"),
        verdict_color="#1B7A50",
        xticks=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
        note_x=0.615, obs_dx=0.0,
    )

    _null_panel(
        fig, ax, top, 0.40 + HALF + 0.36, 0.88, HALF, 4.70, "B", "Sepsis",
        "Observed separation does not exceed null reference", D.NULL_SEPSIS,
        obs_color="#E8501E",
        verdict="Observed separation does not exceed null reference",
        verdict_fill="#FDF0E6", verdict_edge="#F3C9A2",
        verdict_icon=lambda a, cx, cy, s: ico_warning(a, cx, cy, s, "#E8871A"),
        verdict_color="#D9531E",
        xticks=[0.14, 0.16, 0.18, 0.20, 0.22, 0.24],
        note_x=0.605, obs_dx=-0.0075,
    )

    panel_c(fig, ax)
    return save(fig, os.path.join(outdir, "fig3_null_reference.png"))


if __name__ == "__main__":
    print(build())
