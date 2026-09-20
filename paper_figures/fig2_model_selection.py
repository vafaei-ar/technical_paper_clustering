"""
Figure 2 -- Model-selection landscape and selected solutions.

Panels A/B: silhouette (x) vs mean full-refit subsample ARI (y) for candidate
models; marker shape = model family, marker colour = k, red star = selected
solution. Panel C gives the selection rationale.
"""
from __future__ import annotations

import os

from clustro_style import *
import clustro_data as D

W, H = 15.0, 10.85
PW = W - 0.80
HALF = (PW - 0.36) / 2

FAMILY_MARKER = {"PCA k-means": "o", "Raw k-means": "s", "Raw agglomerative": "^"}
K_COLOR = {2: "#2E86C1", 3: "#E8871A", 4: "#2E9E68"}


def _callout(ax, xy, xytext, lines, fc, ec, tc, ha="center", size=11.5,
             arrow_color=None, weight_first="bold"):
    text = "\n".join(lines)
    ax.annotate(
        text, xy=xy, xytext=xytext, textcoords="data", ha=ha, va="center",
        fontsize=size, color=tc, linespacing=1.35, zorder=8,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=fc, edgecolor=ec, lw=1.0),
        arrowprops=dict(arrowstyle="-|>", color=arrow_color or ec, lw=1.3,
                        shrinkA=3, shrinkB=6,
                        connectionstyle="arc3,rad=0.16"),
    )


def _scatter_panel(fig, ax_c, x0, y0, w, h, letter, title, candidates, selected,
                   xlim, ylim, xticks=None, yticks=None, annotations=()):
    panel(ax_c, x0, y0, w, h, letter=letter, title=title)

    ax = axes_in(fig, x0 + 1.05, y0 + 0.92, w - 1.45, h - 1.72)
    tidy_axes(ax, grid="both")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel("Silhouette score", fontsize=15.6, color=TEXT, labelpad=6)
    ax.set_ylabel("Mean full-refit subsample ARI", fontsize=15.6, color=TEXT,
                  labelpad=6)

    for fam, k, sil, ari in candidates:
        ax.scatter(sil, ari, marker=FAMILY_MARKER[fam], s=95,
                   color=K_COLOR[k], edgecolors="white", linewidths=0.6,
                   zorder=6)
        ax.annotate(f"$k$ = {k}", (sil, ari), textcoords="offset points",
                    xytext=(11, -9), fontsize=13.3, color=TEXT, zorder=7)

    sel = [c for c in candidates if (c[0], c[1]) == selected][0]
    ax.scatter(sel[2], sel[3], marker="*", s=420, color="#E23B3B",
               edgecolors="#7A0B18", linewidths=1.2, zorder=8)

    for ann in annotations:
        _callout(ax, **ann)
    return ax


def panel_c(fig, ax):
    x0, y0, w, h = 0.40, 5.80, PW, 3.96
    panel(ax, x0, y0, w, h, letter="C", title="Selection rationale")

    top = y0 + 0.88
    box_y, box_h = top + 0.60, 2.16
    groups = [
        (x0 + 0.30, 5.62, FILL_BLUE,
         "Stroke: $k$ = 3, separation-stability compromise",
         [("$k$ = 4", ["Higher stability (ARI)", "Lower separation"], False),
          ("$k$ = 3 (selected)", ["Good separation", "High stability", "Balanced solution"], True),
          ("$k$ = 2", ["Higher separation", "Similar stability"], False)]),
        (x0 + 6.18, 5.22, FILL_ORANGE,
         "Sepsis: $k$ = 3, most reproducible compromise",
         [("$k$ = 2", ["Lower stability", "Unstable lower tail"], False),
          ("$k$ = 3 (selected)", ["High stability", "Good separation", "Strong worst-case stability"], True),
          ("$k$ = 4", ["Slightly higher separation", "Lower worst-case robustness"], False)]),
    ]

    for gx, gw, fill, heading, boxes in groups:
        rbox(ax, gx, top, gw, 0.40, fc=fill, ec="none", r=0.08, z=2)
        txt(ax, gx + 0.16, top + 0.21, heading, size=10.0, weight="bold",
            color=INK, z=5)
        bw = (gw - 0.30) / 3
        for i, (btitle, items, is_sel) in enumerate(boxes):
            bx = gx + i * (bw + 0.15)
            rbox(ax, bx, box_y, bw, box_h,
                 fc=FILL_RED if is_sel else FILL_GREEN,
                 ec=EDGE_RED if is_sel else EDGE_GREEN,
                 lw=1.5 if is_sel else 0.9, r=0.10, z=2)
            txt(ax, bx + 0.20, box_y + 0.26, btitle, size=9.6, weight="bold",
                color=RED if is_sel else "#1B7A50", z=5)
            bullets(ax, bx + 0.22, box_y + 0.48, items, bw - 0.40, size=8.6,
                    color=TEXT, dot_color=RED if is_sel else "#1B7A50")

    ix, iw = x0 + 11.70, 2.30
    rbox(ax, ix, top, iw, box_y + box_h - top, fc="#F3F8FC", ec="none",
         r=0.10, z=2)
    ico_gear(ax, ix + 0.22, top + 0.26, 0.30, "#40536B")
    end = para(ax, ix + 0.44, top + 0.06,
               "Model choice uses several criteria", iw - 0.56,
               size=8.8, color=INK, weight="bold")
    para(ax, ix + 0.14, max(end, top + 0.60) + 0.12,
         "Separation, full-refit stability, cluster-size adequacy and robustness "
         "are weighed together rather than any single metric.",
         iw - 0.30, size=8.8, color=TEXT)


def legend_strip(fig, ax):
    x0, y0, w, h = 0.40, 10.02, PW, 0.68
    rbox(ax, x0, y0, w, h, fc="white", ec=BORDER, lw=1.1, r=0.12, z=1)
    cy = y0 + h / 2

    fams = [("PCA k-means", 4), ("Raw k-means", 3), ("Raw agglomerative", 2)]
    size = 9.6
    for _ in range(8):
        need = (measure(fig, "Model family (marker shape)", size, "bold") + 0.34
                + sum(measure(fig, f, size) + 0.52 for f, _ in fams)
                + measure(fig, "Number of clusters (color)", size, "bold") + 0.54
                + sum(measure(fig, f"k = {k}", size) + 0.50 for k in (2, 3, 4))
                + measure(fig, "Selected solution", size) + 0.80)
        if need <= w - 0.60:
            break
        size *= 0.94

    x = x0 + 0.30
    txt(ax, x, cy, "Model family (marker shape)", size=size, weight="bold",
        color=INK, va="center")
    x += measure(fig, "Model family (marker shape)", size, "bold") + 0.34
    for fam, k in fams:
        ax.plot([x], [cy], marker=FAMILY_MARKER[fam], ms=9, color=K_COLOR[k],
                zorder=5, markeredgecolor="white", markeredgewidth=0.6)
        txt(ax, x + 0.20, cy, fam, size=size, color=TEXT)
        x += measure(fig, fam, size) + 0.52

    x += 0.04
    vline(ax, x, y0 + 0.12, y0 + h - 0.12, color=GRID, lw=1.0)
    x += 0.24
    txt(ax, x, cy, "Number of clusters (color)", size=size, weight="bold",
        color=INK)
    x += measure(fig, "Number of clusters (color)", size, "bold") + 0.30
    for k in (2, 3, 4):
        legend_dot(ax, x, cy, f"$k$ = {k}", K_COLOR[k], size=size)
        x += measure(fig, f"k = {k}", size) + 0.50

    x += 0.04
    vline(ax, x, y0 + 0.12, y0 + h - 0.12, color=GRID, lw=1.0)
    x += 0.30
    ax.plot([x], [cy], marker="*", ms=19, color="#E23B3B",
            markeredgecolor="#7A0B18", markeredgewidth=1.0, zorder=5)
    txt(ax, x + 0.26, cy, "Selected solution", size=size, color=TEXT)


def build(outdir="."):
    use_style()
    fig, ax = canvas(W, H)
    figure_title(ax, 0.42, 0.50,
                 "Figure 2. Model-selection landscape and selected solutions")

    s4 = D.point(D.STROKE_CANDIDATES, "PCA k-means", 4)
    s3 = D.point(D.STROKE_CANDIDATES, "PCA k-means", 3)
    s2 = D.point(D.STROKE_CANDIDATES, "PCA k-means", 2)
    sa2 = D.point(D.STROKE_CANDIDATES, "Raw agglomerative", 2)

    _scatter_panel(
        fig, ax, 0.40, 0.90, HALF, 4.78, "A", "Stroke",
        D.STROKE_CANDIDATES, D.STROKE_SELECTED,
        xlim=(0.118, 0.232), ylim=(-0.03, 1.12),
        xticks=[0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        annotations=[
            dict(xy=s4, xytext=(0.1345, 1.075),
                 lines=["$k$ = 4", "(higher stability)"], fc=FILL_GREEN,
                 ec="#9AD3B7", tc="#1B7A50"),
            dict(xy=s3, xytext=(0.1715, 0.985),
                 lines=["Selected $k$ = 3", "(best compromise)"], fc=FILL_RED,
                 ec=EDGE_RED, tc=RED),
            dict(xy=s2, xytext=(0.211, 1.055),
                 lines=["$k$ = 2", "(higher separation)"], fc=FILL_GREEN,
                 ec="#9AD3B7", tc="#1B7A50"),
            dict(xy=sa2, xytext=(0.203, 0.455),
                 lines=["Agglomerative", "(low stability)"], fc=FILL_BLUE,
                 ec=EDGE_BLUE, tc=BLUE_DK),
        ],
    )

    e3 = D.point(D.SEPSIS_CANDIDATES, "PCA k-means", 3)
    e4 = D.point(D.SEPSIS_CANDIDATES, "PCA k-means", 4)
    e2 = D.point(D.SEPSIS_CANDIDATES, "PCA k-means", 2)

    _scatter_panel(
        fig, ax, 0.40 + HALF + 0.36, 0.90, HALF, 4.78, "B", "Sepsis",
        D.SEPSIS_CANDIDATES, D.SEPSIS_SELECTED,
        xlim=(0.1555, 0.1795), ylim=(0.800, 1.000),
        xticks=[0.1575, 0.1600, 0.1625, 0.1650, 0.1675, 0.1700, 0.1725,
                0.1750, 0.1775],
        yticks=[0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
        annotations=[
            dict(xy=e3, xytext=(0.1668, 0.9725),
                 lines=["Selected $k$ = 3", "(best compromise)"], fc=FILL_RED,
                 ec=EDGE_RED, tc=RED),
            dict(xy=e4, xytext=(0.1758, 0.9855),
                 lines=["$k$ = 4", "(higher separation,", "lower robustness)"],
                 fc=FILL_ORANGE, ec=EDGE_ORANGE, tc=ORANGE_DK),
            dict(xy=e2, xytext=(0.1640, 0.856),
                 lines=["$k$ = 2", "(lower stability)"], fc=FILL_ORANGE,
                 ec=EDGE_ORANGE, tc=ORANGE_DK),
        ],
    )

    panel_c(fig, ax)
    legend_strip(fig, ax)
    return save(fig, os.path.join(outdir, "fig2_model_selection.png"))


if __name__ == "__main__":
    print(build())
