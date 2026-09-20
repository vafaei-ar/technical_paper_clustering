"""
Figure 1 -- Clustro workflow and decision logic.

Panel A: seven-step end-to-end workflow.
Panel B: naive one-shot clustering vs the Clustro workflow.
Panel C: 2 x 2 decision logic (separation vs reproducibility).
"""
from __future__ import annotations

import os

from clustro_style import *
import clustro_data as D

W, H = 15.0, 11.70
PW = W - 0.80
HALF = (PW - 0.36) / 2


def _card_header(ax, x, y, w, num, title, title_w):
    hh = max(0.60, text_height(title, title_w, 8.4, leading=1.24) + 0.20)
    rbox(ax, x, y, w, hh, fc=FILL_HEAD, ec="none", r=0.09, z=2)
    ax.add_patch(Circle((x + 0.23, y + 0.22), 0.125, facecolor="#5C86AE",
                        edgecolor="none", zorder=4))
    txt(ax, x + 0.23, y + 0.225, str(num), size=7.6, color="white",
        weight="bold", ha="center", va="center", z=5)
    para(ax, x + 0.44, y + (hh - text_height(title, title_w, 8.4, 1.24)) / 2,
         title, title_w, size=8.4, color=INK, weight="bold", leading=1.24)
    return y + hh + 0.10


def _mini(ax, x, y, w, h, label, sub=None, draw=None, fc="white"):
    rbox(ax, x, y, w, h, fc=fc, ec=EDGE_GRAY, lw=0.8, r=0.07, z=3)
    txt(ax, x + w / 2, y + 0.145, label, size=7.5, weight="bold", color=INK,
        ha="center", z=5)
    if sub:
        txt(ax, x + w / 2, y + 0.305, sub, size=6.5, color=MUTED, ha="center", z=5)
    if draw:
        draw(ax, x + w / 2, y + h * 0.66, min(w, h * 1.15) * 0.92)


def _row_item(ax, x, y, w, icon, text_, size=7.6, icon_color=BLUE, icon_s=0.22):
    icon(ax, x + 0.13, y + 0.14, icon_s, icon_color)
    end = para(ax, x + 0.30, y, text_, w - 0.32, size=size, color=TEXT, leading=1.25)
    return max(end, y + 0.30) + 0.09


def panel_a(fig, ax):
    x0, y0, w, h = 0.40, 0.92, PW, 4.62
    panel(ax, x0, y0, w, h, letter="A", title="End-to-end workflow")

    n = 7
    inner_x, inner_w = x0 + 0.26, w - 0.52
    gap = 0.15
    cw = (inner_w - gap * (n - 1)) / n
    cy, ch = y0 + 0.66, h - 0.88
    tw = cw - 0.56

    xs = [inner_x + i * (cw + gap) for i in range(n)]
    for x in xs:
        rbox(ax, x, cy, cw, ch, fc="white", ec="#D3DFEA", lw=0.9, r=0.10, z=2)
    for x in xs[:-1]:
        arrow(ax, x + cw + 0.03, cy + ch / 2, x + cw + gap - 0.03, cy + ch / 2,
              color="#9FB4C8", lw=1.3, ms=8)

    x = xs[0]
    yy = _card_header(ax, x, cy, cw, 1, "Cohort assembly", tw)
    for (name, nn, col, fill, edge) in (
        ("Stroke", D.N_STROKE, BLUE, FILL_BLUE, EDGE_BLUE),
        ("Sepsis", D.N_SEPSIS, ORANGE, FILL_ORANGE, EDGE_ORANGE),
    ):
        rbox(ax, x + 0.14, yy, cw - 0.28, 0.78, fc=fill, ec=edge, lw=0.8, r=0.08, z=3)
        ico_people(ax, x + 0.45, yy + 0.34, 0.23, col)
        txt(ax, x + 0.72, yy + 0.24, name, size=8.2, weight="bold", color=col, z=5)
        txt(ax, x + 0.72, yy + 0.48, f"(n = {nn:,})", size=7.6, color=col, z=5)
        yy += 0.92
    txt(ax, x + cw / 2, cy + ch - 0.44, "Real-world", size=7.0, color=MUTED,
        ha="center", style="italic", z=5)
    txt(ax, x + cw / 2, cy + ch - 0.22, "EHR cohorts", size=7.0, color=MUTED,
        ha="center", style="italic", z=5)

    x = xs[1]
    yy = _card_header(ax, x, cy, cw, 2,
                      "Preprocessing and feature prep", tw) + 0.18
    for icon, label in ((ico_db, "Cleaning and harmonization"),
                        (ico_gear, "Feature engineering"),
                        (ico_doc, "Standardization, missing data")):
        yy = _row_item(ax, x + 0.10, yy, cw - 0.20, icon, label) + 0.14

    x = xs[2]
    yy = _card_header(ax, x, cy, cw, 3, "Candidate model grid", tw) + 0.06
    mw, mh = (cw - 0.30) / 2, 0.92
    _mini(ax, x + 0.12, yy, mw, mh, "PCA", "(reduced space)",
          draw=lambda a, cx, cyy, s: ico_scatter(a, cx, cyy, s * 0.9, BLUE, 18, 3, 0.42))
    _mini(ax, x + 0.22 + mw, yy, mw, mh, "Raw space", "(all features)",
          draw=lambda a, cx, cyy, s: ico_scatter(a, cx, cyy, s * 0.9, "#8FA0B3", 22, 9, 0.50))
    yy2 = yy + mh + 0.12
    _mini(ax, x + 0.12, yy2, mw, mh, "k-means", "(k = 2...10)",
          draw=lambda a, cx, cyy, s: ico_clusters(a, cx, cyy, s * 0.85))
    _mini(ax, x + 0.22 + mw, yy2, mw, mh, "Agglomerative", "(k = 2...10)",
          draw=lambda a, cx, cyy, s: ico_dendro(a, cx, cyy - 0.02, s * 0.80))

    x = xs[3]
    yy = _card_header(ax, x, cy, cw, 4, "Repeated 80% subsampling", tw) + 0.10
    ico_sheets(ax, x + cw / 2, yy + 0.40, 0.66)
    yy += 0.92
    yy = para(ax, x + cw / 2, yy, "80% subsamples, full preprocessing refit",
              cw - 0.36, size=7.6, color=TEXT, ha="center", leading=1.3)
    yy += 0.26
    ico_refresh(ax, x + cw / 2, yy + 0.18, 0.32, "#4C6B8A")
    yy += 0.46
    para(ax, x + cw / 2, yy, "Repeat x 50 iterations",
         cw - 0.36, size=7.6, color=TEXT, ha="center", leading=1.3)

    x = xs[4]
    yy = _card_header(ax, x, cy, cw, 5, "Evaluation metrics", tw) + 0.10
    for icon, label in (
        (ico_bars, "Internal separation (silhouette)"),
        (ico_shield, "Full-refit stability"),
        (ico_people, "Minimum cluster size (>= 1% and n >= 30)"),
        (ico_curve, "Null-reference benchmarking"),
    ):
        yy = _row_item(ax, x + 0.10, yy, cw - 0.20, icon, label, icon_color="#3C6E9F") + 0.10

    x = xs[5]
    yy = _card_header(ax, x, cy, cw, 6, "Sensitivity analyses", tw) + 0.14
    for icon, label in (
        (ico_people, "Representation sensitivity"),
        (ico_scale, "Balanced-block weighting"),
        (ico_gear, "Feature redundancy tests"),
    ):
        yy = _row_item(ax, x + 0.10, yy, cw - 0.20, icon, label, icon_color="#3C6E9F") + 0.16

    x = xs[6]
    yy = _card_header(ax, x, cy, cw, 7,
                      "Interpretation and outputs", tw)
    yy += 0.04
    ico_doc_bars(ax, x + cw / 2, yy + 0.34, 0.62, "#3C6E9F")
    yy += 0.72
    checklist(ax, x + 0.12, yy, [
        "Selected clustering solution",
        "Phenotype characterization",
        "Robustness and sensitivity",
        "Reproducible code and outputs",
    ], cw - 0.22, size=7.5)


PANEL_BC = dict(y0=5.74, h=5.80)


def panel_b(fig, ax):
    x0, y0, h = 0.40, PANEL_BC["y0"], PANEL_BC["h"]
    w = HALF
    panel(ax, x0, y0, w, h, letter="B",
          title="Why Clustro instead of naive clustering")

    colw = (w - 0.64 - 0.30) / 2
    lx, rx = x0 + 0.32, x0 + 0.32 + colw + 0.30
    hy = y0 + 0.92

    for x, label, fc in ((lx, "Naive workflow", "#E9EDF2"),
                         (rx, "Clustro", "#D7E7F6")):
        rbox(ax, x, hy, colw, 0.48, fc=fc, ec="none", r=0.09, z=2)
        txt(ax, x + colw / 2, hy + 0.25, label, size=11, weight="bold",
            color=INK, ha="center", z=5)

    flow_top = hy + 0.48 + 0.26
    box_top = y0 + 4.32
    tsize = 8.6

    def flow(x, items, icon_color):
        heights = [max(0.40, text_height(lbl, colw - 0.80, tsize))
                   for _, lbl in items]
        gap = max(0.24, (box_top - flow_top - sum(heights)) / len(items))
        yy = flow_top
        for (icon, label), ih in zip(items, heights):
            icon(ax, x + 0.30, yy + ih / 2, 0.27, icon_color)
            para(ax, x + 0.64, yy + (ih - text_height(label, colw - 0.80, tsize)) / 2,
                 label, colw - 0.78, size=tsize, color=TEXT)
            yy += ih
            arrow(ax, x + colw / 2, yy + gap * 0.22, x + colw / 2, yy + gap * 0.80,
                  color="#9FB4C8", lw=1.4, ms=9)
            yy += gap

    flow(lx, [
        (ico_db, "One-shot clustering (full dataset)"),
        (lambda a, cx, cy_, s, c: ico_clusters(a, cx, cy_, s * 1.15),
         "Direct phenotype interpretation"),
        (ico_doc, "Publish results"),
    ], "#8593A6")

    flow(rx, [
        (lambda a, cx, cy_, s, c: _grid_icon(a, cx, cy_, s),
         "Compare candidate models (PCA / raw; k-means / GMM / agglomerative)"),
        (ico_refresh, "Repeated subsampling with full preprocessing refit"),
        (ico_curve, "Null-reference benchmarking"),
        (ico_gear, "Sensitivity analyses"),
    ], "#3C6E9F")

    rbox(ax, lx, box_top, colw, 0.84, fc=FILL_RED, ec=EDGE_RED, lw=0.9, r=0.10, z=2)
    ico_warning(ax, lx + 0.36, box_top + 0.42, 0.32, "#D64545")
    para(ax, lx + 0.66, box_top + 0.20,
         "High risk of spurious phenotypes",
         colw - 0.82, size=tsize, color="#B03030", weight="bold")

    rbox(ax, rx, box_top, colw, 0.84, fc=FILL_GREEN, ec=EDGE_GREEN, lw=0.9,
         r=0.10, z=2)
    ico_check_circle(ax, rx + 0.36, box_top + 0.42, 0.32, GREEN)
    para(ax, rx + 0.66, box_top + 0.20,
         "More reliable and interpretable phenotypes",
         colw - 0.82, size=tsize, color="#16714A", weight="bold")

    cy = box_top + 1.06
    for line in ("Trustworthy clusters require more than",
                 "one successful run."):
        txt(ax, rx + colw / 2, cy, line, size=8.2, color=MUTED, style="italic",
            ha="center", z=5)
        cy += 0.20


def _grid_icon(ax, cx, cy, s):
    w = 0.95 * s
    rbox(ax, cx - w / 2, cy - w / 2, w, w, fc="white", ec="#3C6E9F", lw=1.1,
         r=0.10 * s, z=4)
    hline(ax, cx - w / 2, cx + w / 2, cy - w / 2 + 0.30 * w, color="#3C6E9F",
          lw=0.9, z=5)
    hline(ax, cx - w / 2, cx + w / 2, cy - w / 2 + 0.65 * w, color="#3C6E9F",
          lw=0.9, z=5)
    vline(ax, cx - w / 2 + 0.38 * w, cy - w / 2, cy + w / 2, color="#3C6E9F",
          lw=0.9, z=5)


def panel_c(fig, ax):
    x0, y0, h = 0.40 + HALF + 0.36, PANEL_BC["y0"], PANEL_BC["h"]
    w = HALF
    panel(ax, x0, y0, w, h, letter="C", title="Decision logic")

    grid_x, grid_w = x0 + 1.40, w - 1.40 - 0.32
    cw = (grid_w - 0.20) / 2
    col_x = [grid_x, grid_x + cw + 0.20]

    rbox(ax, grid_x, y0 + 0.84, grid_w, 0.34, fc="#DCEAF7", ec="none", r=0.07, z=2)
    txt(ax, grid_x + grid_w / 2, y0 + 1.01,
        "Cluster separation (vs. null reference)", size=9.0, color=INK,
        ha="center", z=5)
    for x, label in zip(col_x, ("High (better than null)",
                                "Null-like (no better than null)")):
        txt(ax, x + cw / 2, y0 + 1.40, label, size=9.0, weight="bold",
            color=TEXT, ha="center", z=5)

    row_y = [y0 + 1.60, y0 + 3.30]
    row_h = [1.60, 1.40]

    ax.text(x0 + 0.56, (row_y[0] + row_y[1] + row_h[1]) / 2,
            "Reproducibility\n(full-refit stability)", rotation=90, ha="center",
            va="center", size=8.6, color=TEXT, zorder=5, linespacing=1.4)

    for y, hh, top, bot in zip(row_y, row_h, ("High", "Low"),
                               ("(stable)", "(unstable)")):
        txt(ax, x0 + 1.06, y + hh / 2 - 0.11, top, size=9.0, weight="bold",
            color=TEXT, ha="center", z=5)
        txt(ax, x0 + 1.06, y + hh / 2 + 0.11, bot, size=8.1, color=MUTED,
            ha="center", z=5)

    cells = [
        (0, 0, "Promising", GREEN, FILL_GREEN, "#7FC6A3", ico_trophy,
         ["High separation", "High reproducibility", "Adequate cluster sizes"]),
        (0, 1, "Caution", "#E08A1E", FILL_ORANGE, "#F0C089", ico_warning,
         ["Null-like separation", "High reproducibility",
          "May reflect weak structure"]),
        (1, 0, "Reject", RED, FILL_RED, EDGE_RED, ico_cross,
         ["Poor reproducibility", "Results not reliable"]),
        (1, 1, "Reject", RED, FILL_RED, EDGE_RED, ico_cross,
         ["Tiny or fragmented clusters", "Not interpretable"]),
    ]
    for r_, c_, title, col, fill, edge, icon, items in cells:
        x, y, hh = col_x[c_], row_y[r_], row_h[r_]
        lw = 1.7 if r_ == 0 else 1.0
        rbox(ax, x, y, cw, hh, fc=fill, ec=edge, lw=lw, r=0.11, z=2)
        icon(ax, x + 0.38, y + 0.36, 0.30, col)
        txt(ax, x + 0.64, y + 0.36, title, size=11.5, weight="bold", color=col, z=5)
        bullets(ax, x + 0.26, y + 0.64, items, cw - 0.46, size=8.0, color=TEXT,
                dot_color=col)

    ly = row_y[1] + row_h[1] + 0.26
    for label, nn, col, note in (
        ("Stroke", D.N_STROKE, BLUE,
         "promising region: higher separation, stable."),
        ("Sepsis", D.N_SEPSIS, ORANGE,
         "caution region: null-like separation, reproducible."),
    ):
        ax.add_patch(Circle((x0 + 0.50, ly + 0.16), 0.12, facecolor=col,
                            edgecolor="none", zorder=4))
        ly = rich_line(ax, x0 + 0.78, ly, f"{label} (n = {nn:,})", note,
                       w - 1.15, size=8.6, lead_color=TEXT, fig=fig) + 0.12


def build(outdir="."):
    use_style()
    fig, ax = canvas(W, H)
    figure_title(ax, 0.42, 0.50, "Figure 1. Clustro workflow and decision logic")
    panel_a(fig, ax)
    panel_b(fig, ax)
    panel_c(fig, ax)
    return save(fig, os.path.join(outdir, "fig1_workflow.png"))


if __name__ == "__main__":
    print(build())
