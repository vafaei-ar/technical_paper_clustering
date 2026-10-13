"""
Figure 1 -- Clustro workflow and decision logic.

Panel A: seven-step end-to-end workflow, with screening before full-refit stability.
Panel B: naive one-shot clustering vs the Clustro workflow.
Panel C: illustrative 2 x 2 diagnostics (separation vs reproducibility).

Layout notes
------------
Every block here is content-driven rather than pinned to hand-tuned
coordinates:

* panel A cards spread their rows over the full card height (`_spread`) and
  shrink their own text only if that particular card still overflows;
* panel B measures both flow columns, places the outcome boxes below the
  taller one, and sizes those boxes to their own labels;
* panel C derives its row heights from the tallest cell in each row;
* `build()` then sizes the canvas to whatever panels B and C need.

So raising a value in TEXT_SIZE makes the type bigger and the layout
re-flows instead of colliding. On-page type size is
``TEXT_SIZE * FONT_SCALE * (printed width / W)`` -- if the figure will be
printed narrow, lower `W` rather than lowering the sizes.
"""
from __future__ import annotations

import os

from clustro_style import *
import clustro_data as D

W, H = 15.0, 12.10
PW = W - 0.80
HALF = (PW - 0.36) / 2

# --------------------------------------------------------------------------- #
# USER-ADJUSTABLE DISPLAY SETTINGS
# Change these values first when tuning Figure 1.
# TEXT_SIZE values are nominal point sizes and are also scaled by
# clustro_style.FONT_SCALE.
# TEXT_POS values use the figure's inch-based top-left coordinate system.
# --------------------------------------------------------------------------- #
TEXT_SIZE = {
    "figure_title": 18.0,
    "card_number": 8.4,
    "card_title": 9.2,
    "mini_label": 8.4,
    "mini_sub": 7.4,
    "row_item": 9.0,
    "cohort_name": 9.6,
    "cohort_n": 8.8,
    "cohort_note": 8.2,
    "subsample_text": 9.0,
    "checklist": 8.8,
    "panel_b_header": 12.0,
    "panel_b_body": 9.2,
    "panel_b_note": 8.8,
    "decision_header": 9.8,
    "decision_col_header": 9.8,
    "decision_axis": 9.2,
    "decision_axis_main": 9.6,
    "decision_axis_small": 8.6,
    "decision_cell_title": 10.7,
    "decision_bullets": 8.8,
    "decision_cohort_note": 9.4,
}

TEXT_POS = {
    "figure_title": (0.42, 0.50),
    "panel_a_origin": (0.40, 0.92),
    "panel_bc_y": 6.02,
    "panel_bc_h": 5.70,
    "decision_grid_x_offset": 1.16,
    "decision_top_header_y_offset": 0.91,
    "decision_col_header_y_offset": 1.26,
    "decision_vertical_axis_x_offset": 0.34,
    "decision_side_label_x_offset": 0.80,
}


def _card_header(ax, x, y, w, num, title, title_w):
    hh = max(0.60, text_height(title, title_w, TEXT_SIZE["card_title"], leading=1.24) + 0.20)
    rbox(ax, x, y, w, hh, fc=FILL_HEAD, ec="none", r=0.09, z=2)
    ax.add_patch(Circle((x + 0.23, y + 0.22), 0.125, facecolor="#5C86AE",
                        edgecolor="none", zorder=4))
    txt(ax, x + 0.23, y + 0.225, str(num), size=TEXT_SIZE["card_number"], color="white",
        weight="bold", ha="center", va="center", z=5)
    para(ax, x + 0.44, y + (hh - text_height(title, title_w, TEXT_SIZE["card_title"], 1.24)) / 2,
         title, title_w, size=TEXT_SIZE["card_title"], color=INK, weight="bold", leading=1.24)
    return y + hh + 0.10


def _mini(ax, x, y, w, h, label, sub=None, draw=None, fc="white"):
    rbox(ax, x, y, w, h, fc=fc, ec=EDGE_GRAY, lw=0.8, r=0.07, z=3)
    txt(ax, x + w / 2, y + 0.145, label, size=TEXT_SIZE["mini_label"], weight="bold", color=INK,
        ha="center", z=5)
    if sub:
        txt(ax, x + w / 2, y + 0.305, sub, size=TEXT_SIZE["mini_sub"], color=MUTED, ha="center", z=5)
    if draw:
        draw(ax, x + w / 2, y + h * 0.66, min(w, h * 1.15) * 0.92)


def _spread(y_top, y_bot, heights, min_gap=0.08):
    """Even gap that makes a stack of blocks fill the space available.

    Returns the gap to leave between consecutive blocks so the column reads as
    a full card rather than content pinned to the top with dead space below.
    """
    if len(heights) < 2:
        return min_gap
    free = (y_bot - y_top) - sum(heights)
    return max(min_gap, free / (len(heights) - 1))


def _row_item(ax, x, y, w, icon, text_, size=None, icon_color=BLUE, icon_s=0.26):
    if size is None:
        size = TEXT_SIZE["row_item"]
    icon(ax, x + 0.13, y + 0.14, icon_s, icon_color)
    end = para(ax, x + 0.30, y, text_, w - 0.32, size=size, color=TEXT, leading=1.25)
    return max(end, y + 0.30) + 0.09


def panel_a(fig, ax):
    x0, y0 = TEXT_POS["panel_a_origin"]
    w, h = PW, 4.90
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

    bottom = cy + ch - 0.16          # content must finish by here
    isize = TEXT_SIZE["row_item"]

    def rows(x, start, items, icon_color=BLUE):
        """Icon + label rows, spread so they fill the card (never past it)."""
        tw_ = cw - 0.66
        size = isize
        for _ in range(12):                      # shrink only if this card needs it
            heights = [max(0.34, text_height(lbl, tw_, size, leading=1.25))
                       for _, lbl in items]
            if sum(heights) + 0.08 * (len(items) - 1) <= bottom - start:
                break
            size *= 0.95
        g = _spread(start, bottom, heights, min_gap=0.08)
        yy = start
        for (icon, label), ih in zip(items, heights):
            icon(ax, x + 0.18, yy + ih / 2, 0.26, icon_color)
            para(ax, x + 0.50, yy + (ih - text_height(label, tw_, size, 1.25)) / 2,
                 label, tw_, size=size, color=TEXT, leading=1.25)
            yy += ih + g

    # ---- 1. cohort assembly
    x = xs[0]
    yy = _card_header(ax, x, cy, cw, 1, "Cohort assembly", tw)
    chip_h = 0.92
    note_h = 0.52
    g = _spread(yy, bottom, [chip_h, chip_h, note_h], min_gap=0.14)
    for (name, nn, col, fill, edge) in (
        ("Stroke", D.N_STROKE, BLUE, FILL_BLUE, EDGE_BLUE),
        ("Sepsis", D.N_SEPSIS, ORANGE, FILL_ORANGE, EDGE_ORANGE),
    ):
        rbox(ax, x + 0.12, yy, cw - 0.24, chip_h, fc=fill, ec=edge, lw=0.9,
             r=0.09, z=3)
        ico_people(ax, x + 0.46, yy + chip_h / 2, 0.27, col)
        txt(ax, x + 0.76, yy + 0.33, name, size=TEXT_SIZE["cohort_name"],
            weight="bold", color=col, z=5)
        txt(ax, x + 0.76, yy + 0.60, f"(n = {nn:,})", size=TEXT_SIZE["cohort_n"],
            color=col, z=5)
        yy += chip_h + g
    txt(ax, x + cw / 2, yy + 0.12, "Real-world", size=TEXT_SIZE["cohort_note"],
        color=MUTED, ha="center", style="italic", z=5)
    txt(ax, x + cw / 2, yy + 0.38, "EHR cohorts", size=TEXT_SIZE["cohort_note"],
        color=MUTED, ha="center", style="italic", z=5)

    # ---- 2. preprocessing
    x = xs[1]
    yy = _card_header(ax, x, cy, cw, 2, "Preprocessing and feature prep", tw)
    rows(x, yy + 0.10, [(ico_db, "Cleaning and harmonization"),
                        (ico_gear, "Feature engineering"),
                        (ico_doc, "Standardization, missing data")])

    # ---- 3. candidate model grid
    x = xs[2]
    yy = _card_header(ax, x, cy, cw, 3, "Candidate model grid", tw) + 0.04
    minis = [
        ("Raw / PCA", "(2 spaces)",
         lambda a_, cx, cyy, s_: ico_scatter(a_, cx, cyy, s_, BLUE, 16, 3, 0.42)),
        ("K-means", "(k = 2 to 10)",
         lambda a_, cx, cyy, s_: ico_clusters(a_, cx, cyy, s_ * 0.95)),
        ("GMM", "(k = 2 to 10)",
         lambda a_, cx, cyy, s_: ico_clusters(a_, cx, cyy, s_ * 0.85)),
        ("Agglomerative", "(k = 2 to 10)",
         lambda a_, cx, cyy, s_: ico_dendro(a_, cx, cyy, s_ * 0.85)),
    ]
    mh = (bottom - yy - 3 * 0.10) / 4
    for label, sub, draw in minis:
        rbox(ax, x + 0.10, yy, cw - 0.20, mh, fc="white", ec=EDGE_GRAY, lw=0.8,
             r=0.08, z=3)
        txt(ax, x + 0.24, yy + mh / 2 - 0.13, label, size=TEXT_SIZE["mini_label"],
            weight="bold", color=INK, z=5)
        txt(ax, x + 0.24, yy + mh / 2 + 0.13, sub, size=TEXT_SIZE["mini_sub"],
            color=MUTED, z=5)
        draw(ax, x + cw - 0.44, yy + mh / 2, mh * 0.62)
        yy += mh + 0.10

    # ---- 4. initial candidate screening
    x = xs[3]
    yy = _card_header(ax, x, cy, cw, 4, "Candidate screening", tw)
    rows(x, yy + 0.10, [
        (ico_bars, "Separation and internal validity"),
        (ico_people, "Minimum cluster size (1% and n >= 30)"),
        (ico_shield, "Across-seed agreement"),
        (ico_check_circle, "Shortlist candidates"),
    ], icon_color="#3C6E9F")

    # ---- 5. repeated full-pipeline refit for the shortlist
    x = xs[4]
    yy = _card_header(ax, x, cy, cw, 5, "Full-refit stability", tw)
    t1 = "80% subsamples; refit all preprocessing"
    t2 = "50 repeated subsamples"
    ssize = TEXT_SIZE["subsample_text"]
    h1 = text_height(t1, cw - 0.36, ssize, 1.3)
    h2 = text_height(t2, cw - 0.36, ssize, 1.3)
    blocks = [0.72, h1, 0.56, h2]
    g = _spread(yy + 0.06, bottom, blocks, min_gap=0.12)
    yy += 0.06
    ico_sheets(ax, x + cw / 2, yy + 0.36, 0.70)
    yy += blocks[0] + g
    para(ax, x + cw / 2, yy, t1, cw - 0.36, size=ssize, color=TEXT,
         ha="center", leading=1.3)
    yy += h1 + g
    ico_refresh(ax, x + cw / 2, yy + 0.28, 0.36, "#4C6B8A")
    yy += blocks[2] + g
    para(ax, x + cw / 2, yy, t2, cw - 0.36, size=ssize, color=TEXT,
         ha="center", leading=1.3)

    # ---- 6. reference and specification checks
    x = xs[5]
    yy = _card_header(ax, x, cy, cw, 6, "Reference and sensitivity checks", tw)
    rows(x, yy + 0.10, [
        (ico_curve, "Gaussian-copula reference"),
        (ico_people, "Representation / cluster count"),
        (ico_scale, "Balanced feature blocks"),
        (ico_gear, "Redundant features"),
    ], icon_color="#3C6E9F")

    # ---- 7. interpretation and outputs
    x = xs[6]
    yy = _card_header(ax, x, cy, cw, 7, "Interpretation and outputs", tw)
    items = ["Selected clustering solution", "Phenotype characterization",
             "Robustness and sensitivity", "Reproducible code and outputs"]
    csize = TEXT_SIZE["checklist"]
    ico_doc_bars(ax, x + cw / 2, yy + 0.42, 0.72, "#3C6E9F")
    yy += 0.90
    for _ in range(12):
        heights = [text_height(it, cw - 0.46, csize, 1.32) for it in items]
        if sum(heights) + 0.05 * (len(items) - 1) <= bottom - yy:
            break
        csize *= 0.95
    g = _spread(yy, bottom, heights, min_gap=0.05)
    checklist(ax, x + 0.12, yy, items, cw - 0.26, size=csize, gap=g)


PANEL_BC = dict(y0=TEXT_POS["panel_bc_y"], h=TEXT_POS["panel_bc_h"])


PANEL_B_FLOWS = {
    "naive": ["One-shot clustering on full data",
              "Interpret assigned groups",
              "Report groups without checks"],
    "clustro": ["Screen candidate models (raw / PCA; k-means / GMM / "
                "agglomerative)",
                "Full-pipeline stability testing",
                "Null-reference comparisons",
                "Feature and model sensitivity"],
}


OUTCOME_TEXT = {"naive": "Weak partitions may go unnoticed",
                "clustro": "Auditable results, including weak solutions"}


def _outcome_box_h(colw, tsize):
    """Outcome boxes are as tall as the taller of the two labels needs."""
    return max(0.84, max(text_height(t, colw - 0.82, tsize)
                         for t in OUTCOME_TEXT.values()) + 0.34)


def _flow_height(colw, tsize, min_gap=0.18):
    """Height the taller of the two columns needs, arrows included."""
    need = 0.0
    for labels in PANEL_B_FLOWS.values():
        hs = [max(0.38, text_height(l, colw - 0.64, tsize)) for l in labels]
        need = max(need, sum(hs) + len(hs) * min_gap)
    return need


def panel_b_height():
    """Total panel height panel B needs with the current text sizes."""
    colw = (HALF - 0.94) / 2
    flow_top = 0.92 + 0.48 + 0.26
    tsize = TEXT_SIZE["panel_b_body"]
    return (flow_top + _flow_height(colw, tsize) + _outcome_box_h(colw, tsize)
            + 0.62)


def panel_b(fig, ax):
    x0, y0, h = 0.40, PANEL_BC["y0"], PANEL_BC["h"]
    w = HALF
    panel(ax, x0, y0, w, h, letter="B",
          title="Why audit a clustering analysis?")

    colw = (w - 0.64 - 0.30) / 2
    lx, rx = x0 + 0.32, x0 + 0.32 + colw + 0.30
    hy = y0 + 0.92

    for x, label, fc in ((lx, "Naive workflow", "#E9EDF2"),
                         (rx, "Clustro", "#D7E7F6")):
        rbox(ax, x, hy, colw, 0.48, fc=fc, ec="none", r=0.09, z=2)
        txt(ax, x + colw / 2, hy + 0.25, label, size=TEXT_SIZE["panel_b_header"], weight="bold",
            color=INK, ha="center", z=5)

    flow_top = hy + 0.48 + 0.26
    tsize = TEXT_SIZE["panel_b_body"]
    box_top = flow_top + _flow_height(colw, tsize)

    def flow(x, items, icon_color):
        heights = [max(0.38, text_height(lbl, colw - 0.64, tsize))
                   for _, lbl in items]
        gap = max(0.18, (box_top - flow_top - sum(heights)) / len(items))
        yy = flow_top
        for (icon, label), ih in zip(items, heights):
            icon(ax, x + 0.26, yy + ih / 2, 0.27, icon_color)
            para(ax, x + 0.54, yy + (ih - text_height(label, colw - 0.64, tsize)) / 2,
                 label, colw - 0.64, size=tsize, color=TEXT)
            yy += ih
            arrow(ax, x + colw / 2, yy + gap * 0.22, x + colw / 2, yy + gap * 0.80,
                  color="#9FB4C8", lw=1.4, ms=9)
            yy += gap

    flow(lx, [
        (ico_db, "One-shot clustering on full data"),
        (lambda a, cx, cy_, s, c: ico_clusters(a, cx, cy_, s * 1.15),
         "Interpret assigned groups"),
        (ico_doc, "Report groups without checks"),
    ], "#8593A6")

    flow(rx, [
        (lambda a, cx, cy_, s, c: _grid_icon(a, cx, cy_, s),
         "Screen models: raw / PCA, k-means / GMM / agglomerative"),
        (ico_refresh, "Refit pipeline in subsamples"),
        (ico_curve, "Compare with null references"),
        (ico_gear, "Test feature / model choices"),
    ], "#3C6E9F")

    box_h = _outcome_box_h(colw, tsize)
    rbox(ax, lx, box_top, colw, box_h, fc=FILL_RED, ec=EDGE_RED, lw=0.9, r=0.10,
         z=2)
    ico_warning(ax, lx + 0.36, box_top + box_h / 2, 0.32, "#D64545")
    para(ax, lx + 0.66,
         box_top + (box_h - text_height(OUTCOME_TEXT["naive"], colw - 0.82,
                                        tsize)) / 2,
         OUTCOME_TEXT["naive"],
         colw - 0.82, size=tsize, color="#B03030", weight="bold")

    rbox(ax, rx, box_top, colw, box_h, fc=FILL_GREEN, ec=EDGE_GREEN, lw=0.9,
         r=0.10, z=2)
    ico_check_circle(ax, rx + 0.36, box_top + box_h / 2, 0.32, GREEN)
    para(ax, rx + 0.66,
         box_top + (box_h - text_height(OUTCOME_TEXT["clustro"], colw - 0.82,
                                        tsize)) / 2,
         OUTCOME_TEXT["clustro"],
         colw - 0.82, size=tsize, color="#16714A", weight="bold")

    cy = box_top + box_h + 0.24
    for line in ("Auditing can identify stable partitions",
                 "that lack robust support."):
        txt(ax, rx + colw / 2, cy, line, size=TEXT_SIZE["panel_b_note"], color=MUTED, style="italic",
            ha="center", z=5)
        cy += 0.20

    # Use the lower portion of the comparison panel for the possible
    # audit outcomes rather than leaving it empty when panel C is taller.
    foot_y = y0 + h - 1.16
    if foot_y > cy + 0.20:
        rbox(ax, lx, foot_y, w - 0.64, 0.82, fc=FILL_GRAY,
             ec=EDGE_GRAY, lw=0.7, r=0.08, z=2)
        para(ax, lx + 0.20, foot_y + 0.16,
             "An audit may retain, qualify, or reject a candidate partition."
             " It does not validate clinical subtypes.",
             w - 1.00, size=TEXT_SIZE["panel_b_note"],
             color=TEXT, leading=1.25)


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


DECISION_CELLS = [
    (0, 0, "Proceed to robustness checks", GREEN, FILL_GREEN, "#7FC6A3", "check",
     ["Above reference separation", "High resampling stability", "Test feature sensitivity"]),
    (0, 1, "Interpret cautiously", "#E08A1E", FILL_ORANGE, "#F0C089", "warning",
     ["Not above reference", "Stable under resampling", "No subtype validation"]),
    (1, 0, "Unstable partition", RED, FILL_RED, EDGE_RED, "cross",
     ["Above reference separation", "Low resampling stability"]),
    (1, 1, "Weak evidence", RED, FILL_RED, EDGE_RED, "cross",
     ["Not above reference", "Low resampling stability"]),
]
CELL_ICONS = {"check": ico_check_circle, "warning": ico_warning, "cross": ico_cross}


def _decision_row_heights(cw):
    """Rows make room for wrapped, meaning-first titles and bullets."""
    size = TEXT_SIZE["decision_bullets"]
    title_size = TEXT_SIZE["decision_cell_title"]
    heights = [0.0, 0.0]
    for r_, _c, title, _col, _f, _e, _i, items in DECISION_CELLS:
        title_h = text_height(title, cw - 0.95, title_size, 1.16)
        body = sum(text_height(it, cw - 0.52, size, 1.32) + 0.045
                   for it in items)
        heights[r_] = max(heights[r_], 0.28 + title_h + 0.17 + body + 0.19)
    return heights


def panel_c_height():
    """Total panel height panel C needs with the current text sizes."""
    cw = (HALF - TEXT_POS["decision_grid_x_offset"] - 0.32 - 0.16) / 2
    rh = _decision_row_heights(cw)
    return 1.44 + rh[0] + 0.10 + rh[1] + 0.28 + 1.95


def panel_c(fig, ax):
    x0, y0, h = 0.40 + HALF + 0.36, PANEL_BC["y0"], PANEL_BC["h"]
    w = HALF
    panel(ax, x0, y0, w, h, letter="C", title="Decision logic")

    grid_x = x0 + TEXT_POS["decision_grid_x_offset"]
    grid_w = w - TEXT_POS["decision_grid_x_offset"] - 0.32
    cw = (grid_w - 0.16) / 2
    col_x = [grid_x, grid_x + cw + 0.16]

    rbox(ax, grid_x, y0 + 0.74, grid_w, 0.34, fc="#DCEAF7", ec="none", r=0.07, z=2)
    txt(ax, grid_x + grid_w / 2, y0 + TEXT_POS["decision_top_header_y_offset"],
        "Separation relative to selected reference", size=TEXT_SIZE["decision_header"], color=INK,
        ha="center", z=5)
    for x, label in zip(col_x, ("Above reference",
                                "Not above reference")):
        txt(ax, x + cw / 2, y0 + TEXT_POS["decision_col_header_y_offset"], label, size=TEXT_SIZE["decision_col_header"], weight="bold",
            color=TEXT, ha="center", z=5)

    row_h = _decision_row_heights(cw)
    row_y = [y0 + 1.44, y0 + 1.44 + row_h[0] + 0.10]

    ax.text(x0 + TEXT_POS["decision_vertical_axis_x_offset"], (row_y[0] + row_y[1] + row_h[1]) / 2,
            "Reproducibility\n(full-refit stability)", rotation=90, ha="center",
            va="center", size=TEXT_SIZE["decision_axis"], color=TEXT, zorder=5, linespacing=1.4)

    for y, hh, top, bot in zip(row_y, row_h, ("High", "Low"),
                               ("(stable)", "(unstable)")):
        txt(ax, x0 + TEXT_POS["decision_side_label_x_offset"], y + hh / 2 - 0.11, top, size=TEXT_SIZE["decision_axis_main"], weight="bold",
            color=TEXT, ha="center", z=5)
        txt(ax, x0 + TEXT_POS["decision_side_label_x_offset"], y + hh / 2 + 0.11, bot, size=TEXT_SIZE["decision_axis_small"], color=MUTED,
            ha="center", z=5)

    for r_, c_, title, col, fill, edge, icon_key, items in DECISION_CELLS:
        icon = CELL_ICONS[icon_key]
        x, y, hh = col_x[c_], row_y[r_], row_h[r_]
        lw = 1.7 if r_ == 0 else 1.0
        rbox(ax, x, y, cw, hh, fc=fill, ec=edge, lw=lw, r=0.11, z=2)
        title_h = text_height(title, cw - 0.95, TEXT_SIZE["decision_cell_title"], 1.16)
        icon(ax, x + 0.36, y + 0.32, 0.28, col)
        para(ax, x + 0.66, y + 0.13, title, cw - 0.95,
             size=TEXT_SIZE["decision_cell_title"],
             color=col, weight="bold", leading=1.16)
        bullets(ax, x + 0.28, y + 0.28 + title_h + 0.11, items,
                cw - 0.52, size=TEXT_SIZE["decision_bullets"],
                color=TEXT, dot_color=col)


    ly = row_y[1] + row_h[1] + 0.24
    para(ax, x0 + 0.33, ly,
         "Both axes are diagnostics. Feature and model sensitivity must also be checked.",
         w - 0.65, size=TEXT_SIZE["decision_axis_small"],
         color=MUTED, leading=1.24)
    ly += text_height("Both axes are diagnostics. Feature and model sensitivity must also be checked.",
                      w - 0.65, TEXT_SIZE["decision_axis_small"], 1.24) + 0.11
    for label, nn, col, note in (
        ("Stroke", D.N_STROKE, BLUE,
         "above reference and stable; weighting-sensitive."),
        ("Sepsis", D.N_SEPSIS, ORANGE,
         "below all 100 references despite stability."),
    ):
        ax.add_patch(Circle((x0 + 0.50, ly + 0.16), 0.12, facecolor=col,
                            edgecolor="none", zorder=4))
        ly = rich_line(ax, x0 + 0.78, ly, f"{label} (n = {nn:,})", note,
                       w - 1.15, size=TEXT_SIZE["decision_cohort_note"],
                       lead_color=TEXT, fig=fig) + 0.12


def build(outdir="."):
    use_style()
    # Preserve editable TrueType glyph outlines in vector PDF/EPS exports.
    # Matplotlib's default PDF Type 3 fonts are unsuitable for many journals.
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    # panels B and C share a height: take whichever needs more, then size the
    # canvas to fit it, so text can grow without ever colliding.
    PANEL_BC["h"] = max(TEXT_POS["panel_bc_h"], panel_b_height(), panel_c_height())
    fig_h = max(H, PANEL_BC["y0"] + PANEL_BC["h"] + 0.40)
    fig, ax = canvas(W, fig_h)
    fig._crop_top_inches = 0.48
    # The manuscript caption supplies the figure number and title.
    panel_a(fig, ax)
    panel_b(fig, ax)
    panel_c(fig, ax)
    return save(fig, os.path.join(outdir, "fig1_workflow.png"))


if __name__ == "__main__":
    print(build())