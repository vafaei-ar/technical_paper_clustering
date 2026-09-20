"""
Figure 4 -- Selected phenotypes reveal distinct profiles in stroke and sepsis.

Each panel is a hand-laid-out heatmap: feature-group gutter, row labels, one
column per cluster, then a summary card per cluster.
"""
from __future__ import annotations

import os
import textwrap

import numpy as np
from matplotlib.colors import Normalize

from clustro_style import *
import clustro_data as D

W, H = 15.0, 11.45
PW = W - 0.80
HALF = (PW - 0.36) / 2

NORM = Normalize(-D.HEAT_VMAX, D.HEAT_VMAX)

HEADER_STYLE = {
    "blue": (FILL_BLUE, "#CBDFF2", "#17548A"),
    "orange": ("#FBE2CB", "#F2C69B", "#B5590F"),
    "red": ("#FBD5D8", "#F0B2B8", "#B3122B"),
}

GROUP_ICONS = {
    "people": ico_people,
    "droplet": ico_droplet,
    "kidneys": ico_kidneys,
    "heart": ico_heart,
    "gear": ico_gear,
    "cells": ico_cells,
    "lungs": ico_lungs,
}

SUMMARY = {
    "Stroke": [
        ("Preserved hematologic", D.STROKE_CLUSTERS[0][1], "blue", "people",
         ["Near-normal hematologic profile", "Lower comorbidity burden",
          "Lower glucose and creatinine"]),
        ("Renal-anemic", D.STROKE_CLUSTERS[1][1], "orange", "kidneys",
         ["Anemia, lower hematologic indices",
          "Higher creatinine",
          "More cardiovascular comorbidities"]),
        ("Hyperglycemic", D.STROKE_CLUSTERS[2][1], "red", "droplet",
         ["Markedly higher glucose", "Higher prevalence of diabetes",
          "More cardiovascular comorbidities"]),
    ],
    "Sepsis": [
        ("Reference / neutrophil", D.SEPSIS_CLUSTERS[0][1], "blue", "people",
         ["Reference inflammatory profile",
          "Lower organ dysfunction", "Fewer acute complications"]),
        ("IG-high", D.SEPSIS_CLUSTERS[1][1], "orange", "cells",
         ["Markedly elevated immature granulocytes",
          "Higher creatinine and anion gap",
          "More organ dysfunction"]),
        ("Eosinophil-lymphocyte", D.SEPSIS_CLUSTERS[2][1], "red", "blood",
         ["Higher eosinophils", "Higher lymphocytes", "Lower neutrophils",
          "Distinct inflammatory profile"]),
    ],
}


def _heat_panel(fig, ax, x0, y0, w, h, letter, cohort, subtitle, clusters, groups,
                summaries):
    panel(ax, x0, y0, w, h, letter=letter, title=cohort, subtitle=subtitle)

    ico_w, lab_w = 0.88, 2.02
    grid_x = x0 + 0.30 + ico_w + lab_w
    grid_w = w - (grid_x - x0) - 0.30
    cw = grid_w / len(clusters)

    head_y, head_h = y0 + 1.14, 0.84
    rows = [(rl, vals) for _, _, rr in groups for rl, vals in rr]
    lab_size = 8.6
    widest = max(measure(fig, rl, lab_size) for rl, _ in rows)
    if widest > lab_w - 0.22:
        lab_size = max(7.0, lab_size * (lab_w - 0.22) / widest)
    body_y = head_y + head_h + 0.06
    body_h = 3.98
    rh = body_h / len(rows)

    for j, (name, n, tone) in enumerate(clusters):
        fill, edge, tc = HEADER_STYLE[tone]
        hx = grid_x + j * cw
        rbox(ax, hx + 0.03, head_y, cw - 0.06, head_h, fc=fill, ec=edge,
             lw=0.9, r=0.08, z=2)
        fsize = 8.8
        lines = textwrap.wrap(name.replace("\n", " "),
                              chars_for(cw - 0.20, fsize))
        if len(lines) > 2:
            fsize = 7.8
            lines = textwrap.wrap(name.replace("\n", " "),
                                  chars_for(cw - 0.14, fsize))
        step = 0.185 * fsize / 8.8
        ty = head_y + (head_h - (len(lines) + 1) * step) / 2 + 0.10
        for line in lines:
            txt(ax, hx + cw / 2, ty, line, size=fsize, weight="bold", color=tc,
                ha="center", z=5)
            ty += step
        txt(ax, hx + cw / 2, ty, f"(n = {n:,})", size=8.0, color=tc,
            ha="center", z=5)

    r_i = 0
    for gname, gicon, grows in groups:
        gy = body_y + r_i * rh
        gh = len(grows) * rh
        rbox(ax, x0 + 0.30, gy + 0.02, ico_w, gh - 0.04, fc="#F4F7FA",
             ec="none", r=0.07, z=2)
        icon = GROUP_ICONS[gicon]
        isz = 0.26 if gh > 0.62 else 0.19
        icon(ax, x0 + 0.30 + ico_w / 2,
             gy + gh / 2 - (0.19 if gh > 0.62 else 0.13),
             isz, "#4A6076")
        glines = [
            ln
            for raw in gname.split("\n")
            for ln in textwrap.wrap(raw, chars_for(ico_w - 0.08, fs(7.0)))
        ]
        ty = gy + gh / 2 + 0.06 - 0.075 * (len(glines) - 2)
        for line in glines:
            txt(ax, x0 + 0.30 + ico_w / 2, ty, line, size=7.0,
                color="#3F5468", ha="center", weight="bold", z=5)
            ty += 0.155

        for rl, vals in grows:
            ry = body_y + r_i * rh
            sbox(ax, x0 + 0.30 + ico_w + 0.04, ry, lab_w - 0.08, rh,
                 fc="white", ec="#E6ECF2", lw=0.7, z=2)
            txt(ax, x0 + 0.30 + ico_w + 0.12, ry + rh / 2, rl,
                size=lab_size, color=TEXT, z=5)
            for j, v in enumerate(vals):
                sbox(ax, grid_x + j * cw, ry, cw, rh,
                     fc=HEAT_CMAP(NORM(v)), ec="#4A5563", lw=0.70, z=2)
            r_i += 1

    sy = body_y + body_h + 0.22
    sh = h - (sy - y0) - 0.30
    sw = (w - 0.60 - 0.30) / 3
    for j, (name, n, tone, icon_key, items) in enumerate(summaries):
        fill, edge, tc = HEADER_STYLE[tone]
        sx = x0 + 0.30 + j * (sw + 0.15)
        rbox(ax, sx, sy, sw, sh, fc="white", ec=edge, lw=1.0, r=0.10, z=2)
        rbox(ax, sx, sy, sw, 0.66, fc=fill, ec="none", r=0.10, z=3)
        sbox(ax, sx, sy + 0.48, sw, 0.18, fc=fill, ec="none", z=3)
        txt(ax, sx + sw / 2, sy + 0.21, name.replace("\n", " "),
            size=9.0, weight="bold", color=tc, ha="center", z=5)
        txt(ax, sx + sw / 2, sy + 0.44, f"(n = {n:,})", size=8.4,
            color=tc, ha="center", z=5)
        ic = {
            "people": ico_people,
            "kidneys": ico_kidneys,
            "droplet": ico_droplet,
            "cells": ico_cells,
            "blood": ico_blood_cell,
        }[icon_key]
        ic(ax, sx + 0.36, sy + 0.96, 0.32, tc)
        bullets(ax, sx + 0.16, sy + 1.18, items, sw - 0.34, size=7.8,
                color=TEXT, dot_color=tc, leading=1.28, gap=0.035)


def colorbar(fig, ax, y):
    cw, ch = 3.90, 0.26
    cx = W / 2 - cw / 2
    cax = axes_in(fig, cx, y, cw, ch)
    grad = np.linspace(-D.HEAT_VMAX, D.HEAT_VMAX, 512)[None, :]
    cax.imshow(
        grad,
        aspect="auto",
        cmap=HEAT_CMAP,
        norm=NORM,
        extent=[-D.HEAT_VMAX, D.HEAT_VMAX, 0, 1],
    )
    cax.set_yticks([])
    ticks = [t for t in (-2, -1, 0, 1, 2) if abs(t) <= D.HEAT_VMAX]
    cax.set_xticks(ticks)
    cax.tick_params(length=3, labelsize=12.5, colors=MUTED, pad=3)
    for sp in cax.spines.values():
        sp.set_color("#C3CEDA")
        sp.set_linewidth(0.8)

    txt(ax, cx - 0.22, y + ch / 2, "Lower than cohort reference", size=8.6,
        color=TEXT, ha="right")
    txt(ax, cx + cw + 0.22, y + ch / 2, "Higher than cohort reference",
        size=8.6, color=TEXT, ha="left")
    txt(
        ax,
        W / 2,
        y + ch + 0.50,
        "Standardized feature contrast: continuous = median difference/IQR; "
        "binary = standardized prevalence difference",
        size=8.2,
        color=MUTED,
        ha="center",
    )


def build(outdir="."):
    use_style()
    fig, ax = canvas(W, H)
    figure_title(
        ax,
        0.42,
        0.48,
        "Figure 4. Selected phenotypes reveal distinct profiles in stroke and sepsis",
        size=16.5,
    )

    _heat_panel(
        fig, ax, 0.40, 0.88, HALF, 9.44, "A", "Stroke",
        "Clinical and laboratory profiles by cluster",
        D.STROKE_CLUSTERS, D.STROKE_ROWS, SUMMARY["Stroke"],
    )
    _heat_panel(
        fig, ax, 0.40 + HALF + 0.36, 0.88, HALF, 9.44, "B", "Sepsis",
        "Inflammatory and organ dysfunction profiles by cluster",
        D.SEPSIS_CLUSTERS, D.SEPSIS_ROWS, SUMMARY["Sepsis"],
    )

    colorbar(fig, ax, 10.56)
    return save(fig, os.path.join(outdir, "fig4_phenotypes.png"))


if __name__ == "__main__":
    print(build())
