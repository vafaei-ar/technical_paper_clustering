"""
clustro_style.py
================
Shared design system for the Clustro figure set (Figures 1-5).

Design notes
------------
* Every figure is drawn on a single "canvas" axes whose data units are **inches**
  with the origin at the **top-left** (y increases downward).  That makes it easy
  to lay panels out the way you would in a drawing program, and it keeps rounded
  corners perfectly circular (the axes uses ``aspect="equal"``).
* Real plotting axes are added with :func:`axes_in`, which takes the same
  top-left inch coordinates and converts them to the figure-fraction rectangle
  that ``fig.add_axes`` expects.
* All colours live in one place so the whole set stays visually consistent.

Only numpy + matplotlib are required.
"""
from __future__ import annotations

import math
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

INK = "#10294B"
TEXT = "#20304A"
MUTED = "#5C6B80"
FAINT = "#8D9BAC"

BLUE = "#1C6FB4"
BLUE_DK = "#12507F"
ORANGE = "#E0701C"
ORANGE_DK = "#BE5A11"
GREEN = "#1E8E5A"
RED = "#C0392B"
CRIMSON = "#B3122B"

BORDER = "#C9DAE9"
GRID = "#E3E9F0"

FILL_BLUE = "#EAF2FB"
FILL_ORANGE = "#FDF0E4"
FILL_GREEN = "#E7F4EC"
FILL_RED = "#FDEBEA"
FILL_GRAY = "#F2F5F8"
FILL_HEAD = "#E4EEF8"

EDGE_BLUE = "#B7D2EA"
EDGE_ORANGE = "#F2C8A3"
EDGE_GREEN = "#A8D8BF"
EDGE_RED = "#F1BCB7"
EDGE_GRAY = "#DCE3EB"

STROKE = dict(color=BLUE, fill=FILL_BLUE, edge=EDGE_BLUE, label="Stroke")
SEPSIS = dict(color=ORANGE, fill=FILL_ORANGE, edge=EDGE_ORANGE, label="Sepsis")

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "clustro_heat",
    ["#2B5FA8", "#7CA8D4", "#C5D9EC", "#FFFFFF", "#F8CBB4", "#EE8A61", "#C2251B"],
)

FONT_SCALE = 1.56


def fs(size: float) -> float:
    return size * FONT_SCALE


def use_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Carlito", "Calibri", "Source Sans Pro", "Lato",
                                "Liberation Sans", "DejaVu Sans"],
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "axes.edgecolor": "#B7C3D1",
            "axes.linewidth": 0.9,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 13.2,
            "ytick.labelsize": 13.2,
            "text.color": TEXT,
            "axes.unicode_minus": False,
        }
    )


def canvas(width_in: float, height_in: float):
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(height_in, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig._canvas_size = (width_in, height_in)
    return fig, ax


def overlay(fig):
    W, H = fig._canvas_size
    ax = fig.add_axes([0, 0, 1, 1], zorder=30)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    return ax


def axes_in(fig, x, y, w, h, **kwargs):
    W, H = fig._canvas_size
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H], **kwargs)


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def rbox(ax, x, y, w, h, fc="white", ec=BORDER, lw=1.0, r=0.10, z=1,
         ls="-", alpha=1.0):
    r = min(r, w / 2 - 1e-6, h / 2 - 1e-6)
    p = FancyBboxPatch(
        (x + r, y + r), w - 2 * r, h - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        mutation_scale=1, facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=z, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def sbox(ax, x, y, w, h, fc="white", ec="none", lw=0.8, z=1):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=9, color=TEXT, weight="normal", ha="left", va="center",
        style="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=fs(size), color=color, fontweight=weight,
                   ha=ha, va=va, style=style, zorder=z, **kw)


def chars_for(width_in: float, size: float) -> int:
    return max(6, int(width_in * 72.0 / (0.50 * size)))


def para(ax, x, y, s, width_in, size=8, color=TEXT, weight="normal",
         leading=1.30, ha="left", va="top", z=5):
    eff = fs(size)
    lines = textwrap.wrap(s, chars_for(width_in, eff), break_long_words=False) or [""]
    dy = leading * eff / 72.0
    for i, line in enumerate(lines):
        txt(ax, x, y + (i + 0.5) * dy, line, size=size, color=color,
            weight=weight, ha=ha, va="center", z=z)
    return y + len(lines) * dy


def measure(fig, s, size=8, weight="normal"):
    fig.canvas.draw_idle()
    renderer = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=fs(size), fontweight=weight)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.width / fig.dpi


def text_height(s, width_in, size, leading=1.30):
    eff = fs(size)
    n = len(textwrap.wrap(s, chars_for(width_in, eff), break_long_words=False) or [""])
    return n * leading * eff / 72.0


def rich_line(ax, x, y, lead, rest, width_in, size=8, lead_color=TEXT,
              color=TEXT, leading=1.32, z=5, fig=None, sep=0.10):
    eff = fs(size)
    lead_w = (measure(fig, lead, size, "bold") if fig is not None
              else 0.52 * eff / 72.0 * len(lead)) + sep
    dy = leading * eff / 72.0
    txt(ax, x, y + 0.5 * dy, lead, size=size, weight="bold", color=lead_color, z=z)

    words = rest.split()
    first, i = [], 0
    limit = max(4, chars_for(width_in - lead_w, eff))
    while i < len(words) and len(" ".join(first + [words[i]])) <= limit:
        first.append(words[i])
        i += 1
    txt(ax, x + lead_w, y + 0.5 * dy, " ".join(first), size=size, color=color, z=z)

    cy = y + dy
    for line in textwrap.wrap(" ".join(words[i:]), chars_for(width_in, eff), break_long_words=False):
        txt(ax, x, cy + 0.5 * dy, line, size=size, color=color, z=z)
        cy += dy
    return cy


def bullets(ax, x, y, items, width_in, size=8, color=TEXT, dot_color=None,
            leading=1.32, gap=0.045, bullet="•", z=5, weight="normal"):
    dot_color = dot_color or color
    eff = fs(size)
    dy = leading * eff / 72.0
    indent = 0.115 * eff / 8.0
    cy = y
    for item in items:
        lines = textwrap.wrap(item, chars_for(width_in - indent, eff), break_long_words=False) or [""]
        for i, line in enumerate(lines):
            yy = cy + (i + 0.5) * dy
            if i == 0:
                txt(ax, x, yy, bullet, size=size, color=dot_color, z=z)
            txt(ax, x + indent, yy, line, size=size, color=color, weight=weight, z=z)
        cy += len(lines) * dy + gap
    return cy


def checklist(ax, x, y, items, width_in, size=7.5, color=TEXT, mark_color=GREEN,
              leading=1.32, gap=0.05):
    eff = fs(size)
    dy = leading * eff / 72.0
    indent = 0.20
    cy = y
    for item in items:
        lines = textwrap.wrap(item, chars_for(width_in - indent, eff), break_long_words=False) or [""]
        ico_check(ax, x + 0.055, cy + 0.5 * dy, 0.10, mark_color)
        for i, line in enumerate(lines):
            txt(ax, x + indent, cy + (i + 0.5) * dy, line, size=size, color=color)
        cy += len(lines) * dy + gap
    return cy


def arrow(ax, x1, y1, x2, y2, color=FAINT, lw=1.5, ms=9, z=4, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0), zorder=z)


def hline(ax, x1, x2, y, color=GRID, lw=1.0, z=2, ls="-"):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=z, ls=ls,
            solid_capstyle="butt")


def vline(ax, x, y1, y2, color=GRID, lw=1.0, z=2, ls="-"):
    ax.plot([x, x], [y1, y2], color=color, lw=lw, zorder=z, ls=ls,
            solid_capstyle="butt")


def figure_title(ax, x, y, s, size=18):
    txt(ax, x, y, s, size=size, weight="bold", color=INK, va="center")


def panel(ax, x, y, w, h, letter=None, title=None, subtitle=None,
          fc="white", ec=BORDER, lw=1.1, r=0.14, title_size=14.2,
          sub_size=10.0, letter_fc=FILL_GRAY, letter_color=INK, pad=0.28):
    rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=r, z=1)
    cy = y + pad
    if letter is not None:
        bx, by, bs = x + pad, cy, 0.40
        rbox(ax, bx, by, bs, bs, fc=letter_fc, ec="none", r=0.09, z=3)
        txt(ax, bx + bs / 2, by + bs / 2 + 0.012, letter, size=15, weight="bold",
            color=letter_color, ha="center", va="center", z=4)
        tx = bx + bs + 0.22
    else:
        tx = x + pad
    if title is not None:
        txt(ax, tx, cy + 0.20, title, size=title_size, weight="bold", color=INK)
    if subtitle is not None:
        txt(ax, tx, cy + 0.56, subtitle, size=sub_size, color=MUTED)
        return cy + 0.80
    return cy + 0.52


def legend_dot(ax, x, y, label, color, size=9, r=0.075, weight="normal",
               label_color=None):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=4))
    txt(ax, x + r + 0.12, y, label, size=size, color=label_color or TEXT,
        weight=weight)


def _poly(ax, pts, color, lw=0, ec="none", z=4, alpha=1.0):
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=ec,
                         linewidth=lw, zorder=z, alpha=alpha))


def ico_people(ax, cx, cy, s=0.22, color=BLUE):
    for dx, sc in ((-0.85, 0.72), (0.85, 0.72), (0.0, 1.0)):
        x = cx + dx * s * 0.55
        head_r = 0.26 * s * sc
        hy = cy - 0.30 * s * sc
        ax.add_patch(Circle((x, hy), head_r, facecolor=color, edgecolor="none", zorder=4))
        w, h = 0.70 * s * sc, 0.42 * s * sc
        rbox(ax, x - w / 2, hy + head_r * 1.15, w, h, fc=color, ec="none",
             r=min(0.09 * s, h / 2), z=4)


def ico_person(ax, cx, cy, s=0.22, color=BLUE):
    ax.add_patch(Circle((cx, cy - 0.32 * s), 0.27 * s, facecolor=color,
                        edgecolor="none", zorder=4))
    rbox(ax, cx - 0.38 * s, cy - 0.02 * s, 0.76 * s, 0.48 * s, fc=color, ec="none",
         r=0.10 * s, z=4)


def ico_db(ax, cx, cy, s=0.22, color=BLUE):
    w, h = 0.90 * s, 0.95 * s
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + 0.12 * s), w, h - 0.24 * s,
                           facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Ellipse((cx, cy - h / 2 + 0.12 * s), w, 0.26 * s,
                         facecolor=color, edgecolor="none", zorder=4))
    ax.add_patch(Ellipse((cx, cy + h / 2 - 0.12 * s), w, 0.26 * s,
                         facecolor=color, edgecolor="none", zorder=4))
    for k in (0.15, 0.45):
        ax.add_patch(Arc((cx, cy - h / 2 + 0.12 * s + k * h), w, 0.26 * s,
                         theta1=200, theta2=340, color="white", lw=0.8, zorder=5))


def ico_gear(ax, cx, cy, s=0.22, color=BLUE):
    R = 0.48 * s
    for i in range(8):
        a = i * math.pi / 4
        ax.add_patch(Rectangle((cx - 0.10 * s, cy - 0.10 * s), 0.20 * s, 0.68 * s,
                               angle=math.degrees(a), rotation_point=(cx, cy),
                               facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Circle((cx, cy), R, facecolor=color, edgecolor="none", zorder=4))
    ax.add_patch(Circle((cx, cy), 0.19 * s, facecolor="white", edgecolor="none", zorder=5))


def ico_doc(ax, cx, cy, s=0.22, color=BLUE):
    w, h = 0.78 * s, 0.98 * s
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fc="none", ec=color, lw=1.1,
         r=0.10 * s, z=4)
    for i, f in enumerate((0.28, 0.50, 0.72)):
        ww = w * 0.62
        hline(ax, cx - ww / 2, cx + ww / 2 * (0.6 if i == 2 else 1.0),
              cy - h / 2 + f * h, color=color, lw=1.0, z=5)


def ico_doc_bars(ax, cx, cy, s=0.30, color=BLUE):
    w, h = 0.86 * s, 1.08 * s
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fc="white", ec=color, lw=1.2,
         r=0.10 * s, z=4)
    hline(ax, cx - 0.30 * s, cx + 0.30 * s, cy - h / 2 + 0.20 * s, color=color, lw=1.0, z=5)
    hline(ax, cx - 0.30 * s, cx + 0.10 * s, cy - h / 2 + 0.34 * s, color=color, lw=1.0, z=5)
    base = cy + h / 2 - 0.16 * s
    for i, bh in enumerate((0.26, 0.44, 0.34)):
        bx = cx - 0.30 * s + i * 0.24 * s
        ax.add_patch(Rectangle((bx, base - bh * s), 0.15 * s, bh * s,
                               facecolor=color, edgecolor="none", zorder=5))


def ico_bars(ax, cx, cy, s=0.22, color=BLUE):
    base = cy + 0.45 * s
    for i, bh in enumerate((0.45, 0.75, 0.95)):
        ax.add_patch(Rectangle((cx - 0.45 * s + i * 0.33 * s, base - bh * s),
                               0.22 * s, bh * s, facecolor=color,
                               edgecolor="none", zorder=4))


def ico_shield(ax, cx, cy, s=0.22, color=BLUE):
    w, h = 0.80 * s, 1.0 * s
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
           (cx + w / 2, cy + 0.05 * h), (cx, cy + h / 2), (cx - w / 2, cy + 0.05 * h)]
    _poly(ax, pts, color, z=4)
    ax.plot([cx - 0.20 * s, cx - 0.04 * s, cx + 0.24 * s],
            [cy + 0.02 * s, cy + 0.18 * s, cy - 0.20 * s],
            color="white", lw=1.3, solid_capstyle="round", zorder=5)


def ico_curve(ax, cx, cy, s=0.22, color=BLUE):
    xs = np.linspace(-1, 1, 60)
    ys = np.exp(-4.5 * xs ** 2)
    px = cx + xs * 0.55 * s
    py = cy + 0.45 * s - ys * 0.85 * s
    ax.fill_between(px, py, cy + 0.45 * s, color=color, alpha=0.30, zorder=3, lw=0)
    ax.plot(px, py, color=color, lw=1.2, zorder=4)
    hline(ax, cx - 0.58 * s, cx + 0.58 * s, cy + 0.45 * s, color=color, lw=1.0, z=4)


def ico_refresh(ax, cx, cy, s=0.24, color=BLUE):
    R = 0.45 * s
    ax.add_patch(Arc((cx, cy), 2 * R, 2 * R, theta1=35, theta2=320,
                     color=color, lw=1.6, zorder=4))
    a = math.radians(35)
    tip = (cx + R * math.cos(a), cy - R * math.sin(a))
    _poly(ax, [(tip[0] - 0.05 * s, tip[1] - 0.20 * s),
               (tip[0] + 0.22 * s, tip[1] - 0.02 * s),
               (tip[0] - 0.08 * s, tip[1] + 0.14 * s)], color, z=5)


def ico_sheets(ax, cx, cy, s=0.30, color="#BBD2E6", ec="#7FA6C9"):
    for i, dx in enumerate((-0.22, 0.0, 0.22)):
        rbox(ax, cx - 0.42 * s + dx * s, cy - 0.52 * s + dx * s * 0.55,
             0.85 * s, 1.0 * s, fc="white" if i == 2 else color, ec=ec, lw=0.9,
             r=0.08 * s, z=3 + i)
    x0, y0 = cx - 0.42 * s + 0.22 * s, cy - 0.52 * s + 0.12 * s
    for k in range(4):
        hline(ax, x0 + 0.12 * s, x0 + 0.72 * s, y0 + (0.22 + 0.18 * k) * s,
              color=ec, lw=0.7, z=7)


def ico_scatter(ax, cx, cy, s=0.30, color=BLUE, n=16, seed=3, spread=0.55):
    rng = np.random.default_rng(seed)
    px = np.clip(rng.normal(0, 0.22, n), -0.45, 0.45) * s
    py = np.clip(rng.normal(0, 0.16, n), -0.32, 0.32) * s
    ax.scatter(cx + px, cy + py, s=4.0, color=color, zorder=4, linewidths=0)


def ico_clusters(ax, cx, cy, s=0.30, colors=(BLUE, ORANGE, "#7E57C2"), seed=7):
    rng = np.random.default_rng(seed)
    offs = [(-0.27, -0.13), (0.24, -0.17), (0.02, 0.20)]
    for (dx, dy), c in zip(offs, colors):
        px = cx + dx * s + np.clip(rng.normal(0, 0.06, 7), -0.11, 0.11) * s
        py = cy + dy * s + np.clip(rng.normal(0, 0.05, 7), -0.10, 0.10) * s
        ax.scatter(px, py, s=4.0, color=c, zorder=4, linewidths=0)


def ico_dendro(ax, cx, cy, s=0.30, color="#9AA7B6",
               leaf_colors=(BLUE, ORANGE, "#7E57C2", GREEN)):
    xs = np.linspace(cx - 0.55 * s, cx + 0.55 * s, 4)
    base = cy + 0.48 * s
    for x, c in zip(xs, leaf_colors):
        ax.add_patch(Circle((x, base + 0.10 * s), 0.07 * s, facecolor=c,
                            edgecolor="none", zorder=5))
        ax.plot([x, x], [base, base - 0.22 * s], color=color, lw=1.0, zorder=4)
    for (a, b), hgt in (((0, 1), 0.22), ((2, 3), 0.22)):
        ax.plot([xs[a], xs[b]], [base - hgt * s] * 2, color=color, lw=1.0, zorder=4)
    m1, m2 = xs[:2].mean(), xs[2:].mean()
    ax.plot([m1, m1], [base - 0.22 * s, base - 0.48 * s], color=color, lw=1.0, zorder=4)
    ax.plot([m2, m2], [base - 0.22 * s, base - 0.48 * s], color=color, lw=1.0, zorder=4)
    ax.plot([m1, m2], [base - 0.48 * s] * 2, color=color, lw=1.0, zorder=4)


def ico_scale(ax, cx, cy, s=0.24, color=BLUE):
    ax.plot([cx, cx], [cy - 0.45 * s, cy + 0.45 * s], color=color, lw=1.4, zorder=4)
    ax.plot([cx - 0.55 * s, cx + 0.55 * s], [cy - 0.30 * s] * 2, color=color,
            lw=1.4, zorder=4)
    hline(ax, cx - 0.30 * s, cx + 0.30 * s, cy + 0.45 * s, color=color, lw=1.4, z=4)
    for sx in (-1, 1):
        x = cx + sx * 0.55 * s
        _poly(ax, [(x - 0.26 * s, cy - 0.06 * s), (x + 0.26 * s, cy - 0.06 * s),
                   (x + 0.13 * s, cy + 0.20 * s), (x - 0.13 * s, cy + 0.20 * s)],
              color, z=4)


def ico_check(ax, cx, cy, s=0.12, color=GREEN, lw=1.5):
    ax.plot([cx - 0.85 * s, cx - 0.22 * s, cx + 0.95 * s],
            [cy + 0.02 * s, cy + 0.62 * s, cy - 0.70 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=6)


def ico_check_circle(ax, cx, cy, s=0.22, color=GREEN):
    ax.add_patch(Circle((cx, cy), 0.52 * s, facecolor=color, edgecolor="none", zorder=4))
    ico_check(ax, cx, cy, 0.24 * s, "white", lw=1.6)


def ico_warning(ax, cx, cy, s=0.22, color="#E8871A"):
    h = 0.95 * s
    _poly(ax, [(cx, cy - h / 2), (cx + 0.58 * s, cy + h / 2),
               (cx - 0.58 * s, cy + h / 2)], color, z=4)
    ax.plot([cx, cx], [cy - 0.14 * s, cy + 0.20 * s], color="white", lw=1.5,
            solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((cx, cy + 0.33 * s), 0.055 * s, facecolor="white",
                        edgecolor="none", zorder=5))


def ico_cross(ax, cx, cy, s=0.22, color=RED, lw=2.2):
    d = 0.42 * s
    ax.plot([cx - d, cx + d], [cy - d, cy + d], color=color, lw=lw,
            solid_capstyle="round", zorder=5)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=color, lw=lw,
            solid_capstyle="round", zorder=5)


def ico_trophy(ax, cx, cy, s=0.22, color=GREEN):
    _poly(ax, [(cx - 0.34 * s, cy - 0.52 * s), (cx + 0.34 * s, cy - 0.52 * s),
               (cx + 0.20 * s, cy + 0.10 * s), (cx - 0.20 * s, cy + 0.10 * s)],
          color, z=4)
    for sx in (-1, 1):
        ax.add_patch(Arc((cx + sx * 0.40 * s, cy - 0.30 * s), 0.34 * s, 0.40 * s,
                         theta1=-90 if sx > 0 else 90,
                         theta2=90 if sx > 0 else 270,
                         color=color, lw=1.5, zorder=4))
    ax.plot([cx, cx], [cy + 0.10 * s, cy + 0.34 * s], color=color, lw=2.0,
            zorder=4, solid_capstyle="butt")
    rbox(ax, cx - 0.30 * s, cy + 0.34 * s, 0.60 * s, 0.16 * s, fc=color,
         ec="none", r=0.05 * s, z=4)


def ico_droplet(ax, cx, cy, s=0.22, color=BLUE):
    th = np.linspace(0, 2 * np.pi, 80)
    R = 0.42 * s
    x = cx + R * np.sin(th)
    y = cy + 0.18 * s + R * np.cos(th)
    _poly(ax, list(zip(x, y)), color, z=4)
    _poly(ax, [(cx, cy - 0.55 * s), (cx + 0.40 * s, cy + 0.20 * s),
               (cx - 0.40 * s, cy + 0.20 * s)], color, z=4)


def ico_heart(ax, cx, cy, s=0.22, color="#4A5C72"):
    t = np.linspace(0, 2 * np.pi, 160)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    _poly(ax, list(zip(cx + x / 16 * 0.50 * s, cy + y / 16 * 0.50 * s)), color, z=4)


def ico_kidneys(ax, cx, cy, s=0.22, color="#4A5C72"):
    for sx in (-1, 1):
        ax.add_patch(Ellipse((cx + sx * 0.30 * s, cy), 0.42 * s, 0.72 * s,
                             angle=sx * 12, facecolor=color, edgecolor="none", zorder=4))
        ax.add_patch(Ellipse((cx + sx * 0.46 * s, cy), 0.24 * s, 0.44 * s,
                             angle=sx * 12, facecolor="white", edgecolor="none", zorder=5))


def ico_lungs(ax, cx, cy, s=0.22, color="#4A5C72"):
    for sx in (-1, 1):
        rbox(ax, cx + sx * 0.10 * s - (0.34 * s if sx < 0 else 0), cy - 0.18 * s,
             0.34 * s, 0.62 * s, fc=color, ec="none", r=0.14 * s, z=4)
    ax.plot([cx, cx], [cy - 0.48 * s, cy + 0.10 * s], color=color, lw=1.4, zorder=5)


def ico_cells(ax, cx, cy, s=0.22, color=BLUE, seed=11):
    for dx, dy, r in ((-0.34, -0.20, 0.22), (0.30, -0.26, 0.17),
                      (0.02, 0.24, 0.26), (0.42, 0.24, 0.15)):
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), r * s, facecolor="white",
                            edgecolor=color, lw=1.1, zorder=4))
        ax.add_patch(Circle((cx + dx * s, cy + dy * s), r * s * 0.42,
                            facecolor=color, edgecolor="none", zorder=5))


def ico_blood_cell(ax, cx, cy, s=0.22, color="#C2184B"):
    ax.add_patch(Circle((cx, cy), 0.46 * s, facecolor="white", edgecolor=color,
                        lw=1.6, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.20 * s, facecolor=color, edgecolor="none",
                        alpha=0.55, zorder=5))


def ico_glucose(ax, cx, cy, s=0.22, color="#C2184B"):
    ico_droplet(ax, cx, cy, s, color)


def tidy_axes(ax, grid="y", spines=("left", "bottom"), grid_color=GRID):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in spines)
        ax.spines[side].set_color("#B7C3D1")
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=grid_color, lw=0.9)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=grid_color, lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.9, colors=MUTED, labelsize=13.3)
