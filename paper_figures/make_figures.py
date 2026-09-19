from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

INK = "#10294B"
TEXT = "#20304A"
MUTED = "#5C6B80"
BLUE = "#1C6FB4"
ORANGE = "#E0701C"
GREEN = "#1E8E5A"
RED = "#C0392B"
GRAY = "#8795A5"
LIGHT = "#F5F8FB"
BORDER = "#CAD9E7"
GRID = "#E6ECF2"

N_STROKE = 9835
N_SEPSIS = 15842


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#B8C5D1",
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def rounded(ax, xy, w, h, fc="white", ec=BORDER, lw=1.0, r=0.025, z=0):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        transform=ax.transAxes,
        fc=fc,
        ec=ec,
        lw=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def clean_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.7, color=GRID)
    ax.tick_params(labelsize=9)


def save(fig, outdir: Path, stem: str, pdf: bool, dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    if pdf:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png


def load_inputs(reporting: Path):
    files = {
        "stroke_audit": reporting / "tableS_candidate_model_audit_stroke.csv",
        "sepsis_audit": reporting / "tableS_candidate_model_audit_sepsis.csv",
        "null_ext": reporting / "null_reference_silhouette_extended_summary.csv",
        "null_nested": reporting / "null_reference_summary.csv",
        "sens": reporting / "figure4_sensitivity_values.csv",
        "stroke_repr": reporting / "representation_k_sensitivity_stroke.csv",
        "sepsis_repr": reporting / "representation_k_sensitivity_sepsis.csv",
        "redundancy": reporting / "sepsis_wbc_redundancy_sensitivity.csv",
        "balanced": reporting / "balanced_block_summary.csv",
        "stroke_sizes": reporting / "table_cluster_sizes_stroke.csv",
        "sepsis_sizes": reporting / "table_cluster_sizes_sepsis.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing reporting outputs:\n" + "\n".join(missing))
    return {k: pd.read_csv(v) for k, v in files.items()}


def candidate_rows(df):
    fam = {
        ("pca", "kmeans"): "PCA k-means",
        ("none", "kmeans"): "Raw k-means",
        ("none", "agglomerative"): "Raw agglomerative",
    }
    use = df[df["mean_subsample_ari"].notna()].copy()
    rows = []
    for _, r in use.iterrows():
        key = (str(r["reduction"]).lower(), str(r["clusterer"]).lower())
        if key in fam:
            rows.append(
                {
                    "family": fam[key],
                    "k": int(r["k"]),
                    "silhouette": float(r["silhouette_mean"]),
                    "ari": float(r["mean_subsample_ari"]),
                }
            )
    return pd.DataFrame(rows)


def figure1(outdir, pdf, dpi):
    fig = plt.figure(figsize=(14, 10.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(
        0.035,
        0.965,
        "Figure 1. Clustro workflow and decision logic",
        fontsize=21,
        fontweight="bold",
        color=INK,
        va="top",
    )

    rounded(ax, (0.025, 0.54), 0.95, 0.36)
    ax.text(0.04, 0.88, "A", fontsize=18, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(
        0.075,
        0.88,
        "End-to-end workflow",
        fontsize=15,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
    )
    cards = [
        ("1", "Cohort assembly", f"Stroke\nn={N_STROKE:,}\n\nSepsis\nn={N_SEPSIS:,}"),
        ("2", "Preprocessing", "Cleaning\nfeature preparation\nmissingness handling"),
        ("3", "Candidate grid", "Raw / PCA\nk-means / GMM /\nagglomerative\nk=2...10"),
        ("4", "Full-refit stability", "Repeated 80%\nsubsamples\n50 repeats"),
        ("5", "Evaluation", "Separation\nstability\ncluster size\nnull reference"),
        ("6", "Sensitivity", "Raw vs PCA\nbalanced blocks\nfeature redundancy"),
        ("7", "Interpretation", "Profiles\nrobustness\nfigures + tables\nopen code"),
    ]
    x0, y0, total_w, gap = 0.045, 0.585, 0.91, 0.012
    cw = (total_w - 6 * gap) / 7
    for i, (n, title, body) in enumerate(cards):
        x = x0 + i * (cw + gap)
        rounded(ax, (x, y0), cw, 0.245, fc="white", ec="#D6E1EA")
        ax.add_patch(
            plt.Circle(
                (x + 0.02, y0 + 0.213),
                0.014,
                transform=ax.transAxes,
                color="#5D86AE",
            )
        )
        ax.text(
            x + 0.02,
            y0 + 0.213,
            n,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        ax.text(
            x + 0.042,
            y0 + 0.225,
            title,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            color=INK,
            va="top",
        )
        ax.text(
            x + 0.012,
            y0 + 0.165,
            body,
            transform=ax.transAxes,
            fontsize=7.4,
            color=TEXT,
            va="top",
            linespacing=1.35,
        )
        if i < 6:
            ax.annotate(
                "",
                xy=(x + cw + gap * 0.9, y0 + 0.12),
                xytext=(x + cw + 0.002, y0 + 0.12),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#91A6B8"),
            )

    rounded(ax, (0.025, 0.055), 0.455, 0.445)
    ax.text(0.04, 0.475, "B", fontsize=18, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(
        0.075,
        0.475,
        "Why Clustro instead of naive clustering",
        fontsize=14,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
    )
    rounded(ax, (0.05, 0.095), 0.18, 0.32, fc="#F2F4F6")
    rounded(ax, (0.26, 0.095), 0.19, 0.32, fc="#EEF5FB")
    ax.text(
        0.14,
        0.39,
        "Naive workflow",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.355,
        0.39,
        "Clustro",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.14,
        0.34,
        "One-shot clustering\n↓\nDirect interpretation\n↓\nPublish",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=TEXT,
        linespacing=1.45,
    )
    ax.text(
        0.355,
        0.34,
        "Candidate comparison\n↓\nFull-refit resampling\n↓\nNull benchmarking\n↓\nSensitivity checks",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=TEXT,
        linespacing=1.32,
    )
    rounded(ax, (0.065, 0.11), 0.15, 0.055, fc="#FDEBEA", ec="#F2B9B4")
    rounded(ax, (0.275, 0.11), 0.16, 0.055, fc="#E8F5ED", ec="#AFD9C2")
    ax.text(
        0.14,
        0.137,
        "Risk of spurious structure",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.4,
        fontweight="bold",
        color=RED,
    )
    ax.text(
        0.355,
        0.137,
        "Auditable interpretation",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.4,
        fontweight="bold",
        color=GREEN,
    )

    rounded(ax, (0.50, 0.055), 0.475, 0.445)
    ax.text(0.515, 0.475, "C", fontsize=18, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(
        0.55,
        0.475,
        "Decision logic",
        fontsize=14,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
    )
    qx, qy, qw, qh = 0.575, 0.13, 0.34, 0.27
    quadrants = [
        (qx, qy + qh / 2, qw / 2, qh / 2, "#FDF1E4", ORANGE, "Caution", "Null-like\nbut stable"),
        (qx + qw / 2, qy + qh / 2, qw / 2, qh / 2, "#E8F5ED", GREEN, "Promising", "Beyond null\nand stable"),
        (qx, qy, qw / 2, qh / 2, "#FCEDEE", RED, "Reject", "Null-like\nand unstable"),
        (qx + qw / 2, qy, qw / 2, qh / 2, "#FCEDEE", RED, "Unstable", "Beyond null\nbut unstable"),
    ]
    for x, y, w, h, fc, c, title, sub in quadrants:
        rounded(ax, (x, y), w, h, fc=fc, ec=c, lw=0.9)
        ax.text(
            x + w / 2,
            y + h * 0.63,
            title,
            transform=ax.transAxes,
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=c,
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            sub,
            transform=ax.transAxes,
            ha="center",
            fontsize=8.3,
            color=TEXT,
        )
    ax.scatter([qx + qw * 0.76], [qy + qh * 0.75], transform=ax.transAxes, s=90, color=BLUE, zorder=5)
    ax.text(
        qx + qw * 0.80,
        qy + qh * 0.75,
        "Stroke",
        transform=ax.transAxes,
        va="center",
        fontsize=9,
        color=BLUE,
        fontweight="bold",
    )
    ax.scatter([qx + qw * 0.26], [qy + qh * 0.75], transform=ax.transAxes, s=90, color=ORANGE, zorder=5)
    ax.text(
        qx + qw * 0.30,
        qy + qh * 0.75,
        "Sepsis",
        transform=ax.transAxes,
        va="center",
        fontsize=9,
        color=ORANGE,
        fontweight="bold",
    )
    ax.text(
        qx + qw / 2,
        qy - 0.028,
        "Separation vs null reference  →",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    ax.text(
        qx - 0.035,
        qy + qh / 2,
        "Reproducibility\n↑",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        color=MUTED,
    )
    return save(fig, outdir, "fig1_workflow", pdf, dpi)


def figure2(data, outdir, pdf, dpi):
    stroke = candidate_rows(data["stroke_audit"])
    sepsis = candidate_rows(data["sepsis_audit"])
    fig = plt.figure(figsize=(14, 9.5))
    fig.suptitle(
        "Figure 2. Model-selection landscape and selected solutions",
        x=0.035,
        y=0.985,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.2, 1.3],
        hspace=0.34,
        wspace=0.18,
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )
    shape = {"PCA k-means": "o", "Raw k-means": "s", "Raw agglomerative": "^"}
    color_k = {2: "#377EB8", 3: "#E0701C", 4: "#3FA65A"}
    for j, (name, df) in enumerate([("Stroke", stroke), ("Sepsis", sepsis)]):
        ax = fig.add_subplot(gs[0, j])
        clean_axis(ax)
        for _, r in df.iterrows():
            ax.scatter(
                r.silhouette,
                r.ari,
                s=95,
                marker=shape[r.family],
                color=color_k.get(r.k, GRAY),
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
            ax.annotate(
                f"k={r.k}",
                (r.silhouette, r.ari),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.3,
            )
        sel = df[(df.family == "PCA k-means") & (df.k == 3)].iloc[0]
        ax.scatter(
            sel.silhouette,
            sel.ari,
            s=260,
            marker="*",
            color="#D62828",
            edgecolor="black",
            linewidth=0.9,
            zorder=6,
        )
        ax.annotate(
            "Selected k=3",
            (sel.silhouette, sel.ari),
            xytext=(8, 18),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#9F1D25",
            arrowprops=dict(arrowstyle="->", color="#9F1D25", lw=1),
        )
        ax.set_xlabel("Silhouette score", fontsize=11)
        ax.set_ylabel("Mean full-refit subsample ARI", fontsize=11)
        ax.set_title(f"{'AB'[j]}  {name}", loc="left", fontsize=16, fontweight="bold", color=INK)
        if name == "Stroke":
            ax.set_ylim(0.15, 1.02)
        else:
            ax.set_ylim(0.80, 0.99)

    axc = fig.add_subplot(gs[1, :])
    axc.axis("off")
    rounded(axc, (0.00, 0.03), 1, 0.94)
    axc.text(
        0.01,
        0.90,
        "C  Selection rationale",
        transform=axc.transAxes,
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    cards = [
        (
            0.02,
            "Stroke",
            "k=3 selected as a separation-stability compromise",
            BLUE,
            "k=2: higher separation\nk=3: strong stability + good separation\nk=4: highest mean stability",
        ),
        (
            0.39,
            "Sepsis",
            "k=3 selected as the most reproducible compromise",
            ORANGE,
            "k=2: unstable in some resamples\nk=3: consistently high stability\nk=4: slightly higher separation, weaker worst-case stability",
        ),
        (
            0.76,
            "Principle",
            "No single metric determines the model",
            INK,
            "Selection integrates separation,\nfull-pipeline reproducibility,\ncluster size, and sensitivity analyses.",
        ),
    ]
    for x, head, sub, c, body in cards:
        rounded(axc, (x, 0.12), 0.22, 0.62, fc=LIGHT, ec="#D8E4EE")
        axc.text(
            x + 0.015,
            0.66,
            head,
            transform=axc.transAxes,
            fontsize=11.5,
            fontweight="bold",
            color=c,
        )
        axc.text(
            x + 0.015,
            0.56,
            sub,
            transform=axc.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=TEXT,
            wrap=True,
        )
        axc.text(
            x + 0.015,
            0.42,
            body,
            transform=axc.transAxes,
            fontsize=8.3,
            color=TEXT,
            va="top",
            linespacing=1.35,
        )
    return save(fig, outdir, "fig2_model_selection", pdf, dpi)


def figure3(data, outdir, pdf, dpi):
    ext = data["null_ext"].set_index("dataset")
    nested = data["null_nested"].set_index("dataset")
    fig = plt.figure(figsize=(14, 9.5))
    fig.suptitle(
        "Figure 3. Null-reference benchmarking distinguishes robust structure from plausible artifacts",
        x=0.035,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.4, 1.65],
        hspace=0.26,
        wspace=0.20,
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )
    positions = {}
    for j, (name, c) in enumerate([("stroke", BLUE), ("sepsis", ORANGE)]):
        r = ext.loc[name]
        ax = fig.add_subplot(gs[0, j])
        clean_axis(ax, grid=False)
        lo = float(r.null_silhouette_p05)
        mean = float(r.null_silhouette_mean)
        hi = float(r.null_silhouette_p95)
        obs = float(r.real_silhouette)
        mx = float(r.null_silhouette_max)
        ax.axvspan(lo, hi, color="#DCE7F2", alpha=0.85, label="Null 5th-95th percentile")
        ax.axvline(mean, color=GRAY, ls="--", lw=2, label=f"Null mean = {mean:.3f}")
        ax.axvline(obs, color=c, ls="--", lw=2.5, label=f"Observed = {obs:.3f}")
        ax.scatter([mx], [0.25], marker="|", s=350, color=GRAY, zorder=5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Silhouette score", fontsize=11)
        margin = max((hi - lo) * 0.7, 0.015)
        ax.set_xlim(min(lo, obs) - margin, max(mx, obs) + margin)
        ttl = "Stroke" if name == "stroke" else "Sepsis"
        ax.set_title(f"{'AB'[j]}  {ttl}", loc="left", fontsize=16, fontweight="bold", color=INK)
        msg = (
            "Observed separation exceeds null reference"
            if name == "stroke"
            else "Observed separation does not exceed null reference"
        )
        boxc = "#E8F5ED" if name == "stroke" else "#FDF1E4"
        edge = GREEN if name == "stroke" else ORANGE
        ax.text(
            0.02,
            0.84,
            msg,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=edge,
            bbox=dict(boxstyle="round,pad=0.35", fc=boxc, ec=edge, lw=0.8),
        )
        ax.text(
            0.98,
            0.78,
            f"{int(r.n_null_ge_real)}/{int(r.null_repeats)} null replicates >= observed\n"
            f"empirical p = {float(r.empirical_p_silhouette):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color=TEXT,
        )
        ax.legend(frameon=False, loc="lower left", fontsize=8.5)
        positions[name] = (
            obs - mean,
            float(nested.loc[name, "real_mean_full_refit_subsample_ari"]),
        )

    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0.90, 1.0)
    clean_axis(ax)
    ax.axvline(0, color=GRAY, lw=1.2)
    ax.set_xlabel("Observed silhouette - null mean", fontsize=10.5)
    ax.set_ylabel("Mean full-refit ARI", fontsize=10.5)
    ax.set_title(
        "C  Decision map: separation vs reproducibility",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    for name, c, label in [("stroke", BLUE, "Stroke"), ("sepsis", ORANGE, "Sepsis")]:
        x, y = positions[name]
        ax.scatter(x, y, s=150, color=c, edgecolor="white", linewidth=1, zorder=5)
        ax.annotate(
            label,
            (x, y),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=c,
        )
    ax.text(
        0.98,
        0.95,
        "Higher than null + stable",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.95,
        "Null-like + stable",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=ORANGE,
        fontweight="bold",
    )

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    rounded(ax2, (0, 0), 1, 1)
    ax2.text(
        0.04,
        0.90,
        "Key takeaways",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    s = ext.loc["stroke"]
    e = ext.loc["sepsis"]
    ax2.text(
        0.05,
        0.72,
        "Stroke",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        color=BLUE,
    )
    ax2.text(
        0.05,
        0.61,
        f"Observed {s.real_silhouette:.3f} vs null mean {s.null_silhouette_mean:.3f}\n"
        "0/100 null replicates reached the observed value.",
        transform=ax2.transAxes,
        fontsize=9.5,
        color=TEXT,
        linespacing=1.35,
    )
    ax2.text(
        0.05,
        0.40,
        "Sepsis",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        color=ORANGE,
    )
    ax2.text(
        0.05,
        0.29,
        f"Observed {e.real_silhouette:.3f} vs null mean {e.null_silhouette_mean:.3f}\n"
        "100/100 null replicates reached or exceeded the observed value.",
        transform=ax2.transAxes,
        fontsize=9.5,
        color=TEXT,
        linespacing=1.35,
    )
    ax2.text(
        0.05,
        0.08,
        "High clustering stability alone is not evidence of discrete latent structure.",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    return save(fig, outdir, "fig3_null_reference", pdf, dpi)


def figure4(data, reporting, outdir, pdf, dpi, phenotype_image=None):
    if phenotype_image is None:
        candidates = [
            reporting / "figure3_phenotype_profiles_revised.png",
            reporting / "figure3_phenotype_heatmaps_publication.png",
        ]
        phenotype_image = next((p for p in candidates if p.exists()), None)
    if phenotype_image is None or not Path(phenotype_image).exists():
        raise FileNotFoundError(
            "Phenotype heatmap image not found. Run regenerate_methods_revision_reporting.sh "
            "first or pass --phenotype-image."
        )
    img = plt.imread(phenotype_image)
    fig = plt.figure(figsize=(14, 10.5))
    fig.suptitle(
        "Figure 4. Selected cluster profiles are interpretable, but robustness differs by cohort",
        x=0.035,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.35, 1],
        wspace=0.06,
        left=0.04,
        right=0.98,
        top=0.91,
        bottom=0.06,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "A  Data-faithful phenotype profiles",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
        pad=10,
    )
    axr = fig.add_subplot(gs[0, 1])
    axr.axis("off")
    rounded(axr, (0, 0), 1, 1)
    axr.text(
        0.05,
        0.94,
        "B  Interpretation",
        transform=axr.transAxes,
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    cards = [
        (
            0.05,
            0.58,
            0.90,
            0.28,
            "Stroke",
            "#EAF2FB",
            BLUE,
            [
                "Preserved hematologic: lower overall burden.",
                "Renal-anemic: lower hematologic indices and higher creatinine.",
                "Hyperglycemic: higher glucose and diabetes burden.",
            ],
            "Membership shifts under balanced-block weighting (ARI 0.543).",
        ),
        (
            0.05,
            0.17,
            0.90,
            0.33,
            "Sepsis",
            "#FDF0E4",
            ORANGE,
            [
                "Reference/neutrophil: large reference group.",
                "IG-high: immature-granulocyte and organ-dysfunction signal.",
                "Eosinophil-lymphocyte: eosinophil/lymphocyte enrichment.",
            ],
            "Descriptive only: balanced-block ARI 0.141; WBC feature sensitivity ARI 0.149-0.366.",
        ),
    ]
    for x, y, w, h, title, fc, c, bullets, caution in cards:
        rounded(axr, (x, y), w, h, fc=fc, ec=c, lw=0.9)
        axr.text(
            x + 0.03,
            y + h - 0.05,
            title,
            transform=axr.transAxes,
            fontsize=12,
            fontweight="bold",
            color=c,
            va="top",
        )
        yy = y + h - 0.12
        for b in bullets:
            axr.text(
                x + 0.05,
                yy,
                "• " + b,
                transform=axr.transAxes,
                fontsize=8.8,
                color=TEXT,
                va="top",
                wrap=True,
            )
            yy -= 0.055
        axr.text(
            x + 0.03,
            y + 0.05,
            caution,
            transform=axr.transAxes,
            fontsize=8.4,
            color=RED,
            fontweight="bold",
            va="bottom",
            wrap=True,
        )
    axr.text(
        0.05,
        0.08,
        "Main message",
        transform=axr.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    axr.text(
        0.05,
        0.03,
        "Profiles can look clinically coherent even when cluster membership is specification-sensitive.",
        transform=axr.transAxes,
        fontsize=8.8,
        color=TEXT,
        wrap=True,
    )
    return save(fig, outdir, "fig4_phenotypes", pdf, dpi)


def figure5(data, outdir, pdf, dpi):
    audit_s = data["stroke_audit"]
    audit_e = data["sepsis_audit"]

    def kvals(df, col):
        q = (
            df[
                (df.reduction == "pca")
                & (df.clusterer == "kmeans")
                & df.k.isin([2, 3, 4])
            ]
            .set_index("k")
            .reindex([2, 3, 4])
        )
        return q[col].astype(float).tolist()

    sil = {
        "Stroke": kvals(audit_s, "silhouette_mean"),
        "Sepsis": kvals(audit_e, "silhouette_mean"),
    }
    ari = {
        "Stroke": kvals(audit_s, "mean_subsample_ari"),
        "Sepsis": kvals(audit_e, "mean_subsample_ari"),
    }
    sens = data["sens"]

    def sv(cohort, name):
        return float(
            sens[(sens.cohort == cohort) & (sens.sensitivity == name)].ari.iloc[0]
        )

    fig = plt.figure(figsize=(14, 10.2))
    fig.suptitle(
        "Figure 5. Robustness differs across reasonable clustering specifications",
        x=0.035,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.32,
        wspace=0.22,
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.10,
    )

    ax = fig.add_subplot(gs[0, 0])
    clean_axis(ax)
    ax.set_title("A  k sensitivity", loc="left", fontsize=15, fontweight="bold", color=INK)
    x = np.array([2, 3, 4])
    for name, c in [("Stroke", BLUE), ("Sepsis", ORANGE)]:
        ax.plot(x, ari[name], "-o", lw=2, ms=7, color=c, label=name)
    ax.axvspan(2.8, 3.2, color="#EEF4FA")
    ax.set_xticks(x)
    ax.set_ylim(0.75, 1.01)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean full-refit ARI")
    ax.legend(frameon=False)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#C8D2DD")
    for name, c in [("Stroke", BLUE), ("Sepsis", ORANGE)]:
        ax2.plot(x, sil[name], "--", lw=1.3, color=c, alpha=0.55)
    ax2.set_ylabel("Silhouette score", color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=8)

    ax = fig.add_subplot(gs[0, 1])
    clean_axis(ax, grid=True)
    ax.set_title(
        "B  Representation sensitivity",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    vals = [sv("Stroke", "Raw vs PCA"), sv("Sepsis", "Raw vs PCA")]
    ax.barh([0, 1], vals, color=[BLUE, ORANGE], height=0.45)
    ax.set_yticks([0, 1], ["Stroke", "Sepsis"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("ARI vs primary PCA k=3 (1 = identical)")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    clean_axis(ax, grid=True)
    ax.set_title(
        "C  Balanced-block weighting",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    vals = [sv("Stroke", "Balanced blocks"), sv("Sepsis", "Balanced blocks")]
    ax.barh([0, 1], vals, color=[BLUE, ORANGE], height=0.45)
    ax.set_yticks([0, 1], ["Stroke", "Sepsis"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("ARI vs primary solution")
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    clean_axis(ax, grid=True)
    ax.set_title(
        "D  WBC redundancy sensitivity in sepsis",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    red = data["redundancy"]
    mapping = {
        "drop_wbc_counts_keep_percentages": "Drop WBC counts",
        "drop_wbc_percentages_keep_counts": "Drop WBC percentages",
    }
    items = []
    for key, label in mapping.items():
        r = red[red.variant == key]
        if not r.empty:
            items.append((label, float(r.ari_vs_reference.iloc[0])))
    yy = np.arange(len(items))
    vv = [v for _, v in items]
    ax.barh(yy, vv, color=ORANGE, height=0.45)
    ax.set_yticks(yy, [k for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("ARI vs primary sepsis solution")
    for i, v in enumerate(vv):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")

    fig.text(
        0.5,
        0.035,
        "Takeaway: representation is nearly invariant, but weighting and feature specification expose "
        "much greater fragility in sepsis than in stroke.",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    return save(fig, outdir, "fig5_robustness", pdf, dpi)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Regenerate the five final Clustro paper figures from analysis outputs "
            "already present under results/."
        )
    )
    p.add_argument("--reporting-dir", default="results/methods_revision_reporting")
    p.add_argument("--outdir", default="results/final_paper_figures")
    p.add_argument("--phenotype-image", default=None)
    p.add_argument("--only", nargs="*", type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()
    style()

    reporting = Path(args.reporting_dir)
    outdir = Path(args.outdir)
    data = load_inputs(reporting)
    wanted = args.only or [1, 2, 3, 4, 5]
    for n in wanted:
        if n == 1:
            path = figure1(outdir, args.pdf, args.dpi)
        elif n == 2:
            path = figure2(data, outdir, args.pdf, args.dpi)
        elif n == 3:
            path = figure3(data, outdir, args.pdf, args.dpi)
        elif n == 4:
            path = figure4(
                data,
                reporting,
                outdir,
                args.pdf,
                args.dpi,
                Path(args.phenotype_image) if args.phenotype_image else None,
            )
        else:
            path = figure5(data, outdir, args.pdf, args.dpi)
        print(f"Figure {n}: {path}")


if __name__ == "__main__":
    main()
