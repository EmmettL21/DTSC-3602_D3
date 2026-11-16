"""Figures for the D3 modeling note, drawn from results/evaluation.json.

    .venv/Scripts/python eval/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
# Color follows the model everywhere. Baselines are neutral; candidates take slots 1-2.
COLOR = {"random": "#c3c2b7", "tfidf_cosine": INK2,
         "jobswipe_gte_cosine": "#2a78d6", "jobtinder_lightgbm": "#eb6834",
         "jobswipe_resume_only": "#7fb0ea", "bge_cosine": "#f2a27f"}
LABEL = {"random": "Random (floor)", "tfidf_cosine": "Baseline: TF-IDF cosine",
         "jobswipe_gte_cosine": "JobSwipe (gte-small cosine)",
         "jobtinder_lightgbm": "Job-Tinder (LightGBM)",
         "jobswipe_resume_only": "  JobSwipe, resume only", "bge_cosine": "  bge-small cosine only"}
ORDER = ["random", "tfidf_cosine", "jobswipe_gte_cosine", "jobtinder_lightgbm",
         "jobswipe_resume_only", "bge_cosine"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8,
})


def fig_comparison(rep: dict) -> None:
    """AUC (with 95% CI), NDCG@10 seeker and macro F1, both splits, every model."""
    metrics = [("auc_good_vs_no", "AUC, Good vs No Fit"), ("ndcg_seeker_at_10", "NDCG@10, seeker deck"),
               ("macro_f1", "Macro F1, 3 classes")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    ys = np.arange(len(ORDER))[::-1]
    for ax, (key, title) in zip(axes, metrics):
        for y, name in zip(ys, ORDER):
            c = COLOR[name]
            for split, marker, fill, dy in (("cold_jd", "o", "none", 0.15), ("cold_resume", "o", c, -0.15)):
                r = rep[split][name]
                v = r[key]
                if key == "auc_good_vs_no":
                    lo, hi = r["auc_ci"]
                    ax.plot([lo, hi], [y + dy] * 2, color=c, lw=2, solid_capstyle="round", alpha=0.55)
                ax.plot(v, y + dy, marker=marker, ms=7, mfc=fill, mec=c, mew=2, ls="none")
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        if key == "auc_good_vs_no":
            ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    axes[0].set_yticks(ys, [LABEL[n] for n in ORDER])
    for t, n in zip(axes[0].get_yticklabels(), ORDER):
        t.set_color(INK if n in ("tfidf_cosine", "jobswipe_gte_cosine", "jobtinder_lightgbm") else INK2)
    h = [plt.Line2D([], [], marker="o", ms=7, mfc="none", mec=INK2, mew=2, ls="none"),
         plt.Line2D([], [], marker="o", ms=7, mfc=INK2, mec=INK2, mew=2, ls="none")]
    fig.legend(h, ["Cold-JD split (unseen jobs, n=1,759)", "Cold-resume split (unseen resumes, 5-fold, n=8,000)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.02))
    fig.text(0.01, 0.99, "Same data, same splits, same metrics. Lines on AUC are 95% cluster-bootstrap CIs.",
             color=INK2, va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(FIG / "fig1_model_comparison.png", dpi=200)
    plt.close(fig)


def fig_errors(rep: dict) -> None:
    """AUC by group for the conditions that separate the models."""
    shown = ["tfidf_cosine", "jobswipe_gte_cosine", "jobtinder_lightgbm"]
    conds = [("jd_length", "Job description length"), ("resume_title_detected", "JobSwipe finds a job title"),
             ("skill_coverage", "Resume covers job's skills"), ("job_seniority", "Job seniority")]
    ea = rep["error_analysis_cold_resume"]
    fig, axes = plt.subplots(1, len(conds), figsize=(11, 3.2))
    for ax, (cond, title) in zip(axes, conds):
        rows = [r for r in ea[cond] if r["n"] >= 150]
        ys = np.arange(len(rows))[::-1]
        for off, name in zip((0.18, 0.0, -0.18), shown):  # dodge so equal values stay visible
            vals = [r[name]["auc"] for r in rows]
            ax.plot(vals, ys + off, marker="o", ms=7, ls="none", color=COLOR[name], mec=SURFACE, mew=1.2,
                    label=LABEL[name])
        ax.set_yticks(ys, [f"{r['group']}\n(n={r['n']:,})" for r in rows], fontsize=7.5)
        ax.tick_params(axis="y", length=0)
        ax.axvline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(0.4, 0.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.text(0.01, 0.99, "AUC (Good vs No Fit) within each group, cold-resume out-of-fold predictions. "
             "Groups with n < 150 omitted.", color=INK2, va="top", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG / "fig2_error_analysis.png", dpi=200)
    plt.close(fig)


def fig_badge() -> None:
    """JobSwipe's displayed '% match' for pairs of each true label."""
    d = np.load(RES / "oof_scores.npz")
    y, cos = d["y"], d["jobswipe_gte_cosine_score"]
    pct = np.clip((cos - 0.80) / 0.16, 0, 1) * 100
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    bins = np.linspace(0, 100, 21)
    shades = {0: "#b8d3f5", 1: "#6aa0e6", 2: "#1c4f8f"}  # one hue, light -> dark (ordinal)
    for lab, name in ((0, "No Fit"), (1, "Potential Fit"), (2, "Good Fit")):
        h, _ = np.histogram(pct[y == lab], bins=bins)
        ax.plot((bins[:-1] + bins[1:]) / 2, h / h.sum() * 100, color=shades[lab], lw=2,
                label=f"{name} (median {np.median(pct[y == lab]):.0f}%)")
    ax.set_xlabel("'% match' badge JobSwipe would display")
    ax.set_ylabel("% of pairs")
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper left")
    ax.set_title("Displayed match % overlaps heavily across true labels", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_jobswipe_badge.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rep = json.loads((RES / "evaluation.json").read_text(encoding="utf-8"))
    fig_comparison(rep)
    fig_errors(rep)
    fig_badge()
    print(f"wrote {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
