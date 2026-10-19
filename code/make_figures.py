"""Regenerate paper figures from stored results (300 dpi, sized for one IEEE column ~3.4 in).
    python3 make_figures.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7})


def flowchart():
    steps = [
        ("Input: 12 UCI datasets", "Pima: zero-coded missing values set to NaN", "#e3f2fd", "#1565c0"),
        ("In-fold preprocessing", "median imputation + standardisation,\nfitted on the training fold only", "#e3f2fd", "#1565c0"),
        ("Candidate pool: 13 classifiers", "3-fold CV accuracy and out-of-fold\npredictions on the training fold", "#e3f2fd", "#1565c0"),
        ("Competitive-margin selection (studied)", "eligible: CV accuracy within 0.05 of best;\ngreedy max. pairwise disagreement, k = 4", "#fff0dc", "#e65100"),
        ("Reliability weights", "w_k proportional to exp(2 x CV accuracy_k)", "#fff0dc", "#e65100"),
        ("Belief revision rounds (beta on inner split)", "ablation: changes accuracy by about 0", "#fff0dc", "#e65100"),
        ("Prediction", "argmax of weighted mean probability;\ntest fold used once, for scoring only", "#e3f2fd", "#1565c0"),
        ("Evaluation", "repeated stratified 5-fold CV, McNemar,\nHolm over 12 datasets; frozen test on 7 more", "#e8f5e9", "#2e7d32"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 100); ax.axis("off")
    h, gap = 9.2, 2.6
    top = 99
    for i, (t, sub, fc, ec) in enumerate(steps):
        y = top - i * (h + gap) - h
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(5, y + h - 2.4, t, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(5, y + 3.2, sub, ha="center", va="center", fontsize=6.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(5, y - gap + 0.2), xytext=(5, y), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333"))
    fig.savefig(os.path.join(FIG, "methodology_flowchart.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def results_chart():
    d = pd.read_csv(os.path.join(RES, "summary_table.csv")).sort_values("delta_pp")
    names = {"pima_diabetes": "Pima", "heart_disease": "Heart", "australian_credit": "Australian",
             "haberman_survival": "Haberman", "liver_disorders": "Liver", "magic_gamma": "Magic",
             "spambase": "Spambase", "banknote_auth": "Banknote", "breast_cancer_wisconsin": "Breast cancer",
             "german_credit": "German", "ionosphere": "Ionosphere", "sonar": "Sonar"}
    col = ["#2e7d32" if v == "win" else "#c62828" if v == "loss" else "#9e9e9e" for v in d.verdict_after_holm]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.barh([names[k] for k in d.dataset], d.delta_pp, color=col, height=0.7)
    ax.axvline(0, color="black", lw=0.6)
    for y, v in enumerate(d.delta_pp):
        ax.text(v + (0.08 if v >= 0 else -0.08), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=6)
    ax.set_xlim(-1.8, 4.3)
    ax.set_xlabel("Accuracy difference vs. soft voting (percentage points)", fontsize=6.5)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.98, 0.04, "green: significant win after Holm\ngrey: not significant", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "results_summary.png"), dpi=300)
    plt.close(fig)


def baseline_ranks():
    import json
    sm = json.load(open(os.path.join(RES, "baselines", "baseline_summary.json")))
    lab = {"V": "Soft voting (4 fixed)", "B": "Best single", "W": "All 13, weighted", "T": "Top-4 by accuracy, weighted",
           "S": "Margin selection, no revision", "A": "Proposed (full)"}
    items = sorted(sm["mean_rank"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    cols = ["#e65100" if k in ("A", "S") else "#1565c0" if k == "T" else "#9e9e9e" for k, _ in items]
    ax.barh([lab[k] for k, _ in items][::-1], [v for _, v in items][::-1], color=cols[::-1], height=0.65)
    for y, (k, v) in enumerate(items[::-1]):
        ax.text(v + 0.05, y, f"{v:.2f}  ({100*sm['mean_acc'][k]:.2f}%)", va="center", fontsize=6)
    ax.set_xlim(0, 6.2)
    ax.set_xlabel("Mean rank (1 = best); mean accuracy in brackets", fontsize=6)
    ax.tick_params(labelsize=6.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "baseline_ranks.png"), dpi=300)
    plt.close(fig)


def roc_pr():
    import json
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3))
    for col, (name, title) in enumerate([("heart_disease", "Heart"), ("pima_diabetes", "Pima")]):
        st = json.load(open(os.path.join(RES, "predictions", f"{name}.json")))
        y = np.concatenate([f["y"] for f in st["folds"]])
        for arm, lab, color in (("pV", "Soft voting", "#9e9e9e"), ("pA", "Proposed", "#e65100")):
            pr = np.concatenate([f[arm] for f in st["folds"]])
            fpr, tpr, _ = roc_curve(y, pr)
            axes[0, col].plot(fpr, tpr, color=color, lw=1, label=f"{lab} (AUC {roc_auc_score(y, pr):.3f})")
            pc, rc, _ = precision_recall_curve(y, pr)
            axes[1, col].plot(rc, pc, color=color, lw=1, label=f"{lab} (AP {average_precision_score(y, pr):.3f})")
        axes[0, col].plot([0, 1], [0, 1], color="#bbbbbb", lw=0.6, ls="--")
        axes[0, col].set_title(f"{title}: ROC", fontsize=6.5); axes[0, col].set_xlabel("False positive rate", fontsize=6)
        axes[1, col].set_title(f"{title}: precision-recall", fontsize=6.5); axes[1, col].set_xlabel("Recall", fontsize=6)
        if col == 0:
            axes[0, col].set_ylabel("True positive rate", fontsize=6); axes[1, col].set_ylabel("Precision", fontsize=6)
        for r in (0, 1):
            axes[r, col].legend(fontsize=5, loc="lower right" if r == 0 else "lower left", frameon=False)
            axes[r, col].tick_params(labelsize=5.5)
            for sp in ("top", "right"):
                axes[r, col].spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "roc_pr_curves.png"), dpi=300)
    plt.close(fig)


flowchart(); results_chart(); baseline_ranks(); roc_pr(); print("figures written")
