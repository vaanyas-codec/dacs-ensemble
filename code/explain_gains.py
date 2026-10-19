"""
Exploratory (post hoc) analysis: is the gain of the pool-based arms over soft voting larger where the soft-voting
baseline is weak, i.e. where one of its four members is much worse than another?

For each of the 16 datasets (12 main + 4 pre-specified medical) it recomputes, on the identical folds, the accuracy of
each soft-voting member alone (DT depth 5, naive Bayes, kNN 5, LR), then relates:
    voting_deficit = best member - soft voting      (how far voting falls below its own best member)
    member_range   = best member - worst member
    weakest        = worst member accuracy
to the gains A-V, T-V, W-V (from the stored runs). Spearman rank correlations over 16 datasets, descriptive only.

    python3 explain_gains.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import experiment_final as E
import run_healthcare  # registers the medical loaders in run_baselines.DATASETS
import run_baselines as RB

RES = os.path.join(HERE, "..", "results")
rows = []
for key, (name, loader, cap, repeats) in RB.DATASETS.items():
    path = os.path.join(RES, "healthcare" if key in run_healthcare.LOADERS else "baselines", f"{name}.json")
    st = json.load(open(path))
    arm = {a: float(np.mean([np.mean(np.array(f[a]) == np.array(f["y"])) for f in st["folds"]])) for a in "VBWTSA"}
    X, y = loader()
    if cap and len(y) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=E.RANDOM_STATE)
    splits = list(RepeatedStratifiedKFold(n_splits=E.N_SPLITS, n_repeats=repeats, random_state=E.RANDOM_STATE).split(X, y))
    mem = {"DT": [], "NB": [], "kNN": [], "LR": []}
    for tr, te in splits:
        pre = E.make_preprocessor().fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        for k, m in (("DT", DecisionTreeClassifier(max_depth=5, random_state=42)), ("NB", GaussianNB()),
                     ("kNN", KNeighborsClassifier(n_neighbors=5)), ("LR", LogisticRegression(max_iter=1000, random_state=42))):
            mem[k].append(np.mean(m.fit(Xtr, y[tr]).predict(Xte) == y[te]))
    mem = {k: float(np.mean(v)) for k, v in mem.items()}
    best, worst = max(mem.values()), min(mem.values())
    rows.append({"dataset": name, **{f"m_{k}": v for k, v in mem.items()}, **arm,
                 "deficit": best - arm["V"], "range": best - worst, "weakest": worst, "worst_member": min(mem, key=mem.get),
                 "majority": max(float(np.mean(y)), 1 - float(np.mean(y)))})

print(f"{'dataset':26s} {'DT':>6s} {'NB':>6s} {'kNN':>6s} {'LR':>6s} | {'V':>6s} {'T':>6s} {'A':>6s} | deficit  range  weakest")
for r in rows:
    print(f"{r['dataset']:26s} " + " ".join(f"{100*r['m_'+k]:6.2f}" for k in ('DT', 'NB', 'kNN', 'LR')) + " | " +
          " ".join(f"{100*r[a]:6.2f}" for a in 'VTA') + f" | {100*r['deficit']:6.2f} {100*r['range']:6.2f}  {r['worst_member']}")

print("\nSpearman correlation across the 16 datasets (rho, p):")
res = {}
for g, lab in (("A", "A-V"), ("T", "T-V"), ("W", "W-V")):
    gain = [100 * (r[g] - r["V"]) for r in rows]
    for pred in ("deficit", "range"):
        rho, p = spearmanr([100 * r[pred] for r in rows], gain)
        res[f"{lab} vs {pred}"] = {"rho": float(rho), "p": float(p)}
        print(f"  gain {lab} vs {pred:8s}: rho = {rho:+.2f}, p = {p:.3f}")
os.makedirs(os.path.join(RES, "explain"), exist_ok=True)
json.dump({"rows": rows, "spearman": res}, open(os.path.join(RES, "explain", "gain_explanation.json"), "w"), indent=2)
