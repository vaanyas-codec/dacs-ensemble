"""
Summarise results/baselines/*.json (from run_baselines.py).

Also checks that arm A reproduces the stored main-study accuracy and that the voting
baseline V does too, i.e. that the folds are identical to the main study.

    python3 analyze_baselines.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment_final import mcnemar  # noqa: E402

RES = os.path.join(HERE, "..", "results")
ARMS = ["V", "B", "W", "T", "S", "A"]
LABEL = {"V": "Soft voting (4 fixed)", "B": "Best single in pool", "W": "All 13, accuracy-weighted",
         "T": "Top-4 by accuracy, weighted", "S": "Margin selection, no revision", "A": "Proposed (full)"}
SHORT = {"pima_diabetes": "Pima", "heart_disease": "Heart", "australian_credit": "Australian", "haberman_survival": "Haberman",
         "liver_disorders": "Liver", "magic_gamma": "Magic", "spambase": "Spambase", "banknote_auth": "Banknote",
         "breast_cancer_wisconsin": "Breast cancer", "german_credit": "German", "ionosphere": "Ionosphere", "sonar": "Sonar"}
ORDER = list(SHORT)

rows, pooled = {}, {}
for name in ORDER:
    p = os.path.join(RES, "baselines", f"{name}.json")
    st = json.load(open(p))
    assert len(st["folds"]) == st["n_folds_total"], f"{name} incomplete"
    y = np.concatenate([f["y"] for f in st["folds"]])
    pooled[name] = {a: np.concatenate([f[a] for f in st["folds"]]) for a in ARMS}
    pooled[name]["y"] = y
    rows[name] = {a: float(np.mean([np.mean(np.array(f[a]) == np.array(f["y"])) for f in st["folds"]])) for a in ARMS}
    ref = json.load(open(os.path.join(RES, f"{name}.json")))
    assert abs(rows[name]["A"] - ref["dacs_mean"]) < 1e-9 and abs(rows[name]["V"] - ref["voting_mean"]) < 1e-9, \
        f"{name}: arm A/V do not reproduce the stored main-study result"
print("Arm A and V reproduce the stored main-study accuracy on all 12 datasets (identical folds).\n")

M = np.array([[rows[n][a] for a in ARMS] for n in ORDER])            # datasets x arms
ranks = np.array([rankdata(-m) for m in M])                           # 1 = best
print(f"{'dataset':14s} " + " ".join(f"{a:>7s}" for a in ARMS))
for n, m in zip(ORDER, M):
    print(f"{SHORT[n]:14s} " + " ".join(f"{100*v:7.2f}" for v in m))
print(f"{'MEAN':14s} " + " ".join(f"{100*v:7.2f}" for v in M.mean(0)))
print(f"{'MEAN RANK':14s} " + " ".join(f"{v:7.2f}" for v in ranks.mean(0)))

print("\nPaired comparisons over 12 datasets (difference in fold-mean accuracy, percentage points):")
pairs = [("A", "V"), ("A", "B"), ("A", "W"), ("A", "T"), ("A", "S"), ("S", "T"), ("T", "V"), ("W", "V"), ("B", "V")]
summary = []
for x, z in pairs:
    d = 100 * (M[:, ARMS.index(x)] - M[:, ARMS.index(z)])
    w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    sig_up = sig_dn = 0
    for n, dd in zip(ORDER, d):
        pm = mcnemar(pooled[n]["y"], pooled[n][x], pooled[n][z])
        if pm < 0.05:
            sig_up += dd > 0
            sig_dn += dd < 0
    summary.append({"pair": f"{x} vs {z}", "mean_diff_pp": round(float(d.mean()), 3), "median_diff_pp": round(float(np.median(d)), 3),
                    "wins": wins, "losses": losses, "wilcoxon_p": round(float(w), 4),
                    "mcnemar_raw_p_lt_0.05_better": int(sig_up), "mcnemar_raw_p_lt_0.05_worse": int(sig_dn)})
    print(f"  {x} - {z}: mean {d.mean():+.2f}, median {np.median(d):+.2f}, ahead on {wins}, behind on {losses}, "
          f"Wilcoxon p = {w:.3f}; per-dataset McNemar (uncorrected) p<0.05: better {int(sig_up)}, worse {int(sig_dn)}")

with open(os.path.join(RES, "baselines", "baseline_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset"] + ARMS)
    for n, m in zip(ORDER, M):
        w.writerow([n] + [round(v, 4) for v in m])
    w.writerow(["mean"] + [round(v, 4) for v in M.mean(0)])
    w.writerow(["mean_rank"] + [round(v, 2) for v in ranks.mean(0)])
json.dump({"mean_acc": dict(zip(ARMS, map(float, M.mean(0)))), "mean_rank": dict(zip(ARMS, map(float, ranks.mean(0)))),
           "pairs": summary}, open(os.path.join(RES, "baselines", "baseline_summary.json"), "w"), indent=2)
