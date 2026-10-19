"""
Analysis of the pre-specified medical evaluation, exactly as fixed in PREREGISTRATION_HEALTHCARE.md:
per-dataset fold-mean accuracy for every arm, pooled McNemar tests A vs V, A vs T and S vs T, Holm correction over
the 4 datasets (separately per comparison), and the pre-registered decision rule.

    python3 analyze_healthcare.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiment_final import holm_bonferroni, mcnemar  # noqa: E402

RES = os.path.join(HERE, "..", "results", "healthcare")
SETS = ["wdbc", "hepatitis", "breast_cancer_ljubljana", "prostate_recurrence"]
ARMS = ["V", "B", "W", "T", "S", "A"]

acc, pooled, n = {}, {}, {}
for s in SETS:
    st = json.load(open(os.path.join(RES, f"{s}.json")))
    assert len(st["folds"]) == st["n_folds_total"], f"{s} incomplete"
    n[s] = st["n"]
    pooled[s] = {a: np.concatenate([f[a] for f in st["folds"]]) for a in ARMS}
    pooled[s]["y"] = np.concatenate([f["y"] for f in st["folds"]])
    acc[s] = {a: float(np.mean([np.mean(np.array(f[a]) == np.array(f["y"])) for f in st["folds"]])) for a in ARMS}

from healthcare_data import LOADERS
_KEY = {"wdbc": "wdbc", "hepatitis": "hepatitis", "breast_cancer_ljubljana": "ljubljana", "prostate_recurrence": "prostate"}
facts = {}
for s_ in SETS:
    X_, y_ = LOADERS[_KEY[s_]]()
    facts[s_] = {"features": int(X_.shape[1]), "positive_pct": round(100 * float(np.mean(y_)), 1),
                 "majority_class_pct": round(100 * max(float(np.mean(y_)), 1 - float(np.mean(y_))), 1),
                 "missing_cells": int(np.isnan(X_).sum())}
out = {"datasets": {}, "comparisons": {}}
print(f"{'dataset':26s} {'n':>4s} " + " ".join(f"{a:>7s}" for a in ARMS))
for s in SETS:
    print(f"{s:26s} {n[s]:4d} " + " ".join(f"{100*acc[s][a]:7.2f}" for a in ARMS))
    out["datasets"][s] = {"n": n[s], **facts[s], **{a: acc[s][a] for a in ARMS}}

for x, z in (("A", "V"), ("A", "T"), ("S", "T")):
    raw = {s: mcnemar(pooled[s]["y"], pooled[s][x], pooled[s][z]) for s in SETS}
    hp = holm_bonferroni(raw)
    diffs = {s: 100 * (acc[s][x] - acc[s][z]) for s in SETS}
    rows = {s: {"diff_pp": diffs[s], "p_raw": raw[s], "holm_p": hp[s]["holm_p"],
                "sig_up": bool(hp[s]["sig"] and diffs[s] > 0), "sig_down": bool(hp[s]["sig"] and diffs[s] < 0)} for s in SETS}
    out["comparisons"][f"{x}-{z}"] = {"mean_diff_pp": float(np.mean(list(diffs.values()))), "per_dataset": rows,
                                      "n_sig_better": sum(r["sig_up"] for r in rows.values()),
                                      "n_sig_worse": sum(r["sig_down"] for r in rows.values())}
    print(f"\n{x} - {z}: mean {np.mean(list(diffs.values())):+.2f} pp")
    for s in SETS:
        r = rows[s]
        print(f"  {s:26s} {r['diff_pp']:+6.2f} pp  p={r['p_raw']:.4f}  holm={r['holm_p']:.4f}  {'BETTER' if r['sig_up'] else 'WORSE' if r['sig_down'] else ''}")

av, st_ = out["comparisons"]["A-V"], out["comparisons"]["S-T"]
rule1 = av["n_sig_better"] >= 2 and av["n_sig_worse"] == 0
rule2 = st_["mean_diff_pp"] > 0 and st_["n_sig_better"] >= 1
out["decision"] = {"method_better_than_voting_on_medical_data": "SUPPORTED" if rule1 else "NOT SUPPORTED",
                   "diversity_step_helps": "SUPPORTED" if rule2 else "NOT SUPPORTED"}
print("\nPre-registered decision rule:")
print("  Method significantly better than soft voting on >=2 of 4 (none worse):", out["decision"]["method_better_than_voting_on_medical_data"])
print("  Diversity step helps (S>T on average and significant on >=1):         ", out["decision"]["diversity_step_helps"])
json.dump(out, open(os.path.join(RES, "healthcare_summary.json"), "w"), indent=2)
