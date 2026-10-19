"""
EXPLORATORY (not preregistered; added after the confirmatory results were seen).

Arm C = identical selection (margin 0.05) and identical reliability weights as
Arm A, but ZERO revision rounds, i.e. a reliability-weighted vote over the
selected classifiers. Comparing A with C isolates what the iterative revision
step adds beyond selection plus weighting.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import confirmatory as C
from candidate_pool import get_candidate_pool
from debate_ensemble import DACSDebateEnsemble
from experiment_final import mcnemar, holm_bonferroni

out, rows = os.path.join(HERE, "..", "results", "confirmatory"), {}
for key in C.DATASETS:
    X, y, sub = C.load(key)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if sub else 2, random_state=42).split(X, y))
    saved = json.load(open(os.path.join(out, f"{key}.json")))["folds"]
    Ap, Cp, ys, accC, accA = [], [], [], [], []
    for (tr, te), s in zip(splits, saved):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ens = DACSDebateEnsemble(get_candidate_pool(), max_rounds=0, competitive_margin=0.05).fit(Xtr, y[tr])
        pc = ens.predict(Xte)
        assert ens.selected_names_ == s["selA"], "selection should match Arm A"
        Cp += pc.tolist(); Ap += s["A"]; ys += s["y"]
        accC.append(np.mean(pc == y[te])); accA.append(np.mean(np.array(s["A"]) == y[te]))
    ys, Ap, Cp = map(np.array, (ys, Ap, Cp))
    rows[key] = {"accA": float(np.mean(accA)), "accC": float(np.mean(accC)),
                 "dAC": 100 * float(np.mean(accA) - np.mean(accC)), "p": mcnemar(ys, Ap, Cp)}
h = holm_bonferroni({k: r["p"] for k, r in rows.items()})
for k in rows: rows[k]["holm"] = h[k]["holm_p"]
json.dump({"label": "EXPLORATORY, not preregistered", "datasets": rows},
          open(os.path.join(out, "exploratory_no_debate.json"), "w"), indent=2)
print(f"{'dataset':12s} {'noDebate(C)':>11s} {'debate(A)':>10s} {'A-C pp':>8s} {'p':>8s} {'holm':>8s}")
for k, r in rows.items():
    print(f"{k:12s} {r['accC']:11.4f} {r['accA']:10.4f} {r['dAC']:+8.2f} {r['p']:8.4f} {r['holm']:8.4f}")
d = np.array([r["dAC"] for r in rows.values()])
print(f"\nmean A-C = {d.mean():+.2f} pp | A better on {(d>0).sum()}, worse on {(d<0).sum()}, equal on {(d==0).sum()}")
