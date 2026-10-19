"""
Controlled "ladder" from soft voting to the pool-based arms, on the identical folds (exploratory, post hoc).
Each rung changes one ingredient, so the rung where a dataset's gain appears identifies its source.

  V    soft voting, the 4 fixed members (DT depth 5, NB, kNN 5, LR), uniform weights   (the main-study baseline)
  Vw   same 4 members, weights exp(2 * 3-fold CV accuracy)                              -> effect of accuracy weighting alone
  Vp   same 4 members, the member with the lowest CV accuracy dropped, uniform          -> effect of removing the weak link
  Vpw  weakest member dropped AND accuracy weights
  Vs   the 4 members plus RF_10, ExtraTrees_10, GB_shallow from the pool, uniform       -> effect of adding strong learners
  T, W, A: top-4 by accuracy weighted, all 13 weighted, proposed method (from the stored runs)

    python3 ladder.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import experiment_final as E
import run_healthcare  # registers the medical loaders
import run_baselines as RB
from candidate_pool import get_candidate_pool

RES = os.path.join(HERE, "..", "results")
ALPHA = 2.0


def four():
    return {"DT": DecisionTreeClassifier(max_depth=5, random_state=42), "NB": GaussianNB(),
            "kNN": KNeighborsClassifier(n_neighbors=5), "LR": LogisticRegression(max_iter=1000, random_state=42)}


def combine(probs, names, weights=None):
    w = np.ones(len(names)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    return sum(wi * probs[n] for wi, n in zip(w, names))


rows = []
pool = get_candidate_pool()
for key, (name, loader, cap, repeats) in RB.DATASETS.items():
    path = os.path.join(RES, "healthcare" if key in run_healthcare.LOADERS else "baselines", f"{name}.json")
    st = json.load(open(path))
    stored = {a: float(np.mean([np.mean(np.array(f[a]) == np.array(f["y"])) for f in st["folds"]])) for a in "VTWA"}
    X, y = loader()
    if cap and len(y) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=E.RANDOM_STATE)
    acc = {k: [] for k in ("V", "Vw", "Vp", "Vpw", "Vs")}
    dropped = []
    for tr, te in RepeatedStratifiedKFold(n_splits=E.N_SPLITS, n_repeats=repeats, random_state=E.RANDOM_STATE).split(X, y):
        pre = E.make_preprocessor().fit(X[tr])
        Xtr, Xte, ytr, yte = pre.transform(X[tr]), pre.transform(X[te]), y[tr], y[te]
        mem = four()
        cv = {k: float(cross_val_score(clone(m), Xtr, ytr, cv=3, scoring="accuracy").mean()) for k, m in mem.items()}
        probs, classes = {}, None
        for k, m in mem.items():
            fm = clone(m).fit(Xtr, ytr); probs[k] = fm.predict_proba(Xte); classes = fm.classes_
        for k in ("RF_10", "ExtraTrees_10", "GB_shallow"):
            probs[k] = clone(pool[k]).fit(Xtr, ytr).predict_proba(Xte)
        names4 = list(mem)
        worst = min(cv, key=cv.get); dropped.append(worst)
        kept = [n for n in names4 if n != worst]
        wfun = lambda ns: [np.exp(ALPHA * cv[n]) for n in ns]
        lab = lambda p: classes[np.argmax(p, axis=1)]
        acc["V"].append(np.mean(lab(combine(probs, names4)) == yte))
        acc["Vw"].append(np.mean(lab(combine(probs, names4, wfun(names4))) == yte))
        acc["Vp"].append(np.mean(lab(combine(probs, kept)) == yte))
        acc["Vpw"].append(np.mean(lab(combine(probs, kept, wfun(kept))) == yte))
        acc["Vs"].append(np.mean(lab(combine(probs, names4 + ["RF_10", "ExtraTrees_10", "GB_shallow"])) == yte))
    r = {"dataset": name, **{k: float(np.mean(v)) for k, v in acc.items()}, **{f"stored_{a}": stored[a] for a in "VTWA"},
         "most_dropped": max(set(dropped), key=dropped.count)}
    assert abs(r["V"] - stored["V"]) < 1e-9, f"{name}: recomputed V differs from stored V"
    rows.append(r)

rungs = ["Vw", "Vp", "Vpw", "Vs", "stored_T", "stored_W", "stored_A"]
lab = {"Vw": "same 4, accuracy-weighted", "Vp": "same 4, weakest dropped", "Vpw": "weakest dropped + weighted",
       "Vs": "4 + RF/ET/GB, uniform", "stored_T": "top-4 by accuracy, weighted (T)", "stored_W": "all 13 weighted (W)", "stored_A": "proposed (A)"}
print("Recomputed soft voting (V) matches the stored result on all", len(rows), "datasets.\n")
print(f"{'dataset':26s} {'V':>6s} " + " ".join(f"{r_:>8s}" for r_ in rungs) + "   dropped")
for r in rows:
    print(f"{r['dataset']:26s} {100*r['V']:6.2f} " + " ".join(f"{100*r[k]:8.2f}" for k in rungs) + f"   {r['most_dropped']}")
print("\nMean accuracy gain over V (percentage points), 16 datasets:")
summary = {}
for k in rungs:
    g = [100 * (r[k] - r["V"]) for r in rows]
    summary[k] = {"mean": float(np.mean(g)), "median": float(np.median(g)), "ahead": int(sum(x > 1e-9 for x in g)), "behind": int(sum(x < -1e-9 for x in g))}
    print(f"  {lab[k]:34s} mean {np.mean(g):+.2f}  median {np.median(g):+.2f}  ahead {summary[k]['ahead']:2d}  behind {summary[k]['behind']:2d}")
os.makedirs(os.path.join(RES, "explain"), exist_ok=True)
json.dump({"rows": rows, "summary": summary}, open(os.path.join(RES, "explain", "ladder.json"), "w"), indent=2)
