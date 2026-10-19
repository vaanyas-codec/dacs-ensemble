"""
Re-run selected main-study datasets with per-fold predictions AND class
probabilities saved, so that metrics beyond accuracy (precision, recall, F1,
ROC AUC, confusion matrices) can be computed afterwards.

Uses exactly the same folds, seeds, preprocessing and ensemble code path as
experiment_final.run_dataset, so accuracy must match that runner. Three arms on
identical folds:

  A  DACS + Debate, beta chosen on an inner split of the training fold
  F  DACS + Debate, beta fixed at 0.3 (no beta search; a control)
  V  soft-voting baseline (DT, NB, kNN, LR)

Resumable: progress is written after every fold, and the script stops cleanly
when its time budget is spent.

    python3 run_with_predictions.py pima 240    # run/resume Pima for 240 s
    python3 run_with_predictions.py heart 240
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import experiment_final as E
from candidate_pool import get_candidate_pool
from debate_ensemble import DACSDebateEnsemble

OUT_DIR = os.path.join(HERE, "..", "results", "predictions")
LOADERS = {"pima": ("pima_diabetes", E.load_pima), "heart": ("heart_disease", E.load_heart)}
SEED = E.RANDOM_STATE


def baseline():
    return VotingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
            ("nb", GaussianNB()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
        ],
        voting="soft",
    )


def run(key, budget):
    name, loader = LOADERS[key]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    X, y = loader()
    splits = list(RepeatedStratifiedKFold(
        n_splits=E.N_SPLITS, n_repeats=E.N_REPEATS, random_state=SEED).split(X, y))
    state = json.load(open(path)) if os.path.exists(path) else {
        "dataset": name, "n": int(len(y)), "n_features": int(X.shape[1]),
        "n_missing_cells_before_imputation": int(np.isnan(X).sum()),
        "n_folds_total": len(splits), "folds": []}
    done = {f["i"] for f in state["folds"]}
    t0 = time.time()
    for i, (tr, te) in enumerate(splits):
        if i in done:
            continue
        if time.time() - t0 > budget:
            break
        pre = E.make_preprocessor().fit(X[tr])          # fitted on the training fold only
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        ytr, yte = y[tr], y[te]

        pa, beta, ens = E.fit_predict_fold(get_candidate_pool(), Xtr, ytr, Xte)
        proba_a = ens.predict_proba(Xte)[:, 1]
        fixed = DACSDebateEnsemble(get_candidate_pool(), beta=0.3).fit(Xtr, ytr)
        pf, proba_f = fixed.predict(Xte), fixed.predict_proba(Xte)[:, 1]
        v = baseline().fit(Xtr, ytr)
        pv, proba_v = v.predict(Xte), v.predict_proba(Xte)[:, 1]

        state["folds"].append({
            "i": i, "y": yte.tolist(),
            "A": pa.tolist(), "F": pf.tolist(), "V": pv.tolist(),
            "pA": [round(float(p), 6) for p in proba_a],
            "pF": [round(float(p), 6) for p in proba_f],
            "pV": [round(float(p), 6) for p in proba_v],
            "betaA": beta, "selA": ens.selected_names_,
            "diversityA": float(ens.diversity_), "spreadA": float(ens.spread_),
        })
        json.dump(state, open(path, "w"))
        print(f"{name} fold {i+1}/{len(splits)}: A={np.mean(pa==yte):.4f} "
              f"F={np.mean(pf==yte):.4f} V={np.mean(pv==yte):.4f} beta={beta} "
              f"sel={ens.selected_names_}", flush=True)
    n_done = len(state["folds"])
    print(("ALL COMPLETE" if n_done == len(splits) else f"INCOMPLETE {n_done}/{len(splits)}"),
          f"({time.time()-t0:.0f}s this call)")


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 240.0)
