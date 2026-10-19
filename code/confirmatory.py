"""
Confirmatory evaluation, implementing PREREGISTRATION.md exactly.

Runs two arms on datasets that played no part in the method's design:
  Arm A: competitive margin 0.05 (proposed)
  Arm B: competitive margin 1.0 (margin disabled = original selection rule)
plus the soft-voting baseline, on identical folds.

Resumable: progress is written after every fold, and the script stops
cleanly when its time budget is spent, so it can be re-run until complete.

    python3 confirmatory.py            # run/resume (default 240 s budget)
    python3 confirmatory.py 120        # custom budget in seconds
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
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from candidate_pool import get_candidate_pool
from experiment_final import fit_predict_fold

SEED = 42
CAP = 3000
OUT_DIR = os.path.join(HERE, "..", "results", "confirmatory")
DATA_DIR = os.environ.get("CONFIRMATORY_DATA_DIR", os.path.join(HERE, "..", "data", "confirmatory"))

# Fixed order and file names, as listed in PREREGISTRATION.md
DATASETS = ["iris", "wine", "new-thyroid", "wheat-seeds", "oil-spill", "phoneme", "mammography"]


def load(key):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{key}.csv"), header=None)
    assert not df.isna().any().any(), f"{key}: missing values"
    y = LabelEncoder().fit_transform(df.iloc[:, -1].astype(str))
    X = df.iloc[:, :-1].astype(float).values
    subsampled = False
    if len(y) > CAP:
        X, _, y, _ = train_test_split(X, y, train_size=CAP, stratify=y, random_state=SEED)
        subsampled = True
    return X, y, subsampled


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


def run(budget):
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    remaining = []
    for key in DATASETS:
        X, y, subsampled = load(key)
        repeats = 1 if subsampled else 2
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED).split(X, y))
        path = os.path.join(OUT_DIR, f"{key}.json")
        state = json.load(open(path)) if os.path.exists(path) else {
            "key": key, "n": int(len(y)), "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))), "subsampled": subsampled,
            "n_folds_total": len(splits), "folds": [],
        }
        done = {f["i"] for f in state["folds"]}
        for i, (tr, te) in enumerate(splits):
            if i in done:
                continue
            if time.time() - t0 > budget:
                break
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            ytr, yte = y[tr], y[te]
            pa, ba, ea = fit_predict_fold(get_candidate_pool(), Xtr, ytr, Xte, competitive_margin=0.05)
            pb, bb, eb = fit_predict_fold(get_candidate_pool(), Xtr, ytr, Xte, competitive_margin=1.0)
            pv = baseline().fit(Xtr, ytr).predict(Xte)
            state["folds"].append({
                "i": i, "y": yte.tolist(), "A": pa.tolist(), "B": pb.tolist(), "V": pv.tolist(),
                "betaA": ba, "betaB": bb, "selA": ea.selected_names_, "selB": eb.selected_names_,
                "divA": float(ea.diversity_), "divB": float(eb.diversity_),
            })
            json.dump(state, open(path, "w"))
            print(f"{key} fold {i+1}/{len(splits)}: A={np.mean(pa==yte):.3f} "
                  f"B={np.mean(pb==yte):.3f} V={np.mean(pv==yte):.3f}", flush=True)
        n_done = len(state["folds"])
        if n_done < len(splits):
            remaining.append(f"{key} ({n_done}/{len(splits)})")
    elapsed = time.time() - t0
    if remaining:
        print(f"\nINCOMPLETE after {elapsed:.0f}s. Remaining: {', '.join(remaining)}")
    else:
        print(f"\nALL COMPLETE ({elapsed:.0f}s this call)")


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 240.0)
