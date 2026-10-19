"""
Exhaustive 3- and 4-attribute feature subset search on Heart Disease,
matching the search space of Latha & Jeeva (2019): all C(13,3) + C(13,4)
combinations = 1001 subsets.

Each subset is evaluated once on an inner train/validation split (carved
out of the training fold only -- no test-set leakage) using DACS+Debate
with beta fixed at 0.1 (the value every Heart Disease fold already chose
in the main experiment, so no per-subset hyperparameter search is needed).
This keeps runtime feasible; the best subset found is then re-evaluated
properly with full repeated cross-validation in a separate step.

Resumable: progress is saved to feature_search_progress.json after every
batch, so this can be re-run (e.g. across several time-limited calls)
without losing work.
"""
import json
import os
import sys
import time
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from debate_ensemble import DACSDebateEnsemble
from candidate_pool import get_candidate_pool
from experiment_final import load_heart, RANDOM_STATE

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_search_progress.json")
FIXED_BETA = 0.1
BATCH_SIZE = 90  # tuned to fit comfortably within a 5-minute call
FEATURE_NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def all_combos(n_features):
    combos = []
    for size in (3, 4):
        combos.extend(combinations(range(n_features), size))
    return combos


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": 0, "results": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def main():
    X, y = load_heart()
    n_features = X.shape[1]
    combos = all_combos(n_features)
    total = len(combos)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xtr_i, Xval, ytr_i, yval = train_test_split(
        Xtr, ytr, test_size=0.25, stratify=ytr, random_state=RANDOM_STATE
    )

    progress = load_progress()
    start = progress["completed"]
    if start >= total:
        print(f"Already complete: {total}/{total} combinations evaluated.")
        return progress

    end = min(start + BATCH_SIZE, total)
    print(f"Evaluating combinations {start} to {end-1} of {total}...")

    t0 = time.time()
    for i in range(start, end):
        idx = list(combos[i])
        ens = DACSDebateEnsemble(get_candidate_pool(), beta=FIXED_BETA)
        ens.fit(Xtr_i[:, idx], ytr_i)
        acc = float(accuracy_score(yval, ens.predict(Xval[:, idx])))
        progress["results"].append({"features": idx, "val_acc": acc})
    progress["completed"] = end
    save_progress(progress)

    elapsed = time.time() - t0
    print(f"Batch done in {elapsed:.1f}s. Progress: {end}/{total} ({100*end/total:.1f}%)")

    if end >= total:
        best = max(progress["results"], key=lambda r: r["val_acc"])
        best_names = [FEATURE_NAMES[i] for i in best["features"]]
        print(f"\nSEARCH COMPLETE. Best subset: {best_names} (val_acc={best['val_acc']:.4f})")

    return progress


if __name__ == "__main__":
    main()
