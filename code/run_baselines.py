"""
Extra baselines on the SAME folds, caps and preprocessing as the main study.

Question answered: does the competitive-margin + diversity selection add
anything beyond sensible accuracy-based ways of using the same 13-classifier
pool?

Arms (all fitted on the training fold only, scored once on the test fold):
  A  proposed: margin selection + accuracy weights + belief revision (beta chosen on an inner split)
  S  margin selection + accuracy weights, NO revision rounds
  T  top-4 classifiers by cross-validated accuracy, accuracy-weighted (no diversity step)
  W  all 13 classifiers, accuracy-weighted
  B  best single classifier in the pool (by cross-validated accuracy)
  V  soft voting over DT(5), NB, kNN(5), LR (the main-study baseline)

Weights for S, T, W use the same form as the method: w_k proportional to exp(2 * CV accuracy_k).

Resumable (state is written after every fold):
    python3 run_baselines.py pima 200
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
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import experiment_final as E
from candidate_pool import get_candidate_pool
from debate_ensemble import DACSDebateEnsemble, _cv_stats

# key: (results-file name, loader, row cap, repeats) -- mirrors run_one.py and the stored main-study runs
DATASETS = {
    "pima": ("pima_diabetes", E.load_pima, None, 2),
    "heart": ("heart_disease", E.load_heart, None, 2),
    "australian": ("australian_credit", E.load_australian, None, 2),
    "haberman": ("haberman_survival", E.load_haberman, None, 2),
    "liver": ("liver_disorders", E.load_liver, None, 2),
    "magic": ("magic_gamma", E.load_magic, 3000, 1),
    "spambase": ("spambase", E.load_spambase, 2000, 1),
    "banknote": ("banknote_auth", E.load_banknote, None, 2),
    "breast_cancer": ("breast_cancer_wisconsin", E.load_breast_cancer_wisconsin, None, 2),
    "german": ("german_credit", E.load_german_credit, None, 2),
    "ionosphere": ("ionosphere", E.load_ionosphere, None, 2),
    "sonar": ("sonar", E.load_sonar, None, 2),
}
OUT = os.path.join(HERE, "..", "results", "baselines")
ALPHA, K = 2.0, 4
SEED = E.RANDOM_STATE


def weighted(probs, cv_accs, names):
    w = np.exp(ALPHA * np.array([cv_accs[n] for n in names]))
    w = w / w.sum()
    return sum(wi * probs[n] for wi, n in zip(w, names))


def voting():
    return VotingClassifier(estimators=[
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=SEED)), ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier(n_neighbors=5)), ("lr", LogisticRegression(max_iter=1000, random_state=SEED))],
        voting="soft")


def run(key, budget):
    name, loader, cap, repeats = DATASETS[key]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    X, y = loader()
    if cap and len(y) > cap:                                   # same subsample as run_dataset
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=SEED)
    splits = list(RepeatedStratifiedKFold(n_splits=E.N_SPLITS, n_repeats=repeats, random_state=SEED).split(X, y))
    state = json.load(open(path)) if os.path.exists(path) else {
        "dataset": name, "n": int(len(y)), "n_folds_total": len(splits), "folds": []}
    done = {f["i"] for f in state["folds"]}
    pool = get_candidate_pool()
    t0 = time.time()
    for i, (tr, te) in enumerate(splits):
        if i in done:
            continue
        if time.time() - t0 > budget:
            break
        pre = E.make_preprocessor().fit(X[tr])
        Xtr, Xte, ytr, yte = pre.transform(X[tr]), pre.transform(X[te]), y[tr], y[te]

        cv_accs, _ = _cv_stats(Xtr, ytr, pool, 3)
        probs, classes = {}, None
        for n, clf in pool.items():
            m = clone(clf).fit(Xtr, ytr)
            probs[n], classes = m.predict_proba(Xte), m.classes_
        lab = lambda p: classes[np.argmax(p, axis=1)]

        names_by_acc = sorted(cv_accs, key=lambda n: (-cv_accs[n], n))
        pa, beta, _ens = E.fit_predict_fold(pool, Xtr, ytr, Xte)                      # A
        ps = DACSDebateEnsemble(pool, beta=0.3, max_rounds=0).fit(Xtr, ytr).predict(Xte)  # S
        pt = lab(weighted(probs, cv_accs, names_by_acc[:K]))                           # T
        pw = lab(weighted(probs, cv_accs, list(pool)))                                 # W
        pb = lab(probs[names_by_acc[0]])                                               # B
        pv = voting().fit(Xtr, ytr).predict(Xte)                                       # V

        state["folds"].append({"i": i, "y": yte.tolist(), "A": pa.tolist(), "S": ps.tolist(), "T": pt.tolist(),
                               "W": pw.tolist(), "B": pb.tolist(), "V": pv.tolist(), "beta": beta,
                               "best_single": names_by_acc[0]})
        json.dump(state, open(path, "w"))
        a = lambda p: np.mean(np.asarray(p) == yte)
        print(f"{name} fold {i+1}/{len(splits)}: A={a(pa):.3f} S={a(ps):.3f} T={a(pt):.3f} W={a(pw):.3f} "
              f"B={a(pb):.3f} V={a(pv):.3f}", flush=True)
    n_done = len(state["folds"])
    print(("ALL COMPLETE" if n_done == len(splits) else f"INCOMPLETE {n_done}/{len(splits)}"),
          f"[{name}] ({time.time()-t0:.0f}s this call)")


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 200.0)
