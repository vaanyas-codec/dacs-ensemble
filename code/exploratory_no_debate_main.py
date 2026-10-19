"""EXPLORATORY: Arm C (no revision rounds) on the 12 main datasets, resumable by dataset.
Compared against the saved full-method (Arm A) fold-mean accuracy and voting baseline."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import experiment_final as E
from candidate_pool import get_candidate_pool
from debate_ensemble import DACSDebateEnsemble

MAIN = [("pima_diabetes", E.load_pima, None), ("heart_disease", E.load_heart, None),
        ("australian_credit", E.load_australian, None), ("haberman_survival", E.load_haberman, None),
        ("liver_disorders", E.load_liver, None), ("banknote_auth", E.load_banknote, None),
        ("breast_cancer_wisconsin", E.load_breast_cancer_wisconsin, None), ("german_credit", E.load_german_credit, None),
        ("ionosphere", E.load_ionosphere, None), ("sonar", E.load_sonar, None),
        ("magic_gamma", E.load_magic, 3000), ("spambase", E.load_spambase, 2000)]
RES = os.path.join(HERE, "..", "results"); OUT = os.path.join(RES, "exploratory_no_debate_main.json")
budget, t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 240, time.time()
state = json.load(open(OUT)) if os.path.exists(OUT) else {"label": "EXPLORATORY, not preregistered", "datasets": {}}
for key, loader, cap in MAIN:
    if key in state["datasets"]: continue
    if time.time() - t0 > budget: break
    ref = json.load(open(os.path.join(RES, f"{key}.json")))
    X, y = loader()
    if cap and len(y) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=42)
    reps = 1 if ref["n_folds"] == 5 else 2
    accs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42).split(X, y):
        sc = E.make_preprocessor().fit(X[tr])
        ens = DACSDebateEnsemble(get_candidate_pool(), max_rounds=0, competitive_margin=0.05).fit(sc.transform(X[tr]), y[tr])
        accs.append(float(np.mean(ens.predict(sc.transform(X[te])) == y[te])))
    state["datasets"][key] = {"accC": float(np.mean(accs)), "accA": ref["dacs_mean"], "accV": ref["voting_mean"]}
    json.dump(state, open(OUT, "w"), indent=2)
    print(f"{key}: noDebate={np.mean(accs):.4f} fullMethod={ref['dacs_mean']:.4f} voting={ref['voting_mean']:.4f}", flush=True)
left = [k for k, _, _ in MAIN if k not in state["datasets"]]
print("INCOMPLETE, remaining: " + ", ".join(left) if left else "ALL COMPLETE")
