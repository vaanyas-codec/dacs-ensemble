"""
Flask demo for the DACS + Consensus/Debate Ensemble.

Imports the real, corrected implementation from ../code (no duplicated
logic, no hardcoded machine-specific paths). Data is read from ../data.

Run:
    cd app
    python3 app.py
Then open http://127.0.0.1:5050
"""
import hashlib
import os
import sys
import warnings

import joblib
import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "..", "code")
CACHE_DIR = os.path.join(HERE, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)

from candidate_pool import get_candidate_pool  # noqa: E402
from debate_ensemble import DACSDebateEnsemble  # noqa: E402
import experiment_final as ef  # noqa: E402
import healthcare_data as hd  # noqa: E402

app = Flask(__name__)

# Loaders in experiment_final.py use paths relative to the working directory
# ("data/..."), so switch to the package root, where ./data lives.
os.chdir(os.path.join(HERE, ".."))

DATASETS = {
    "pima": ("Pima Diabetes", ef.load_pima, None),
    "heart": ("Heart Disease", ef.load_heart, None),
    "australian": ("Australian Credit", ef.load_australian, None),
    "haberman": ("Haberman Survival", ef.load_haberman, None),
    "liver": ("Liver Disorders", ef.load_liver, None),
    "magic": ("Magic Gamma", ef.load_magic, 3000),
    "spambase": ("Spambase", ef.load_spambase, 2000),
    "banknote": ("Banknote Authentication", ef.load_banknote, None),
    "breast_cancer": ("Breast Cancer Wisconsin", ef.load_breast_cancer_wisconsin, None),
    "german": ("German Credit", ef.load_german_credit, None),
    "ionosphere": ("Ionosphere", ef.load_ionosphere, None),
    "sonar": ("Sonar", ef.load_sonar, None),
    # pre-specified medical evaluation (PREREGISTRATION_HEALTHCARE.md)
    "wdbc": ("WDBC Breast Cancer (medical)", hd.load_wdbc, None),
    "hepatitis": ("Hepatitis (medical)", hd.load_hepatitis, None),
    "ljubljana": ("Breast Cancer Ljubljana (medical)", hd.load_ljubljana, None),
    "prostate": ("Prostate Cancer Recurrence (medical)", hd.load_prostate, None),
}


def _baseline_voting():
    return VotingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("nb", GaussianNB()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ],
        voting="soft",
    )


@app.route("/")
def index():
    return render_template("index.html", datasets={k: v[0] for k, v in DATASETS.items()})


@app.route("/datasets")
def datasets():
    return jsonify({k: v[0] for k, v in DATASETS.items()})


@app.route("/run", methods=["POST"])
def run():
    payload = request.get_json(force=True) or {}
    key = payload.get("dataset", "pima")
    beta = float(payload.get("beta", 0.3))

    if key not in DATASETS:
        return jsonify({"error": f"Unknown dataset '{key}'"}), 400
    if not 0.0 < beta < 1.0:
        return jsonify({"error": "beta must be strictly between 0 and 1"}), 400

    name, loader, cap = DATASETS[key]
    X, y = loader()
    if cap and len(y) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=42)

    # Single held-out split. Beta comes from the user here (no search), so
    # the test split is used only for the final score.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    sc = ef.make_preprocessor()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    # --- Model persistence -------------------------------------------------
    # Cache key covers dataset identity (n, features -- changes if the CSV
    # changes), beta, and a fixed split seed, so a cache hit is only reused
    # when it is provably the same fit that would be produced from scratch.
    # The key also hashes the data CONTENT, so a cache built from a different version of a file (for example the
    # earlier, label-leaking Pima file, which had the same shape) is never reused.
    data_hash = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()[:12]
    cache_key = hashlib.sha256(
        f"{key}:{beta}:{X.shape[0]}:{X.shape[1]}:{data_hash}:seed42".encode()
    ).hexdigest()[:16]
    ens_path = os.path.join(CACHE_DIR, f"ens_{cache_key}.joblib")
    voting_path = os.path.join(CACHE_DIR, f"voting_{key}_{X.shape[0]}_{data_hash}.joblib")

    if os.path.exists(ens_path):
        ens = joblib.load(ens_path)
        cache_hit = True
    else:
        pool = get_candidate_pool()
        ens = DACSDebateEnsemble(pool, beta=beta)
        ens.fit(X_tr, y_tr)
        joblib.dump(ens, ens_path)
        cache_hit = False

    debate_acc = float(accuracy_score(y_te, ens.predict(X_te)))
    per_round = ens.accuracy_per_round(X_te, y_te)

    if os.path.exists(voting_path):
        voting = joblib.load(voting_path)
    else:
        voting = _baseline_voting().fit(X_tr, y_tr)
        joblib.dump(voting, voting_path)
    voting_acc = float(accuracy_score(y_te, voting.predict(X_te)))

    classifiers = [
        {
            "name": n,
            "cv_acc": round(float(ens.cv_accuracies_[n]), 4),
            "weight": round(float(w), 4),
        }
        for n, w in zip(ens.selected_names_, ens.weights_)
    ]

    return jsonify(
        {
            "dataset": name,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_test": int(len(y_te)),
            "beta": beta,
            "selected": ens.selected_names_,
            "classifiers": classifiers,
            "diversity": round(float(ens.diversity_), 4),
            "spread": round(float(ens.spread_), 4),
            "debate_acc": round(debate_acc, 4),
            "voting_acc": round(voting_acc, 4),
            "improvement_pp": round((debate_acc - voting_acc) * 100, 2),
            "rounds": [round(float(a), 4) for a in per_round],
            "cache_hit": cache_hit,
            "note": (
                "Single 80/20 split for demo speed. Rigorous results "
                "(repeated CV, McNemar, Holm correction) are in ../results."
            ),
        }
    )


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    removed = 0
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".joblib"):
            os.remove(os.path.join(CACHE_DIR, fname))
            removed += 1
    return jsonify({"removed": removed})


if __name__ == "__main__":
    app.run(debug=False, port=5050)
