"""
Tests the Disagreement-Gated Debate variant against soft voting, using the
exact same leak-free protocol as experiment_final.py: beta chosen via an
inner train/validation split on the training fold only, one prediction per
test sample, repeated stratified CV, McNemar testing, Holm-Bonferroni
correction across datasets.
"""
import os, json, sys, warnings
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from debate_ensemble import DisagreementGatedDebateEnsemble
from candidate_pool import get_candidate_pool
from experiment_final import (
    BETA_GRID, RANDOM_STATE, mcnemar, holm_bonferroni,
    make_preprocessor, load_pima, load_heart, load_australian, load_haberman, load_liver,
    load_magic, load_spambase, load_banknote, load_breast_cancer_wisconsin,
    load_german_credit, load_ionosphere, load_sonar,
)

warnings.filterwarnings("ignore")
os.makedirs("results_gated", exist_ok=True)

LOADERS = {
    "pima": ("Pima Diabetes", load_pima, None),
    "heart": ("Heart Disease", load_heart, None),
    "australian": ("Australian Credit", load_australian, None),
    "haberman": ("Haberman Survival", load_haberman, None),
    "liver": ("Liver Disorders", load_liver, None),
    "magic": ("Magic Gamma", load_magic, 3000),
    "spambase": ("Spambase", load_spambase, 2000),
    "banknote": ("Banknote Auth", load_banknote, None),
    "breast_cancer": ("Breast Cancer Wisconsin", load_breast_cancer_wisconsin, None),
    "german": ("German Credit", load_german_credit, None),
    "ionosphere": ("Ionosphere", load_ionosphere, None),
    "sonar": ("Sonar", load_sonar, None),
}


def run_gated(name, X, y, max_n_for_speed=None, n_repeats=2):
    if max_n_for_speed and len(y) > max_n_for_speed:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_n_for_speed, stratify=y, random_state=RANDOM_STATE
        )

    print(f"\n{'='*62}\n  {name} [GATED]  (n={len(y)}, f={X.shape[1]})\n{'='*62}")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    pool = get_candidate_pool()
    p1 = [
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]

    gated_accs, voting_accs, gated_fracs = [], [], []
    all_yt, all_pg, all_pv = [], [], []

    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        sc = make_preprocessor()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        Xtr_inner, Xval, ytr_inner, yval = train_test_split(
            Xtr, ytr, test_size=0.25, stratify=ytr, random_state=RANDOM_STATE
        )

        best_beta, best_val_acc = BETA_GRID[0], -1
        for beta in BETA_GRID:
            ens = DisagreementGatedDebateEnsemble(pool, beta=beta)
            ens.fit(Xtr_inner, ytr_inner)
            val_acc = accuracy_score(yval, ens.predict(Xval))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_beta = beta

        final_ens = DisagreementGatedDebateEnsemble(pool, beta=best_beta)
        final_ens.fit(Xtr, ytr)
        pred = final_ens.predict(Xte)
        acc = accuracy_score(yte, pred)
        frac_gated = final_ens.fraction_gated(Xte)

        gated_accs.append(acc)
        gated_fracs.append(frac_gated)

        v = VotingClassifier(estimators=p1, voting="soft")
        v.fit(Xtr, ytr)
        pv = v.predict(Xte)
        voting_accs.append(accuracy_score(yte, pv))

        all_yt.extend(yte)
        all_pg.extend(pred)
        all_pv.extend(pv)

        print(
            f"  Fold {fold+1:2d}: Gated={acc:.4f} (beta={best_beta}), "
            f"Voting={voting_accs[-1]:.4f}, frac_unanimous={frac_gated:.3f}"
        )

    all_yt = np.array(all_yt)
    p = mcnemar(all_yt, np.array(all_pg), np.array(all_pv))

    r = {
        "n": int(len(y)),
        "gated_mean": float(np.mean(gated_accs)),
        "gated_std": float(np.std(gated_accs)),
        "voting_mean": float(np.mean(voting_accs)),
        "voting_std": float(np.std(voting_accs)),
        "frac_gated_mean": float(np.mean(gated_fracs)),
        "mcnemar_p": p,
        "n_folds": len(gated_accs),
    }
    print(f"  Gated Debate: {r['gated_mean']:.4f} +/- {r['gated_std']:.4f}")
    print(f"  Soft Voting:  {r['voting_mean']:.4f} +/- {r['voting_std']:.4f}")
    print(f"  Fraction unanimous (gated, skipped debate): {r['frac_gated_mean']:.3f}")
    print(f"  McNemar p (raw): {p:.4f}")

    with open(f"results_gated/{name.lower().replace(' ', '_')}.json", "w") as f:
        json.dump(r, f, indent=2)
    return r


if __name__ == "__main__":
    key = sys.argv[1]
    fast = len(sys.argv) > 2 and sys.argv[2] == "fast"
    name, loader, cap = LOADERS[key]
    X, y = loader()
    run_gated(name, X, y, max_n_for_speed=cap, n_repeats=1 if fast else 2)
