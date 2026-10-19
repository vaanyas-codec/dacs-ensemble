"""
Final experiment runner for the Consensus/Debate Ensemble + DACS study.

Methodology (all fixes applied from the earlier leak diagnosis):
  - Beta is selected via an inner train/validation split carved out of the
    TRAINING fold only. The test fold is never used for any kind of
    selection, only for the single final prediction per fold.
  - Repeated Stratified K-Fold (5 splits x 3 repeats = 15 folds per dataset)
    for more stable estimates and more statistical power than a single
    5-fold split.
  - DACS selection uses pairwise prediction disagreement (measured via
    cross-validated out-of-fold predictions), not raw accuracy spread.
  - Spread is reported as a mean across folds, not the last fold's value.
  - McNemar's test on pooled fold predictions, plus Holm-Bonferroni
    correction across all datasets tested, reported alongside the raw
    p-values (both shown, nothing hidden).
  - A dedicated Magic Gamma sample-size sweep tests whether significance
    is being driven by sample size rather than a stronger true effect.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2

from debate_ensemble import DACSDebateEnsemble
from candidate_pool import get_candidate_pool

warnings.filterwarnings("ignore")

BETA_GRID = [0.1, 0.2, 0.3, 0.5, 0.7]
N_SPLITS = 5
N_REPEATS = 2  # 10 folds total per dataset
RANDOM_STATE = 42

os.makedirs("results_final", exist_ok=True)


def mcnemar(y_true, a, b):
    ab = ((a == y_true) & (b != y_true)).sum()
    ba = ((a != y_true) & (b == y_true)).sum()
    n = ab + ba
    if n == 0:
        return 1.0
    return float(1 - chi2.cdf((abs(ab - ba) - 1) ** 2 / n, 1))


def holm_bonferroni(pvals_dict):
    """Returns dict of name -> (raw_p, corrected_p, significant_after_correction)."""
    items = sorted(pvals_dict.items(), key=lambda x: x[1])
    m = len(items)
    out = {}
    max_adj_seen = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        max_adj_seen = max(max_adj_seen, adj)
        out[name] = {"raw_p": p, "holm_p": max_adj_seen, "sig": max_adj_seen < 0.05}
    return out


# ---------------- data loaders ----------------


def _data_path(filename):
    """
    Resolves a data file whether the script is run from inside code/ (with
    data/ copied alongside it, the old way) or from the package root (with
    code/ and data/ as siblings, the actual repo layout). Raises the same
    FileNotFoundError pandas would if neither location has the file, so
    callers' existing except FileNotFoundError handling still works.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("data", filename),                      # cwd/data/...
        os.path.join(here, "data", filename),                 # code/data/...
        os.path.join(here, "..", "data", filename),           # ../data/...
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find '{filename}' in any of: {candidates}"
    )


# In the raw UCI/Kaggle Pima file, physiologically impossible zeros encode
# missing measurements in these five columns (Pregnancies == 0 is genuine).
PIMA_ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_pima():
    """Raw Pima data with zero-coded missing values converted to NaN.

    The returned matrix still contains NaN. Imputation happens inside each
    training fold (see make_preprocessor), never on the full dataset: an
    earlier version of data/pima_diabetes.csv had been imputed with
    class-conditional medians, which leaks the label into the features.
    """
    df = pd.read_csv(_data_path("pima_diabetes.csv"))
    df[PIMA_ZERO_MEANS_MISSING] = df[PIMA_ZERO_MEANS_MISSING].replace(0, np.nan)
    return df.drop(columns=["Outcome"]).values.astype(float), df["Outcome"].values.astype(int)


def load_heart():
    df = pd.read_csv(_data_path("heart_disease.csv"))
    df = df.dropna()
    y = (df["target"].astype(float) > 0).astype(int).values
    X = df.drop(columns=["target"]).astype(float).values
    return X, y


def load_generic_numeric(path, header=None):
    df = pd.read_csv(path, header=header)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    y = df.iloc[:, -1].values.astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y


def load_banknote():
    return load_generic_numeric(_data_path("banknote.csv"))


def load_breast_cancer_wisconsin():
    df = pd.read_csv(_data_path("breast_cancer_wisconsin.csv"), header=None)
    df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce").dropna()
    y = (df.iloc[:, -1].astype(float) == 4).astype(int).values  # 2=benign->0, 4=malignant->1
    X = df.iloc[:, :-1].astype(float).values
    return X, y


def load_german_credit():
    df = pd.read_csv(_data_path("german_credit.csv"), header=None)
    y = (df.iloc[:, -1].astype(int) == 2).astype(int).values  # 1=good->0, 2=bad->1
    X_raw = df.iloc[:, :-1]
    X = pd.get_dummies(X_raw, drop_first=True).values.astype(float)
    return X, y


def load_ionosphere():
    df = pd.read_csv(_data_path("ionosphere.csv"), header=None)
    y = (df.iloc[:, -1] == "g").astype(int).values
    X = df.iloc[:, :-1].astype(float).values
    return X, y


def load_sonar():
    df = pd.read_csv(_data_path("sonar.csv"), header=None)
    y = (df.iloc[:, -1] == "R").astype(int).values
    X = df.iloc[:, :-1].astype(float).values
    return X, y


def load_magic():
    return load_generic_numeric(_data_path("magic_real.csv"))


def load_spambase():
    return load_generic_numeric(_data_path("spambase_real.csv"))


def load_australian():
    return load_generic_numeric(_data_path("australian_real.csv"))


def load_haberman():
    return load_generic_numeric(_data_path("haberman_real.csv"))


def load_liver():
    return load_generic_numeric(_data_path("liver_real.csv"))


# ---------------- core experiment ----------------

def make_preprocessor():
    """Median imputation followed by standardisation.

    Fit on the TRAINING fold only, then applied to the test fold. Imputation
    uses the feature median of the training fold, never the label. On a
    dataset with no missing values the imputer is an exact no-op, so results
    for those datasets are unchanged.
    """
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def fit_predict_fold(pool, Xtr, ytr, Xte, competitive_margin=0.05):
    """
    Choose beta on an inner train/validation split of the TRAINING fold
    only, refit on the full training fold, and predict the test fold once.

    The signature deliberately has no parameter for test labels, so nothing
    in here can use them: this is what makes the beta/round-selection leak
    that affected an earlier version impossible by construction.
    """
    Xtr_inner, Xval, ytr_inner, yval = train_test_split(
        Xtr, ytr, test_size=0.25, stratify=ytr, random_state=RANDOM_STATE
    )

    best_beta, best_val_acc = BETA_GRID[0], -1
    for beta in BETA_GRID:
        ens = DACSDebateEnsemble(pool, beta=beta, competitive_margin=competitive_margin)
        ens.fit(Xtr_inner, ytr_inner)
        val_acc = accuracy_score(yval, ens.predict(Xval))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_beta = beta

    final_ens = DACSDebateEnsemble(pool, beta=best_beta, competitive_margin=competitive_margin)
    final_ens.fit(Xtr, ytr)
    return final_ens.predict(Xte), best_beta, final_ens


def run_dataset(name, X, y, max_n_for_speed=None, verbose_folds=False):
    if max_n_for_speed and len(y) > max_n_for_speed:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_n_for_speed, stratify=y, random_state=RANDOM_STATE
        )

    print(f"\n{'='*62}\n  {name}  (n={len(y)}, f={X.shape[1]})\n{'='*62}")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    pool = get_candidate_pool()
    p1 = [
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]

    dacs_accs, voting_accs, spreads, diversities, chosen_betas = [], [], [], [], []
    all_yt, all_pd, all_pv = [], [], []

    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        sc = make_preprocessor()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        # Test labels (yte) are deliberately NOT passed to fit_predict_fold.
        pred, best_beta, final_ens = fit_predict_fold(pool, Xtr, ytr, Xte)
        acc = accuracy_score(yte, pred)

        dacs_accs.append(acc)
        spreads.append(final_ens.spread_)
        diversities.append(final_ens.diversity_)
        chosen_betas.append(best_beta)

        v = VotingClassifier(estimators=p1, voting="soft")
        v.fit(Xtr, ytr)
        pv = v.predict(Xte)
        voting_accs.append(accuracy_score(yte, pv))

        all_yt.extend(yte)
        all_pd.extend(pred)
        all_pv.extend(pv)

        if verbose_folds:
            print(
                f"  Fold {fold+1:2d}: DACS+D={acc:.4f} (b={best_beta}), "
                f"Voting={voting_accs[-1]:.4f}, div={final_ens.diversity_:.3f}, "
                f"sel={final_ens.selected_names_}"
            )

    all_yt = np.array(all_yt)
    p = mcnemar(all_yt, np.array(all_pd), np.array(all_pv))

    r = {
        "n": int(len(y)),
        "dacs_mean": float(np.mean(dacs_accs)),
        "dacs_std": float(np.std(dacs_accs)),
        "voting_mean": float(np.mean(voting_accs)),
        "voting_std": float(np.std(voting_accs)),
        "spread_mean": float(np.mean(spreads)),
        "diversity_mean": float(np.mean(diversities)),
        "mcnemar_p": p,
        "n_folds": len(dacs_accs),
    }
    print(f"  DACS+Debate: {r['dacs_mean']:.4f} +/- {r['dacs_std']:.4f}")
    print(f"  Soft Voting: {r['voting_mean']:.4f} +/- {r['voting_std']:.4f}")
    print(f"  Diversity (mean pairwise disagreement): {r['diversity_mean']:.4f}")
    print(f"  McNemar p (raw): {p:.4f}")

    with open(f"results_final/{name.lower().replace(' ', '_')}.json", "w") as f:
        json.dump(r, f, indent=2)
    return r


def magic_sample_size_sweep(existing_3000_result=None):
    """Tests whether Magic Gamma's significance is a sample-size artifact.
    Reuses the main run's n=3000 result instead of recomputing it."""
    print(f"\n{'='*62}\n  MAGIC GAMMA SAMPLE-SIZE SWEEP\n{'='*62}")
    X, y = load_magic()
    sizes = [300, 700, 1500]
    sweep_results = {}
    for n in sizes:
        Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=RANDOM_STATE)
        r = run_dataset(f"Magic_n{n}", Xs, ys, max_n_for_speed=None, verbose_folds=False)
        sweep_results[n] = r
    if existing_3000_result is not None:
        sweep_results[3000] = existing_3000_result
    with open("results_final/magic_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    return sweep_results


def main():
    datasets = [
        ("Pima Diabetes", *load_pima(), None),
        ("Heart Disease", *load_heart(), None),
        ("Australian Credit", *load_australian(), None),
        ("Haberman Survival", *load_haberman(), None),
        ("Liver Disorders", *load_liver(), None),
        ("Magic Gamma", *load_magic(), 3000),
        ("Spambase", *load_spambase(), 2000),
        ("Banknote Auth", *load_banknote(), None),
        ("Breast Cancer Wisconsin", *load_breast_cancer_wisconsin(), None),
        ("German Credit", *load_german_credit(), None),
        ("Ionosphere", *load_ionosphere(), None),
        ("Sonar", *load_sonar(), None),
    ]

    results = {}
    for name, X, y, cap in datasets:
        results[name] = run_dataset(name, X, y, max_n_for_speed=cap, verbose_folds=True)

    pvals = {name: r["mcnemar_p"] for name, r in results.items()}
    holm = holm_bonferroni(pvals)

    print(f"\n{'='*62}\nFINAL SUMMARY (leak-free, repeated CV, Holm-corrected)\n{'='*62}")
    for name, r in results.items():
        h = holm[name]
        print(f"\n{name} (n={r['n']}, folds={r['n_folds']}):")
        print(f"  DACS+Debate: {r['dacs_mean']:.4f}  Voting: {r['voting_mean']:.4f}  "
              f"Delta: {(r['dacs_mean']-r['voting_mean'])*100:+.2f}pp")
        print(f"  Diversity: {r['diversity_mean']:.4f}  Spread: {r['spread_mean']:.4f}")
        print(f"  p_raw={h['raw_p']:.4f}  p_holm={h['holm_p']:.4f}  "
              f"sig={'YES' if h['sig'] else 'no'}")

    with open("results_final/summary.json", "w") as f:
        json.dump({"results": results, "holm": holm}, f, indent=2)

    magic_sweep = magic_sample_size_sweep(existing_3000_result=results["Magic Gamma"])
    print(f"\n{'='*62}\nMAGIC GAMMA SWEEP SUMMARY\n{'='*62}")
    for n, r in magic_sweep.items():
        print(f"n={n}: Delta={100*(r['dacs_mean']-r['voting_mean']):+.2f}pp  p={r['mcnemar_p']:.4f}")


if __name__ == "__main__":
    main()
