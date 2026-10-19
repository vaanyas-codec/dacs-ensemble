"""Write results/dataset_characteristics.csv (size, features, class balance, missing cells) for the 12 main datasets,
using the same loaders and subsampling caps as the main study.   python3 dataset_characteristics.py"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from sklearn.model_selection import train_test_split
import experiment_final as E

MAIN = [("pima_diabetes", E.load_pima, None), ("heart_disease", E.load_heart, None), ("australian_credit", E.load_australian, None),
        ("haberman_survival", E.load_haberman, None), ("liver_disorders", E.load_liver, None), ("magic_gamma", E.load_magic, 3000),
        ("spambase", E.load_spambase, 2000), ("banknote_auth", E.load_banknote, None),
        ("breast_cancer_wisconsin", E.load_breast_cancer_wisconsin, None), ("german_credit", E.load_german_credit, None),
        ("ionosphere", E.load_ionosphere, None), ("sonar", E.load_sonar, None)]
RES = os.path.join(HERE, "..", "results")
rows = []
for name, loader, cap in MAIN:
    X, y = loader()
    if cap and len(y) > cap:
        X, _, y, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=E.RANDOM_STATE)
    assert json.load(open(os.path.join(RES, f"{name}.json")))["n"] == len(y), name
    rows.append([name, len(y), X.shape[1], round(100 * float(np.mean(y)), 1), int(np.isnan(X).sum())])
with open(os.path.join(RES, "dataset_characteristics.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "n", "features", "positive_pct", "missing_cells_before_imputation"]); w.writerows(rows)
print("written", len(rows), "rows")
