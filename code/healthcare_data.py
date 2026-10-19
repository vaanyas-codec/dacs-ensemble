"""
Loaders for the pre-specified medical datasets (see PREREGISTRATION_HEALTHCARE.md).
All return (X, y) with y in {0,1} and X possibly containing NaN (imputed inside each training fold).
Files live in data/healthcare/.
"""
import os
import numpy as np
import pandas as pd

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "healthcare")


def load_wdbc():
    """Wisconsin Diagnostic Breast Cancer (569 x 30). y = 1 benign, 0 malignant (scikit-learn coding)."""
    d = pd.read_csv(os.path.join(_DIR, "wdbc.csv"))
    return d.drop(columns=["target"]).values.astype(float), d["target"].values.astype(int)


def load_hepatitis():
    """UCI Hepatitis (154 x 19). Class column: 1 = positive class as coded in the file. Missing cells are blank."""
    d = pd.read_csv(os.path.join(_DIR, "hepatitis.csv"), encoding="utf-8-sig")
    return d.drop(columns=["Class"]).apply(pd.to_numeric, errors="coerce").values.astype(float), d["Class"].values.astype(int)


def load_ljubljana():
    """UCI Breast Cancer (Ljubljana), 286 rows, 9 categorical features, y = recurrence. '?' kept as its own category;
    one-hot encoding uses only the set of category labels, never the target."""
    d = pd.read_csv(os.path.join(_DIR, "breast_cancer_ljubljana.csv"), header=None, quotechar="'")
    y = (d.iloc[:, -1].str.strip() == "recurrence-events").astype(int).values
    X = pd.get_dummies(d.iloc[:, :-1].astype(str)).values.astype(float)
    return X, y


def load_prostate():
    """Prostate-cancer recurrence after surgery (medicaldata::blood_storage), 316 rows, y = Recurrence.
    Dropped, as fixed in the protocol before any model was run: row index, Censor and TimeToRecurrence (follow-up
    outcome information) and AnyAdjTherapy and AdjRadTherapy (therapy given after surgery)."""
    d = pd.read_csv(os.path.join(_DIR, "prostate_blood_storage.csv"))
    y = d["Recurrence"].values.astype(int)
    X = d.drop(columns=["rownames", "Recurrence", "Censor", "TimeToRecurrence", "AnyAdjTherapy", "AdjRadTherapy"])
    return X.apply(pd.to_numeric, errors="coerce").values.astype(float), y


LOADERS = {"wdbc": load_wdbc, "hepatitis": load_hepatitis, "ljubljana": load_ljubljana, "prostate": load_prostate}
