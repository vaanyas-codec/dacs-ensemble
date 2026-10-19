"""
Data-integrity guards added after the Pima file was found to have been imputed
with class-conditional medians (a label leak that no amount of re-running the
same file could reveal).
"""
import numpy as np

import data_screen
import experiment_final as E


def test_screen_flags_class_conditional_fill():
    """The detector must catch the exact artefact that went unnoticed."""
    rng = np.random.RandomState(0)
    n = 600
    y = (rng.rand(n) < 0.35).astype(int)
    x = rng.normal(size=n)
    miss = rng.rand(n) < 0.5
    x[miss & (y == 0)] = 102.5      # class-conditional "median" fill
    x[miss & (y == 1)] = 169.5
    flags = data_screen.screen(x.reshape(-1, 1), y)
    assert {f["value"] for f in flags} == {102.5, 169.5}


def test_pima_zeros_are_missing_not_values():
    X, y = E.load_pima()
    names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
    assert np.isnan(X[:, 0]).sum() == 0                      # Pregnancies == 0 is genuine
    for j in (1, 2, 3, 4, 5):
        assert np.isnan(X[:, j]).sum() > 0, names[j]
        assert (X[:, j][~np.isnan(X[:, j])] == 0).sum() == 0, names[j]
    assert len(y) == 768 and int(y.sum()) == 268


def test_pima_has_no_class_conditional_fill_values():
    X, y = E.load_pima()
    assert data_screen.screen(X, y) == []


def test_preprocessor_uses_training_statistics_only():
    Xtr = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 20.0]])
    Xte = np.array([[np.nan, 1000.0], [2.5, np.nan]])        # test values must not influence the fill
    pre = E.make_preprocessor().fit(Xtr)
    imputer = pre.steps[0][1]
    assert imputer.statistics_.tolist() == [2.5, 20.0]       # train medians
    out = pre.transform(Xte)
    assert not np.isnan(out).any()


def test_preprocessor_is_noop_without_missing_values():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(50, 4))
    a = E.make_preprocessor().fit_transform(X)
    from sklearn.preprocessing import StandardScaler
    b = StandardScaler().fit_transform(X)
    assert np.allclose(a, b)
