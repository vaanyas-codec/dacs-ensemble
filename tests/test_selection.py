"""
Regression tests for DACS classifier selection.

These use synthetic estimators with an exactly known error rate, so the
expected behaviour does not depend on any dataset.

FlipEstimator reads the true label from column 0 of X and a uniform random
number u from column 1, and flips its answer when u < flip_rate. Its
accuracy is therefore 1 - flip_rate, and two estimators with different
flip rates disagree on about |flip_a - flip_b| of samples.
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from debate_ensemble import dacs_select


class FlipEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, flip_rate=0.0):
        self.flip_rate = flip_rate

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        label = X[:, 0].astype(int)
        u = X[:, 1]
        return np.where(u < self.flip_rate, 1 - label, label)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    n = 900
    y = rng.randint(0, 2, n)
    u = rng.uniform(0, 1, n)
    return np.column_stack([y, u]), y


def make_pool():
    return {
        "strong_1": FlipEstimator(0.01),
        "strong_2": FlipEstimator(0.02),
        "strong_3": FlipEstimator(0.03),
        "strong_4": FlipEstimator(0.04),
        # Above the old 0.55 floor, but far below the best candidate.
        "weak_1": FlipEstimator(0.30),
        "weak_2": FlipEstimator(0.35),
    }


def test_competitive_margin_excludes_weak_classifiers(data):
    """The fix: weak classifiers must not be chosen just for disagreeing."""
    X, y = data
    selected, _, _, _ = dacs_select(X, y, make_pool(), k=3, min_acc=0.55)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert not any(name.startswith("weak") for name in selected), selected


def test_without_margin_weak_classifiers_get_selected(data):
    """
    Guards the test above. With the margin effectively disabled, the
    original flaw must reappear (a weak classifier is chosen because it
    disagrees most). If this test ever fails, the test above no longer
    proves anything.
    """
    X, y = data
    selected, _, _, _ = dacs_select(
        X, y, make_pool(), k=3, min_acc=0.55, competitive_margin=1.0
    )
    assert any(name.startswith("weak") for name in selected), selected


def test_falls_back_when_too_few_competitive_candidates(data):
    """If fewer than k candidates are competitive, still return k."""
    X, y = data
    pool = {
        "strong_1": FlipEstimator(0.01),
        "strong_2": FlipEstimator(0.02),
        "weak_1": FlipEstimator(0.30),
        "weak_2": FlipEstimator(0.35),
    }
    selected, _, _, _ = dacs_select(X, y, pool, k=3, min_acc=0.55)
    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_reported_diversity_matches_pairwise_disagreement(data):
    """Diversity should be the mean pairwise disagreement of the chosen set."""
    X, y = data
    pool = {
        "a": FlipEstimator(0.01),
        "b": FlipEstimator(0.05),
        "c": FlipEstimator(0.08),
    }
    selected, spread, diversity, cv_accs = dacs_select(X, y, pool, k=3, min_acc=0.55)
    assert set(selected) == {"a", "b", "c"}
    # Expected disagreement rates: |a-b|=.04, |a-c|=.07, |b-c|=.03 -> mean .0467
    assert diversity == pytest.approx(0.0467, abs=0.02)
    assert spread == pytest.approx(0.07, abs=0.03)
