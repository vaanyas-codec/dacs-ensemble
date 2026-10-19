"""Tests for the belief-revision dynamics of DACSDebateEnsemble."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from candidate_pool import get_candidate_pool
from debate_ensemble import DACSDebateEnsemble


class RecordingEnsemble(DACSDebateEnsemble):
    """Records every belief stack the update loop passes to _wavg."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.recorded = []

    def _wavg(self, beliefs):
        self.recorded.append(beliefs.copy())
        return super()._wavg(beliefs)


@pytest.fixture(scope="module")
def fitted_data():
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5, flip_y=0.15, random_state=1
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def weighted_potential(beliefs, weights):
    """sum_k w_k * ||b_k - g||_1 summed over samples, g the weighted mean."""
    w = weights[:, None, None]
    g = (beliefs * w).sum(axis=0)
    return float((weights[:, None] * np.abs(beliefs - g).sum(axis=2)).sum())


def test_output_is_a_valid_probability_distribution(fitted_data):
    Xtr, Xte, ytr, _ = fitted_data
    ens = DACSDebateEnsemble(get_candidate_pool(), beta=0.3).fit(Xtr, ytr)
    p = ens.predict_proba(Xte)
    assert p.shape == (len(Xte), 2)
    assert np.all(p >= -1e-9)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_prediction_is_deterministic(fitted_data):
    Xtr, Xte, ytr, _ = fitted_data
    a = DACSDebateEnsemble(get_candidate_pool(), beta=0.3).fit(Xtr, ytr).predict(Xte)
    b = DACSDebateEnsemble(get_candidate_pool(), beta=0.3).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_weights_are_a_distribution(fitted_data):
    Xtr, _, ytr, _ = fitted_data
    ens = DACSDebateEnsemble(get_candidate_pool(), beta=0.3, k=4).fit(Xtr, ytr)
    assert len(ens.selected_names_) == 4
    assert ens.weights_.sum() == pytest.approx(1.0)
    assert np.all(ens.weights_ > 0)


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_disagreement_potential_never_increases(fitted_data, beta):
    """
    Checks numerically (no proof is claimed) that the
    weight-averaged L1 distance of each classifier from the weighted group
    mean does not increase from one round to the next.
    """
    Xtr, Xte, ytr, _ = fitted_data
    ens = RecordingEnsemble(get_candidate_pool(), beta=beta, max_rounds=8).fit(Xtr, ytr)
    ens.recorded.clear()
    ens.debate(Xte)
    potentials = [weighted_potential(b, ens.weights_) for b in ens.recorded]
    assert len(potentials) >= 2
    for before, after in zip(potentials, potentials[1:]):
        assert after <= before + 1e-9, (before, after)
    # and the process actually reduces disagreement overall
    assert potentials[-1] < potentials[0]


def test_debate_stops_within_max_rounds(fitted_data):
    Xtr, Xte, ytr, yte = fitted_data
    ens = DACSDebateEnsemble(get_candidate_pool(), beta=0.5, max_rounds=8).fit(Xtr, ytr)
    rounds = ens.accuracy_per_round(Xte, yte)
    assert 1 <= len(rounds) <= 9  # round 0 plus at most 8 revision rounds
