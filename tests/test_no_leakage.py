"""
Leakage guards for the evaluation protocol in experiment_final.py.
"""
import inspect

import numpy as np
import pytest


def test_run_dataset_never_scores_rounds_against_test_labels():
    """
    Guards the exact pattern that caused the original bug: calling
    accuracy_per_round(...) with the test fold and taking the max. This is
    a pattern check for one known failure, not a proof that no leak exists;
    the label-shuffle test below is the behavioural check.
    """
    import experiment_final

    src = inspect.getsource(experiment_final.run_dataset)
    assert "accuracy_per_round" not in src
    assert "max(rounds" not in src


@pytest.mark.slow
def test_shuffled_labels_score_at_chance(tmp_path, monkeypatch):
    """
    With labels carrying no information, a leak-free protocol must score
    near 0.5. Any procedure that picks its configuration using test labels
    would score above chance here.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results_final").mkdir()

    import experiment_final

    monkeypatch.setattr(experiment_final, "N_REPEATS", 2)

    rng = np.random.RandomState(7)
    n = 240
    X = rng.normal(size=(n, 6))
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)  # labels independent of X

    result = experiment_final.run_dataset("shuffled_null", X, y)
    assert abs(result["dacs_mean"] - 0.5) < 0.07, result["dacs_mean"]
    assert abs(result["voting_mean"] - 0.5) < 0.07, result["voting_mean"]
    # no real effect exists, so the comparison should not look significant
    assert result["mcnemar_p"] > 0.01
