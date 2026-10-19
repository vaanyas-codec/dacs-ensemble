"""
DACS (Diversity-Aware Classifier Selection) v2 + Consensus/Debate Ensemble.

CHANGE FROM v1: classifier selection now uses actual pairwise prediction
DISAGREEMENT (measured via out-of-fold cross-validated predictions) rather
than raw accuracy spread. Two classifiers can have very different accuracy
and still make the same mistakes (redundant), or similar accuracy and make
different mistakes (genuinely complementary, which is what should help an
ensemble). Disagreement rate captures the second thing directly; accuracy
spread only captures it indirectly.

Accuracy spread is still computed and reported for comparability with
earlier versions of this project, but it is no longer the selection
criterion.
"""

import hashlib

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict



_CV_CACHE = {}
_CV_CACHE_MAX = 8


def _cv_stats(X_train, y_train, candidate_pool, cv):
    """
    Per-candidate cross-validated accuracy and out-of-fold predictions.

    These depend only on the training data, the candidate pool and cv, never
    on beta or the competitive margin, so they are memoized on a hash of
    exactly those inputs. This is a pure speed optimisation: a cache hit
    returns the identical values a fresh computation would produce.
    """
    key = (
        hashlib.sha1(np.ascontiguousarray(X_train).tobytes()).hexdigest(),
        hashlib.sha1(np.ascontiguousarray(y_train).tobytes()).hexdigest(),
        X_train.shape, cv,
        tuple((n, repr(clf)) for n, clf in candidate_pool.items()),
    )
    if key in _CV_CACHE:
        return _CV_CACHE[key]
    cv_accs, oof_preds = {}, {}
    for name, clf in candidate_pool.items():
        try:
            scores = cross_val_score(clone(clf), X_train, y_train, cv=cv, scoring="accuracy")
            cv_accs[name] = float(scores.mean())
            oof_preds[name] = cross_val_predict(clone(clf), X_train, y_train, cv=cv)
        except Exception:
            cv_accs[name] = 0.0
            oof_preds[name] = np.zeros(len(y_train), dtype=int)
    if len(_CV_CACHE) >= _CV_CACHE_MAX:
        _CV_CACHE.pop(next(iter(_CV_CACHE)))
    _CV_CACHE[key] = (cv_accs, oof_preds)
    return cv_accs, oof_preds

def dacs_select(X_train, y_train, candidate_pool, k=4, min_acc=0.55, cv=3, competitive_margin=0.05):
    """
    Selects k classifiers from candidate_pool that (a) individually clear
    min_acc accuracy, AND are within `competitive_margin` of the single best
    candidate's accuracy (this is the fix: diversity is only sought among
    genuinely competitive classifiers, not used as an excuse to include
    weak ones just because being wrong in a unique way maximises
    disagreement), and (b) collectively maximise average pairwise
    disagreement among that competitive subset, via greedy forward
    selection starting from the strongest individual classifier.

    Returns
    -------
    selected : list of names
    spread : accuracy spread among selected (max - min)
    diversity : mean pairwise disagreement rate among selected classifiers
    cv_accs : dict of name -> cv accuracy for ALL candidates
    """
    cv_accs, oof_preds = _cv_stats(X_train, y_train, candidate_pool, cv)
    cv_accs = dict(cv_accs)

    best_overall_acc = max(cv_accs.values())
    competitive_floor = max(min_acc, best_overall_acc - competitive_margin)
    eligible = {n: a for n, a in cv_accs.items() if a >= competitive_floor}
    if len(eligible) < k:
        # Not enough competitive candidates; relax back to the plain min_acc
        # floor rather than silently including weak classifiers unnoticed.
        eligible = {n: a for n, a in cv_accs.items() if a >= min_acc}
    if len(eligible) < k:
        eligible = dict(sorted(cv_accs.items(), key=lambda x: -x[1])[:k])

    names_sorted_by_acc = sorted(eligible.items(), key=lambda x: -x[1])
    selected = [names_sorted_by_acc[0][0]]  # start with the strongest classifier

    def disagreement_rate(name_a, name_b):
        return float(np.mean(oof_preds[name_a] != oof_preds[name_b]))

    while len(selected) < k and len(selected) < len(names_sorted_by_acc):
        best_name, best_score = None, -1.0
        for name, acc in names_sorted_by_acc:
            if name in selected:
                continue
            avg_disagreement = np.mean(
                [disagreement_rate(name, s) for s in selected]
            )
            if avg_disagreement > best_score:
                best_score = avg_disagreement
                best_name = name
        if best_name is None:
            break
        selected.append(best_name)

    accs = [eligible.get(n, cv_accs[n]) for n in selected]
    spread = max(accs) - min(accs)

    pairwise = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairwise.append(disagreement_rate(selected[i], selected[j]))
    diversity = float(np.mean(pairwise)) if pairwise else 0.0

    return selected, spread, diversity, cv_accs


class DACSDebateEnsemble:
    def __init__(self, candidate_pool, k=4, beta=0.3, max_rounds=8,
                 tol=1e-4, disagreement_threshold=0.02, alpha=2.0, min_acc=0.55,
                 competitive_margin=0.05):
        self.candidate_pool = candidate_pool
        self.k = k
        self.beta = beta
        self.max_rounds = max_rounds
        self.tol = tol
        self.disagreement_threshold = disagreement_threshold
        self.alpha = alpha
        self.min_acc = min_acc
        self.competitive_margin = competitive_margin
        self.fitted_ = None
        self.weights_ = None
        self.classes_ = None
        self.selected_names_ = None
        self.cv_accuracies_ = None
        self.spread_ = None
        self.diversity_ = None

    def fit(self, X, y):
        selected_names, spread, diversity, cv_accs = dacs_select(
            X, y, self.candidate_pool, k=self.k, min_acc=self.min_acc,
            competitive_margin=self.competitive_margin
        )
        accs = np.array([cv_accs[n] for n in selected_names])
        exp_w = np.exp(self.alpha * accs)
        self.weights_ = exp_w / exp_w.sum()
        self.selected_names_ = selected_names
        self.cv_accuracies_ = cv_accs
        self.spread_ = spread
        self.diversity_ = diversity
        self.fitted_ = []
        for name in selected_names:
            clf = clone(self.candidate_pool[name])
            clf.fit(X, y)
            self.fitted_.append((name, clf))
        self.classes_ = self.fitted_[0][1].classes_
        return self

    def _beliefs(self, X):
        return np.stack([clf.predict_proba(X) for _, clf in self.fitted_], axis=0)

    def _wavg(self, beliefs):
        w = self.weights_[:, np.newaxis, np.newaxis]
        return (beliefs * w).sum(axis=0)

    def debate(self, X, return_history=False):
        beliefs = self._beliefs(X)
        history = [self._wavg(beliefs)]
        for _ in range(self.max_rounds):
            ga = self._wavg(beliefs)
            nb = np.empty_like(beliefs)
            max_change = 0.0
            for i in range(len(self.fitted_)):
                d = np.abs(beliefs[i] - ga).sum(axis=1)
                u = beliefs[i].copy()
                mask = d > self.disagreement_threshold
                if mask.any():
                    dm = d[mask]
                    persuadability = self.beta * (1.0 - self.weights_[i] + 0.05)
                    sb = np.clip(persuadability * (dm / (dm.max() + 1e-9)), 0, 0.95)[:, np.newaxis]
                    u[mask] = (1 - sb) * beliefs[i][mask] + sb * ga[mask]
                max_change = max(max_change, np.abs(u - beliefs[i]).max())
                nb[i] = u
            beliefs = nb
            history.append(self._wavg(beliefs))
            if max_change < self.tol:
                break
        final = self._wavg(beliefs)
        return (final, history) if return_history else final

    def predict(self, X):
        return self.classes_[np.argmax(self.debate(X), axis=1)]

    def predict_proba(self, X):
        return self.debate(X)

    def accuracy_per_round(self, X, y):
        _, history = self.debate(X, return_history=True)
        return [float(np.mean(self.classes_[np.argmax(b, axis=1)] == y)) for b in history]


class DisagreementGatedDebateEnsemble(DACSDebateEnsemble):
    """
    Variant that only runs debate on samples where the selected classifiers
    disagree at Round 0. Samples with unanimous Round-0 agreement are locked
    to that prediction and never touched by the revision rounds.

    Hypothesis: debate's losses on high-accuracy datasets (e.g. Banknote,
    Spambase) come from perturbing already-correct, unanimous predictions
    on "easy" samples that had nothing to resolve. Gating removes that
    failure mode by construction, since agreeing samples are never revised.
    """

    def predict(self, X):
        beliefs0 = self._beliefs(X)  # (n_clf, n_samples, n_classes), round 0
        round0_hard = np.argmax(beliefs0, axis=2)  # (n_clf, n_samples)
        unanimous = np.all(round0_hard == round0_hard[0:1, :], axis=0)  # (n_samples,)

        debate_final = self.debate(X)  # (n_samples, n_classes), full debate result
        debate_preds = self.classes_[np.argmax(debate_final, axis=1)]

        round0_preds = self.classes_[round0_hard[0]]  # any row works when unanimous

        final_preds = np.where(unanimous, round0_preds, debate_preds)
        return final_preds

    def fraction_gated(self, X):
        """Returns the fraction of samples that were unanimous at Round 0
        (and therefore skipped debate entirely)."""
        beliefs0 = self._beliefs(X)
        round0_hard = np.argmax(beliefs0, axis=2)
        unanimous = np.all(round0_hard == round0_hard[0:1, :], axis=0)
        return float(np.mean(unanimous))
