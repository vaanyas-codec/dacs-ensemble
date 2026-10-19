"""
Standard ensemble baselines: majority/soft voting, boosting, and stacking,
built on the same four base classifiers used by the Consensus/Debate
Ensemble, so comparisons are apples-to-apples.
"""

from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def get_base_estimators(random_state=42):
    """Returns the four base classifiers used throughout the project."""
    return [
        ("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=random_state)),
        ("naive_bayes", GaussianNB()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("logistic_regression", LogisticRegression(max_iter=1000)),
    ]


def build_voting(base_estimators):
    return VotingClassifier(estimators=base_estimators, voting="soft")


def build_boosting(random_state=42):
    # AdaBoost needs a single weak learner; a shallow decision tree is standard.
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=random_state,
    )


def build_stacking(base_estimators, random_state=42):
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
