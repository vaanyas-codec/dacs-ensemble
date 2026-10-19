from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


def get_candidate_pool(random_state=42):
    return {
        "DT_d1": DecisionTreeClassifier(max_depth=1, random_state=random_state),
        "DT_d3": DecisionTreeClassifier(max_depth=3, random_state=random_state),
        "DT_d10": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "NB": GaussianNB(),
        "kNN_1": KNeighborsClassifier(n_neighbors=1),
        "kNN_11": KNeighborsClassifier(n_neighbors=11),
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM_cal": CalibratedClassifierCV(SVC(kernel="rbf", random_state=random_state), cv=3),
        "RF_10": RandomForestClassifier(n_estimators=10, random_state=random_state),
        "LDA": LinearDiscriminantAnalysis(),
        "ExtraTrees_10": ExtraTreesClassifier(n_estimators=10, random_state=random_state),
        "GB_shallow": GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=random_state),
        "MLP_small": MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=random_state),
    }
