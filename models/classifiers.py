from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def get_classifiers():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ),
        "svm": SVC(
            probability=True,
            random_state=42
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5
        )
    }