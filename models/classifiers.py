from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def get_classifiers():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42))
        ]),
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
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ])
    }