from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def get_sampling_strategies():
    return {
        "none": None,
        "smote": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42),
        "random_undersampling": RandomUnderSampler(random_state=42)
    }

def apply_sampling(strategy, X_train, y_train):
    if strategy is None:
        return X_train, y_train
    return strategy.fit_resample(X_train, y_train)