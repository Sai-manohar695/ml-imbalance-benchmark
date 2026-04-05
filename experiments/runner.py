import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from models.classifiers import get_classifiers
from sampling.strategies import get_sampling_strategies, apply_sampling
from db.connection import get_connection

load_dotenv()

DATASETS = {
    "creditcard": "Credit Card Fraud",
    "phoneme": "Phoneme",
    "mammography": "Mammography"
}

def load_dataset(key):
    df = pd.read_csv(f"data/raw/{key}.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        # Remap any label set to 0, 1 (handles [-1,1] and [1,2] cases)
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X.values, y

def get_dataset_id(conn, name):
    cur = conn.cursor()
    cur.execute("SELECT id FROM datasets WHERE name = %s", (name,))
    result = cur.fetchone()
    cur.close()
    return result[0] if result else None

def save_experiment(conn, dataset_id, classifier_name, sampling_name):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO experiments (dataset_id, classifier, sampling_strategy)
        VALUES (%s, %s, %s) RETURNING id
    """, (dataset_id, classifier_name, sampling_name))
    experiment_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return experiment_id

def save_results(conn, experiment_id, fold, metrics):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results (experiment_id, fold, f1, auc_roc, mcc, precision, recall)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        experiment_id,
        fold,
        metrics["f1"],
        metrics["auc_roc"],
        metrics["mcc"],
        metrics["precision"],
        metrics["recall"]
    ))
    conn.commit()
    cur.close()

def run_experiments():
    classifiers = get_classifiers()
    sampling_strategies = get_sampling_strategies()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for dataset_key, dataset_name in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        X, y = load_dataset(dataset_key)
        print(f"  Loaded {len(X)} rows")

        conn = get_connection()
        dataset_id = get_dataset_id(conn, dataset_name)
        conn.close()

        for clf_name, clf in classifiers.items():
            for sampling_name, sampling in sampling_strategies.items():
                print(f"  Running: {clf_name} + {sampling_name}...")

                fold_metrics = []

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    try:
                        X_train, y_train = apply_sampling(sampling, X_train, y_train)
                    except Exception as e:
                        print(f"    Sampling failed on fold {fold}: {e}, skipping...")
                        continue

                    try:
                        clf.fit(X_train, y_train)
                    except Exception as e:
                        print(f"    Training failed on fold {fold}: {e}, skipping...")
                        continue

                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1]

                    metrics = {
                        "f1": f1_score(y_test, y_pred, zero_division=0),
                        "auc_roc": roc_auc_score(y_test, y_prob),
                        "mcc": matthews_corrcoef(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0)
                    }

                    fold_metrics.append(metrics)

                if fold_metrics:
                    conn = get_connection()
                    experiment_id = save_experiment(conn, dataset_id, clf_name, sampling_name)
                    for fold, metrics in enumerate(fold_metrics):
                        save_results(conn, experiment_id, fold, metrics)
                    conn.close()

                    avg_f1 = np.mean([m["f1"] for m in fold_metrics])
                    avg_auc = np.mean([m["auc_roc"] for m in fold_metrics])
                    avg_mcc = np.mean([m["mcc"] for m in fold_metrics])
                    print(f"    Done — avg F1: {avg_f1:.4f} | avg AUC-ROC: {avg_auc:.4f} | avg MCC: {avg_mcc:.4f}")

    print("\nAll experiments complete!")

if __name__ == "__main__":
    run_experiments()