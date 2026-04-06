import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from db.connection import get_connection
from itertools import combinations

def fetch_results():
    import os
    from sqlalchemy import create_engine
    engine = create_engine(os.getenv("DATABASE_URL"))
    query = """
        SELECT 
            d.name as dataset,
            e.classifier,
            e.sampling_strategy,
            r.fold,
            r.f1,
            r.auc_roc,
            r.mcc,
            r.precision,
            r.recall
        FROM results r
        JOIN experiments e ON r.experiment_id = e.id
        JOIN datasets d ON e.dataset_id = d.id
        ORDER BY d.name, e.classifier, e.sampling_strategy, r.fold
    """
    df = pd.read_sql(query, engine)
    return df

def run_wilcoxon(df, metric="f1"):
    """Pairwise Wilcoxon signed-rank test between classifiers."""
    print(f"\n{'='*60}")
    print(f"Wilcoxon Signed-Rank Test — {metric.upper()}")
    print(f"{'='*60}")

    results = []
    classifiers = df["classifier"].unique()

    for dataset in df["dataset"].unique():
        print(f"\nDataset: {dataset}")
        df_ds = df[df["dataset"] == dataset]

        for clf_a, clf_b in combinations(classifiers, 2):
            scores_a = df_ds[df_ds["classifier"] == clf_a][metric].values
            scores_b = df_ds[df_ds["classifier"] == clf_b][metric].values

            min_len = min(len(scores_a), len(scores_b))
            if min_len < 2:
                continue

            scores_a = scores_a[:min_len]
            scores_b = scores_b[:min_len]

            if np.all(scores_a == scores_b):
                continue

            try:
                stat, p = wilcoxon(scores_a, scores_b)
                significant = p < 0.05
                results.append({
                    "dataset": dataset,
                    "classifier_a": clf_a,
                    "classifier_b": clf_b,
                    "metric": metric,
                    "statistic": stat,
                    "p_value": p,
                    "is_significant": significant
                })
                sig = "✓ significant" if significant else "✗ not significant"
                print(f"  {clf_a} vs {clf_b}: p={p:.4f} {sig}")
            except Exception as e:
                print(f"  {clf_a} vs {clf_b}: skipped ({e})")

    return pd.DataFrame(results)

def run_friedman(df, metric="f1"):
    """Friedman test across all classifiers."""
    print(f"\n{'='*60}")
    print(f"Friedman Test — {metric.upper()}")
    print(f"{'='*60}")

    results = []
    classifiers = df["classifier"].unique()

    for dataset in df["dataset"].unique():
        print(f"\nDataset: {dataset}")
        df_ds = df[df["dataset"] == dataset]

        groups = []
        for clf in classifiers:
            scores = df_ds[df_ds["classifier"] == clf][metric].values
            groups.append(scores)

        min_len = min(len(g) for g in groups)
        groups = [g[:min_len] for g in groups]

        if min_len < 2:
            print("  Not enough data, skipping...")
            continue

        try:
            stat, p = friedmanchisquare(*groups)
            significant = p < 0.05
            results.append({
                "dataset": dataset,
                "metric": metric,
                "statistic": stat,
                "p_value": p,
                "is_significant": significant
            })
            sig = "✓ significant" if significant else "✗ not significant"
            print(f"  Friedman stat={stat:.4f}, p={p:.4f} {sig}")
        except Exception as e:
            print(f"  Skipped: {e}")

    return pd.DataFrame(results)

def save_wilcoxon_results(results_df):
    """Save Wilcoxon results to database."""
    if results_df.empty:
        return

    conn = get_connection()
    cur = conn.cursor()

    for _, row in results_df.iterrows():
        cur.execute("""
            INSERT INTO statistical_tests 
            (test_name, metric, classifier_a, classifier_b, p_value, statistic, is_significant)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            "wilcoxon",
            row["metric"],
            row["classifier_a"],
            row["classifier_b"],
            row["p_value"],
            row["statistic"],
            bool(row["is_significant"])
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nSaved {len(results_df)} Wilcoxon results to database.")

def run_all_tests():
    print("Fetching results from database...")
    df = fetch_results()
    print(f"Loaded {len(df)} result rows.")

    all_wilcoxon = []
    for metric in ["f1", "auc_roc", "mcc"]:
        w_results = run_wilcoxon(df, metric=metric)
        all_wilcoxon.append(w_results)
        run_friedman(df, metric=metric)

    combined = pd.concat(all_wilcoxon, ignore_index=True)
    save_wilcoxon_results(combined)

    print("\n\nAll statistical tests complete!")

if __name__ == "__main__":
    run_all_tests()