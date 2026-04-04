import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from db.connection import get_connection

load_dotenv()

DATASETS = {
    "creditcard": {
        "name": "Credit Card Fraud",
        "imbalance_ratio": 0.17,
        "minority_class": "Fraud"
    },
    "phoneme": {
        "name": "Phoneme",
        "imbalance_ratio": 9.1,
        "minority_class": "Nasal"
    },
    "mammography": {
        "name": "Mammography",
        "imbalance_ratio": 2.3,
        "minority_class": "Malignant"
    }
}

def seed_datasets():
    conn = get_connection()
    cur = conn.cursor()

    for key, info in DATASETS.items():
        path = f"data/raw/{key}.csv"
        df = pd.read_csv(path)

        n_samples, n_features = df.shape
        n_features -= 1  # exclude target column

        # Check if already seeded
        cur.execute("SELECT id FROM datasets WHERE name = %s", (info["name"],))
        if cur.fetchone():
            print(f"{info['name']} already in database, skipping...")
            continue

        cur.execute("""
            INSERT INTO datasets (name, n_samples, n_features, imbalance_ratio, minority_class)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            info["name"],
            n_samples,
            n_features,
            info["imbalance_ratio"],
            info["minority_class"]
        ))

        print(f"Seeded {info['name']} — {n_samples} samples, {n_features} features")

    conn.commit()
    cur.close()
    conn.close()
    print("\nAll datasets seeded!")

if __name__ == "__main__":
    seed_datasets()