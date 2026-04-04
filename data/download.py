import os
import pandas as pd
from sklearn.datasets import fetch_openml
from dotenv import load_dotenv

load_dotenv()

DATASETS = {
    "creditcard": {
        "openml_id": 1597,
        "name": "Credit Card Fraud",
        "target": "Class",
        "imbalance_ratio": 0.17,
        "minority_class": "Fraud"
    },
    "phoneme": {
        "openml_id": 1489,
        "name": "Phoneme",
        "target": "Class",
        "imbalance_ratio": 9.1,
        "minority_class": "Nasal"
    },
    "mammography": {
        "openml_id": 310,
        "name": "Mammography",
        "target": "Class",
        "imbalance_ratio": 2.3,
        "minority_class": "Malignant"
    }
}

def download_all():
    os.makedirs("data/raw", exist_ok=True)

    for key, info in DATASETS.items():
        save_path = f"data/raw/{key}.csv"

        if os.path.exists(save_path):
            print(f"{info['name']} already downloaded, skipping...")
            continue

        print(f"Downloading {info['name']}...")
        dataset = fetch_openml(data_id=info["openml_id"], as_frame=True, parser="auto")
        df = dataset.frame
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()

    print("All datasets downloaded!")

if __name__ == "__main__":
    download_all()