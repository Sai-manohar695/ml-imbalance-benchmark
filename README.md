# ML Imbalance Benchmark

**Python · PostgreSQL (Neon) · Scikit-learn · XGBoost · imbalanced-learn · SciPy · Streamlit · Docker**

Most ML tutorials treat SMOTE as a default fix for class imbalance and move on. This project tests that assumption — 4 classifiers crossed with 4 sampling strategies, run on 3 public datasets with imbalance ratios ranging from 0.17% to 16%. Every result goes through Wilcoxon signed-rank and Friedman tests before any claims are made.

The finding: classifier choice matters more than sampling strategy, and random undersampling looks great on AUC-ROC while silently failing on F1 and MCC.

---

## Results Summary

| Dataset | Best Classifier | Best Strategy | Best F1 |
|---|---|---|---|
| Credit Card Fraud (0.17%) | Random Forest | None | 0.8588 |
| Mammography (2.3%) | XGBoost | None | 0.6890 |
| Phoneme (9.1%) | Random Forest | None | 0.9365 |

**Key finding:** Random undersampling consistently scores high on AUC-ROC while F1 and MCC collapse — across all classifiers and all datasets. If you only report AUC-ROC, you'd conclude it works. The other metrics tell a different story.

---

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop
- A free [Neon](https://neon.tech) PostgreSQL account

### 1. Clone the repo
```bash
git clone https://github.com/Sai-manohar695/ml-imbalance-benchmark
cd ml-imbalance-benchmark
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```

Add your Neon connection string to `.env`:
DATABASE_URL=postgresql://user:password@host/dbname

### 5. Create database tables
```bash
python db/connection.py
```

### 6. Download datasets
```bash
python data/download.py
```

### 7. Seed dataset metadata
```bash
python -m db.seed
```

### 8. Run experiments
```bash
python -m experiments.runner
```

This runs 4 classifiers × 4 sampling strategies × 3 datasets × 5 folds = **240 runs**. Results are saved to Neon automatically.

### 9. Run statistical tests
```bash
python -m stats.tests
```

### 10. Launch dashboard
```bash
streamlit run dashboard/app.py
```

Or with Docker:
```bash
docker compose up --build
```

Dashboard runs at `http://localhost:8501`.

---

## Project Structure
ml-imbalance-benchmark/
├── data/
│   └── download.py         # Downloads datasets from OpenML
├── db/
│   ├── connection.py       # PostgreSQL connection + schema
│   └── seed.py             # Seeds dataset metadata
├── experiments/
│   └── runner.py           # Main experiment loop
├── models/
│   └── classifiers.py      # Classifier definitions
├── sampling/
│   └── strategies.py       # Sampling strategy definitions
├── stats/
│   └── tests.py            # Wilcoxon + Friedman tests
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

---

## Datasets

| Dataset | Samples | Imbalance Ratio | Source |
|---|---|---|---|
| Credit Card Fraud | 284,807 | 0.17% | OpenML #1597 |
| Mammography | 11,183 | 2.3% | OpenML #310 |
| Phoneme | 5,404 | 9.1% | OpenML #1489 |

All datasets are downloaded automatically from OpenML on first run.

---

## Classifiers

- Logistic Regression (with StandardScaler)
- Random Forest
- XGBoost
- K-Nearest Neighbors (with StandardScaler)

## Sampling Strategies

- None (baseline)
- SMOTE
- ADASYN
- Random Undersampling

---

## Statistical Tests

All results are validated with:
- **Wilcoxon signed-rank test** — pairwise classifier comparisons per dataset
- **Friedman test** — global ranking significance across all classifiers

Significance threshold: p < 0.05. Results stored in the `statistical_tests` table in Neon.

---

## Dashboard

Four tabs:
- **Metric Comparison** — bar charts across classifiers and strategies
- **Heatmap** — color grid of metric scores per combination
- **Metric Gap Analysis** — AUC-ROC vs F1 gap, the core finding visualized
- **Statistical Tests** — Wilcoxon p-values per classifier pair

---

## What I'd Do Differently

- Add SMOTE variants (Borderline-SMOTE, SVM-SMOTE)
- Test on datasets with imbalance below 0.1%
- Set up experiment logging from day one instead of retrofitting it

---

## Author

**D.P. Sai Manohar**
[GitHub](https://github.com/Sai-manohar695) · [LinkedIn](https://linkedin.com/in/sai-manohar23) · [Portfolio](https://sai-manohar695.github.io)