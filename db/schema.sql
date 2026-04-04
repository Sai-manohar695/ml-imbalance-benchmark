CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    n_samples INTEGER,
    n_features INTEGER,
    imbalance_ratio FLOAT,
    minority_class VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    classifier VARCHAR(100) NOT NULL,
    sampling_strategy VARCHAR(100) NOT NULL,
    random_state INTEGER DEFAULT 42,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    fold INTEGER,
    f1 FLOAT,
    auc_roc FLOAT,
    mcc FLOAT,
    precision FLOAT,
    recall FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS statistical_tests (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100),
    metric VARCHAR(50),
    classifier_a VARCHAR(100),
    classifier_b VARCHAR(100),
    p_value FLOAT,
    statistic FLOAT,
    is_significant BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);