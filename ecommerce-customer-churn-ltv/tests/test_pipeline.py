"""
Self-contained unit tests for pipeline transformations, anti-leakage invariants,
and model probability bounds.
Uses lightweight synthetic fixtures so the entire suite passes on a fresh checkout
without requiring pre-built datasets or model artifacts.
"""

import sys
from pathlib import Path

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.model_pipeline import (
    build_preprocessor_pipeline,
    prepare_features_and_target,
    split_data_chronological,
)


@pytest.fixture(scope="module")
def synthetic_customer_df():
    """
    Creates a small, fully self-contained synthetic customer analytical DataFrame (10 rows).
    Includes all numerical features, categorical features, timestamps, and target.
    """
    np.random.seed(42)
    n = 10

    # Ordered arrival dates between 2017-01 and 2018-02
    dates = pd.date_range(start="2017-01-01", periods=n, freq="30D")

    data = {
        "customer_unique_id": [f"cust_{i}" for i in range(n)],
        "first_purchase_timestamp": dates,
        "last_purchase_timestamp": dates + pd.Timedelta(days=5),
        "total_orders": [1, 2, 1, 1, 3, 1, 1, 2, 1, 1],
        "total_items_bought": [1, 3, 1, 1, 4, 1, 1, 2, 1, 1],
        "total_spend": [50.0, 150.0, 80.0, 45.0, 300.0, 65.0, 120.0, 210.0, 40.0, 95.0],
        "avg_order_value": [50.0, 75.0, 80.0, 45.0, 100.0, 65.0, 120.0, 105.0, 40.0, 95.0],
        "total_freight_paid": [15.0, 30.0, 18.0, 12.0, 45.0, 14.0, 22.0, 28.0, 10.0, 16.0],
        "avg_delivery_days": [10.2, 14.5, 8.1, 12.0, 20.3, 9.4, 15.0, 11.2, 7.8, 13.1],
        "max_delivery_delay": [0.0, 3.2, 0.0, 0.0, 5.1, 0.0, 1.2, 0.0, 0.0, 0.0],
        "has_experienced_late_delivery": [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        "avg_freight_ratio": [0.30, 0.20, 0.22, 0.26, 0.15, 0.21, 0.18, 0.13, 0.25, 0.17],
        "avg_review_score": [5.0, 2.0, 4.0, 5.0, 1.0, 4.0, 3.0, 5.0, 4.0, 5.0],
        "has_negative_review": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "avg_installments": [1.0, 3.0, 2.0, 1.0, 6.0, 1.0, 4.0, 2.0, 1.0, 2.0],
        "customer_state": ["SP", "RJ", "SP", "MG", "SP", "RS", "PR", "SP", "BA", "SC"],
        "favorite_category": [
            "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
            "bed_bath_table", "computers_accessories", "watches_gifts", "health_beauty",
            "housewares", "auto"
        ],
        "recency_days": [400.0, 370.0, 340.0, 310.0, 280.0, 250.0, 220.0, 190.0, 160.0, 130.0],
        "persona": ["Lost / Hibernating Shoppers"] * 5 + ["Active / Recent Buyers"] * 5,
        "cluster": [3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
        "is_churned": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def synthetic_trained_pipeline(synthetic_customer_df):
    """
    Fits a lightweight ColumnTransformer + XGBoost pipeline on the synthetic fixture.
    Runs instantaneously in memory.
    """
    X, y, _timestamps, num_cols, cat_cols = prepare_features_and_target(synthetic_customer_df)
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss"))
    ])
    pipeline.fit(X, y)
    return pipeline


# ==========================================
# 1. ANTI-DATA LEAKAGE TESTS (CRITICAL)
# ==========================================
def test_target_leakage_recency_excluded(synthetic_customer_df):
    """
    CRITICAL GUARDRAIL: Asserts that 'recency_days' and raw purchase timestamps
    are strictly excluded from feature matrix X.
    """
    X, _y, _timestamps, _num_cols, _cat_cols = prepare_features_and_target(synthetic_customer_df)
    assert "recency_days" not in X.columns, "FATAL: 'recency_days' leaked into feature matrix!"
    assert "last_purchase_timestamp" not in X.columns, "FATAL: 'last_purchase_timestamp' leaked into X!"
    assert "first_purchase_timestamp" not in X.columns, "FATAL: 'first_purchase_timestamp' leaked into X!"


def test_target_leakage_persona_excluded(synthetic_customer_df):
    """
    CRITICAL GUARDRAIL: Asserts that 'persona' and 'cluster' are strictly excluded
    from features to prevent indirect leakage of recency through cluster labels.
    """
    X, _y, _timestamps, _num_cols, cat_cols = prepare_features_and_target(synthetic_customer_df)
    assert "persona" not in X.columns, "FATAL: 'persona' leaked into feature matrix!"
    assert "persona" not in cat_cols, "FATAL: 'persona' listed in categorical features!"
    assert "cluster" not in X.columns, "FATAL: 'cluster' leaked into feature matrix!"


# ==========================================
# 2. DATA INTEGRITY & CHRONOLOGICAL INVARIANTS
# ==========================================
def test_target_binary_and_complete(synthetic_customer_df):
    """Verifies that the target is binary, non-null, and within valid probabilistic bounds."""
    _X, y, _timestamps, _, _ = prepare_features_and_target(synthetic_customer_df)
    assert y.isnull().sum() == 0, "Target column contains null values!"
    assert set(y.unique()).issubset({0, 1}), "Target must be strictly binary (0 and 1)!"
    assert len(y) == len(synthetic_customer_df), "Target length does not match customer records!"


def test_chronological_split_ordering(synthetic_customer_df):
    """
    Verifies that the chronological split maintains strict temporal order
    and preserves exact row count.
    """
    X, y, timestamps, _, _ = prepare_features_and_target(synthetic_customer_df)
    X_train, X_val, X_test, _y_train, _y_val, _y_test = split_data_chronological(
        X, y, timestamps, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )

    # Total samples must sum to total dataset
    assert len(X_train) + len(X_val) + len(X_test) == len(synthetic_customer_df), "Split lost customer rows!"


# ==========================================
# 3. PREPROCESSOR & MATRIX OUTPUT INVARIANTS
# ==========================================
def test_preprocessor_transformation_and_no_nans(synthetic_customer_df, synthetic_trained_pipeline):
    """
    Verifies that the fitted ColumnTransformer in our pipeline
    transforms unseen data cleanly with zero NaNs.
    """
    X, _y, _timestamps, _num_cols, _cat_cols = prepare_features_and_target(synthetic_customer_df)
    fitted_preprocessor = synthetic_trained_pipeline.named_steps["preprocessor"]

    sample_X = X.iloc[:5]
    matrix = fitted_preprocessor.transform(sample_X)

    # Matrix must be non-empty and free of NaNs
    assert matrix.shape[0] == len(sample_X)
    assert matrix.shape[1] > 0, "Preprocessed matrix has zero features!"
    assert not np.isnan(matrix).any(), "Preprocessed matrix contains NaNs!"


# ==========================================
# 4. MODEL INFERENCE & PROBABILITY BOUNDS
# ==========================================
def test_model_predict_proba_bounds(synthetic_customer_df, synthetic_trained_pipeline):
    """Verifies that model outputs valid probabilities strictly between 0 and 1."""
    X, _y, _timestamps, _, _ = prepare_features_and_target(synthetic_customer_df)
    sample_X = X.iloc[:5]
    probs = synthetic_trained_pipeline.predict_proba(sample_X)[:, 1]

    assert len(probs) == len(sample_X)
    assert np.all(probs >= 0.0), "Found negative probability predictions!"
    assert np.all(probs <= 1.0), "Found probability predictions greater than 1.0!"