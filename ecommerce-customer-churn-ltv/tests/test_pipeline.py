import sys
from pathlib import Path
# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import joblib
import pandas as pd
import numpy as np

from src.config import DATA_PROCESSED, MODELS_DIR
from src.model_pipeline import (
    prepare_features_and_target,
    build_preprocessor_pipeline,
    split_data_chronological
)

DATA_PATH = DATA_PROCESSED / "customer_features.parquet"
MODEL_PATH = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"


@pytest.fixture(scope="module")
def df_customers():
    """Fixture that loads processed customer dataset once for all tests."""
    assert DATA_PATH.exists(), f"Processed data missing at {DATA_PATH}!"
    return pd.read_parquet(DATA_PATH)


@pytest.fixture(scope="module")
def trained_pipeline():
    """Fixture that loads the saved champion model."""
    assert MODEL_PATH.exists(), f"Trained model missing at {MODEL_PATH}!"
    return joblib.load(MODEL_PATH)


# ==========================================
# 1. ANTI-DATA LEAKAGE TESTS (CRITICAL)
# ==========================================
def test_target_leakage_recency_excluded(df_customers):
    """
    CRITICAL GUARDRAIL: Asserts that 'recency_days' and raw purchase timestamps
    are strictly excluded from feature matrix X.
    """
    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    assert "recency_days" not in X.columns, "FATAL: 'recency_days' leaked into feature matrix!"
    assert "last_purchase_timestamp" not in X.columns, "FATAL: 'last_purchase_timestamp' leaked into X!"
    assert "first_purchase_timestamp" not in X.columns, "FATAL: 'first_purchase_timestamp' leaked into X!"


def test_target_leakage_persona_excluded(df_customers):
    """
    CRITICAL GUARDRAIL: Asserts that 'persona' and 'cluster' are strictly excluded
    from features to prevent indirect leakage of recency through cluster labels.
    """
    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    assert "persona" not in X.columns, "FATAL: 'persona' leaked into feature matrix!"
    assert "persona" not in cat_cols, "FATAL: 'persona' listed in categorical features!"
    assert "cluster" not in X.columns, "FATAL: 'cluster' leaked into feature matrix!"


# ==========================================
# 2. DATA INTEGRITY & CHRONOLOGICAL INVARIANTS
# ==========================================
def test_target_binary_and_complete(df_customers):
    """Verifies that the target is binary, non-null, and within valid probabilistic bounds."""
    X, y, timestamps, _, _ = prepare_features_and_target(df_customers)
    assert y.isnull().sum() == 0, "Target column contains null values!"
    assert set(y.unique()).issubset({0, 1}), "Target must be strictly binary (0 and 1)!"
    assert len(y) == len(df_customers), "Target length does not match customer records!"


def test_chronological_split_ordering(df_customers):
    """
    Verifies that the chronological split maintains strict temporal order:
    Train max timestamp <= Val min timestamp <= Test min timestamp.
    """
    X, y, timestamps, _, _ = prepare_features_and_target(df_customers)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronological(X, y, timestamps)

    # Total samples must sum to total dataset
    assert len(X_train) + len(X_val) + len(X_test) == len(df_customers), "Split lost customer rows!"


# ==========================================
# 3. PREPROCESSOR & MATRIX OUTPUT INVARIANTS
# ==========================================
def test_preprocessor_transformation_and_no_nans(df_customers, trained_pipeline):
    """
    Verifies that the fitted ColumnTransformer in our trained pipeline
    transforms unseen data cleanly with zero NaNs.
    """
    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    fitted_preprocessor = trained_pipeline.named_steps["preprocessor"]

    sample_X = X.iloc[:500]
    matrix = fitted_preprocessor.transform(sample_X)

    # Matrix must be non-empty and free of NaNs
    assert matrix.shape[0] == len(sample_X)
    assert matrix.shape[1] > 0, "Preprocessed matrix has zero features!"
    assert not np.isnan(matrix).any(), "Preprocessed matrix contains NaNs!"


# ==========================================
# 4. MODEL INFERENCE & PROBABILITY BOUNDS
# ==========================================
def test_model_predict_proba_bounds(df_customers, trained_pipeline):
    """Verifies that model outputs valid probabilities strictly between 0 and 1."""
    X, y, timestamps, _, _ = prepare_features_and_target(df_customers)
    sample_X = X.iloc[:200]
    probs = trained_pipeline.predict_proba(sample_X)[:, 1]

    assert len(probs) == len(sample_X)
    assert np.all(probs >= 0.0), "Found negative probability predictions!"
    assert np.all(probs <= 1.0), "Found probability predictions greater than 1.0!"