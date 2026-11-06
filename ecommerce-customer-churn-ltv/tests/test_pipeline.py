import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import joblib
import pandas as pd
import numpy as np

from src.model_pipeline import prepare_features_and_target, build_preprocessor_pipeline, split_data

DATA_PATH = Path("data/processed/customer_features.parquet")
MODEL_PATH = Path("models/best_xgboost_churn_pipeline.joblib")


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
    CRITICAL GUARDRAIL: Asserts that 'recency_days' and any purchase timestamp
    are strictly excluded from feature matrix X.
    """
    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    
    assert "recency_days" not in X.columns, "FATAL: 'recency_days' leaked into feature matrix!"
    assert "last_purchase_timestamp" not in X.columns, "FATAL: 'last_purchase_timestamp' leaked into X!"
    assert "first_purchase_timestamp" not in X.columns, "FATAL: 'first_purchase_timestamp' leaked into X!"


def test_target_leakage_persona_excluded(df_customers):
    """
    CRITICAL GUARDRAIL: Asserts that 'persona' is strictly excluded from features
    to prevent indirect leakage of recency through cluster labels.
    """
    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    
    assert "persona" not in X.columns, "FATAL: 'persona' leaked into feature matrix!"
    assert "persona" not in cat_cols, "FATAL: 'persona' listed in categorical features!"


# ==========================================
# 2. DATA INTEGRITY & SANITY INVARIANTS
# ==========================================
def test_target_balance_and_types(df_customers):
    """Verifies that the target is binary and has no null values."""
    X, y, _, _ = prepare_features_and_target(df_customers)
    
    assert y.isnull().sum() == 0, "Target column contains null values!"
    assert set(y.unique()) == {0, 1}, "Target must be strictly binary (0 and 1)!"
    # Ensure churn rate is in realistic e-commerce range (~55% to 65%)
    churn_rate = y.mean()
    assert 0.50 <= churn_rate <= 0.70, f"Unexpected churn rate: {churn_rate:.2%}"


def test_stratified_split_preserves_churn_distribution(df_customers):
    """Verifies that train and test sets have matching churn rates within 0.1%."""
    X, y, _, _ = prepare_features_and_target(df_customers)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20)

    train_churn = y_train.mean()
    test_churn = y_test.mean()

    assert abs(train_churn - test_churn) < 0.001, (
        f"Stratification failed! Train: {train_churn:.4f}, Test: {test_churn:.4f}"
    )


# ==========================================
# 3. PREPROCESSOR & MATRIX EXPANSION TESTS
# ==========================================
def test_preprocessor_shape_and_no_nans(df_customers, trained_pipeline):
    """
    Verifies that the fitted ColumnTransformer in our trained pipeline
    expands features to exactly 55 columns and outputs zero NaNs.
    """
    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    
    # Extract the fitted preprocessor from our trained pipeline
    fitted_preprocessor = trained_pipeline.named_steps["preprocessor"]
    
    # Transform a test sample
    sample_X = X.iloc[:500]
    matrix = fitted_preprocessor.transform(sample_X)
    
    # Assert exact 55 columns (12 numericals + 27 states + 16 categories)
    assert matrix.shape[1] == 55, f"Expected 55 columns, got {matrix.shape[1]}"
    assert not np.isnan(matrix).any(), "Preprocessed matrix contains NaNs!"


# ==========================================
# 4. MODEL INFERENCE & PROBABILITY BOUNDS
# ==========================================
def test_model_predict_proba_bounds(df_customers, trained_pipeline):
    """Verifies that model outputs valid probabilities strictly between 0 and 1."""
    X, y, _, _ = prepare_features_and_target(df_customers)
    sample_X = X.iloc[:200]
    
    probs = trained_pipeline.predict_proba(sample_X)[:, 1]
    
    assert len(probs) == len(sample_X)
    assert np.all(probs >= 0.0), "Found negative probability predictions!"
    assert np.all(probs <= 1.0), "Found probability predictions greater than 1.0!"