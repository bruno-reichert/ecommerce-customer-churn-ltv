import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    DATA_PROCESSED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)


def prepare_features_and_target(df_customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Separates the target from predictive features, explicitly dropping
    recency and timestamps from X to prevent target data leakage.
    Returns:
        X: Feature matrix
        y: Binary churn target (1 = churned, 0 = repurchased)
        timestamps: Series of first_purchase_timestamp used exclusively for chronological splitting
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
    """
    print("\n--- 1. Separating Features (X) & Target (y) ---")
    df = df_customers.copy()

    # Define the target
    y = df["is_churned"].copy()

    # Extract first purchase timestamp strictly for time-based splitting
    timestamps = pd.to_datetime(df["first_purchase_timestamp"]).copy()

    # Explicit list of numerical predictors (behavioral, logistics, spend)
    numerical_features = [
        "total_orders",
        "total_items_bought",
        "total_spend",
        "avg_order_value",
        "total_freight_paid",
        "avg_delivery_days",
        "max_delivery_delay",
        "has_experienced_late_delivery",
        "avg_freight_ratio",
        "avg_review_score",
        "has_negative_review",
        "avg_installments"
    ]

    # Categorical predictors
    categorical_features = [
        "customer_state",
        "favorite_category",
    ]

    X = df[numerical_features + categorical_features].copy()

    # Simplify favorite_category to top 15 categories + 'other' to prevent sparse explosion
    top_categories = X["favorite_category"].value_counts().nlargest(15).index
    X["favorite_category"] = X["favorite_category"].apply(lambda c: c if c in top_categories else "other")

    print(f"Total Customer Records: {len(X):,}")
    print(f"Numerical Features ({len(numerical_features)}): {', '.join(numerical_features[:4])}...")
    print(f"Categorical Features ({len(categorical_features)}): {', '.join(categorical_features)}")
    print(f"Target Distribution: {y.value_counts(normalize=True).to_dict()}")

    return X, y, timestamps, numerical_features, categorical_features


def build_preprocessor_pipeline(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Creates a robust Scikit-Learn ColumnTransformer:
    - Scales numerical features (StandardScaler)
    - One-hot encodes categorical features (OneHotEncoder)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


def split_data_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Performs a strict Time-Based Chronological Split based on customer arrival
    (first_purchase_timestamp).
    - Train: earliest arrivals (e.g. 70%)
    - Validation: middle arrivals (e.g. 15%) - for threshold & hyperparameter tuning
    - Test: most recent arrivals up to T0 (e.g. 15%) - strictly held-out evaluation
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0!"

    print(f"\n--- 2. Executing Chronological Split ({int(train_ratio*100)}% Train / {int(val_ratio*100)}% Val / {int(test_ratio*100)}% Test) ---")

    # Sort indices chronologically by customer first purchase
    sorted_indices = timestamps.sort_values().index

    n_total = len(sorted_indices)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_idx = sorted_indices[:train_end]
    val_idx = sorted_indices[train_end:val_end]
    test_idx = sorted_indices[val_end:]

    X_train, y_train = X.loc[train_idx].reset_index(drop=True), y.loc[train_idx].reset_index(drop=True)
    X_val, y_val = X.loc[val_idx].reset_index(drop=True), y.loc[val_idx].reset_index(drop=True)
    X_test, y_test = X.loc[test_idx].reset_index(drop=True), y.loc[test_idx].reset_index(drop=True)

    print(f"Training Set:   {len(X_train):,} samples (Dates: {timestamps.loc[train_idx].min().date()} to {timestamps.loc[train_idx].max().date()} | Churn: {y_train.mean()*100:.2f}%)")
    print(f"Validation Set: {len(X_val):,} samples (Dates: {timestamps.loc[val_idx].min().date()} to {timestamps.loc[val_idx].max().date()} | Churn: {y_val.mean()*100:.2f}%)")
    print(f"Testing Set:    {len(X_test):,} samples (Dates: {timestamps.loc[test_idx].min().date()} to {timestamps.loc[test_idx].max().date()} | Churn: {y_test.mean()*100:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    cust_path = DATA_PROCESSED / "customer_features.parquet"
    if not cust_path.exists():
        print(f"❌ File not found at {cust_path}!")
    else:
        print(f"Loading {cust_path}...")
        df_customers = pd.read_parquet(cust_path)
        X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronological(X, y, timestamps)