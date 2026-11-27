"""
Machine Learning Pipeline & Out-of-Time Data Splitting Module.
Evaluates the rare positive class: repurchased_in_window (1 = repeat buyer, 0 = churned).
Includes legitimate recency_days and customer_relationship_days.
Uses Snapshot 1 for Train/Val and Snapshot 2 for Test (Option 2).
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED, RANDOM_SEED


class TopCategoriesGrouper(BaseEstimator, TransformerMixin):
    """
    Learns top N most frequent categories from X_train only;
    maps all other categories to 'other'.
    """
    def __init__(self, top_n: int = 15):
        self.top_n = top_n
        self.top_categories_ = None

    def fit(self, X, y=None):
        series = pd.Series(np.array(X).ravel())
        self.top_categories_ = set(series.value_counts().nlargest(self.top_n).index)
        return self

    def transform(self, X):
        series = pd.Series(np.array(X).ravel())
        transformed = series.apply(lambda c: c if c in self.top_categories_ else "other")
        return transformed.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(["favorite_category"], dtype=object)
        return np.asarray(input_features, dtype=object)


def prepare_features_and_target(df_customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepares feature matrix X and target y.
    TARGET DEFINITION:
    - y = repurchased_in_window (1 = repurchased within 180 days, 0 = churned).
    - Positive class is the rare event (~1.18% - 1.63%).
    FEATURES:
    - Includes legitimate recency_days and customer_relationship_days (measured up to snapshot T0).
    - Excludes persona and cluster (descriptive only).
    """
    df = df_customers.copy()

    # Evaluation Target: 1 = Repurchased (Rare Event), 0 = Churned
    y = df["repurchased_in_window"].copy()

    # Predictor features
    numerical_features = [
        "recency_days",                  # Legitimate recency at snapshot T0
        "customer_relationship_days",    # Account maturity at snapshot T0
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

    categorical_features = [
        "customer_state",
        "favorite_category",
    ]

    X = df[numerical_features + categorical_features].copy()

    return X, y, numerical_features, categorical_features


def build_preprocessor_pipeline(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Creates an atomic Scikit-Learn ColumnTransformer:
    - Numerical: Median Imputer (fitted on X_train only) -> StandardScaler
    - Category: TopCategoriesGrouper (fitted on X_train only) -> OneHotEncoder
    - State: OneHotEncoder
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    category_pipeline = Pipeline([
        ("grouper", TopCategoriesGrouper(top_n=15)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    state_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat_group", category_pipeline, ["favorite_category"]),
            ("cat_state", state_pipeline, ["customer_state"])
        ],
        remainder="drop"
    )
    return preprocessor


def load_out_of_time_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str], List[str]]:
    """
    Option 2: Rolling Snapshots Out-of-Time Splitting.
    - Snapshot 1 (T0 = 2017-09-01): Partitioned into Train (80%) and Val (20%) via Stratification.
    - Snapshot 2 (T0 = 2018-03-01): Held-out Test Set (100% unseen future time horizon).
    """
    train_snap_path = DATA_PROCESSED / "customer_features_train.parquet"
    test_snap_path = DATA_PROCESSED / "customer_features_test.parquet"

    assert train_snap_path.exists() and test_snap_path.exists(), "Snapshot files missing!"

    df_train_snap = pd.read_parquet(train_snap_path)
    df_test_snap = pd.read_parquet(test_snap_path)

    # 1. Prepare Features & Target
    X_train_full, y_train_full, num_cols, cat_cols = prepare_features_and_target(df_train_snap)
    X_test, y_test, _, _ = prepare_features_and_target(df_test_snap)

    # 2. Stratified Train/Val Split within Snapshot 1
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20,
        stratify=y_train_full,
        random_state=RANDOM_SEED
    )

    print("\n--- Out-of-Time Rolling Snapshot Partitions ---")
    print(f"Train Set (Snapshot 1 - 80%): {len(X_train):,} samples (Repurchase Rate: {y_train.mean()*100:.2f}%)")
    print(f"Val Set   (Snapshot 1 - 20%): {len(X_val):,} samples (Repurchase Rate: {y_val.mean()*100:.2f}%)")
    print(f"Test Set  (Snapshot 2 - 100%): {len(X_test):,} samples (Repurchase Rate: {y_test.mean()*100:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols = load_out_of_time_splits()