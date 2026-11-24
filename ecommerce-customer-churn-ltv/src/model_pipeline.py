import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


class TopCategoriesGrouper(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn Transformer that learns the top N most frequent categories
    from the training set and groups all other/unseen categories into 'other'.
    Guarantees zero target leakage or snooping into validation/test splits.
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
        """Passes through the input feature name so ColumnTransformer can label columns cleanly."""
        if input_features is None:
            return np.array(["favorite_category"], dtype=object)
        return np.asarray(input_features, dtype=object)


def prepare_features_and_target(df_customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    """
    Separates the target from predictive features, explicitly dropping
    recency and timestamps from X to prevent target data leakage.
    Leaves raw categories untouched so the Sklearn pipeline can fit on X_train.
    """
    print("\n--- 1. Separating Features (X) & Target (y) ---")
    df = df_customers.copy()

    # Define the target
    y = df["is_churned"].copy()

    # Extract first purchase timestamp strictly for chronological splitting
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

    # Do NOT pre-group favorite_category here; the pipeline will fit it on X_train only!
    X = df[numerical_features + categorical_features].copy()

    print(f"Total Customer Records: {len(X):,}")
    print(f"Numerical Features ({len(numerical_features)}): {', '.join(numerical_features[:4])}...")
    print(f"Categorical Features ({len(categorical_features)}): {', '.join(categorical_features)}")
    print(f"Target Distribution: {y.value_counts(normalize=True).to_dict()}")

    return X, y, timestamps, numerical_features, categorical_features


def build_preprocessor_pipeline(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Creates a hermetic Scikit-Learn ColumnTransformer:
    - Numerical: Median Imputation -> StandardScaler (fitted on X_train only)
    - State: OneHotEncoder (ignore unknowns)
    - Category: TopCategoriesGrouper (fitted on X_train only) -> OneHotEncoder
    """
    # 1. Numerical Sub-Pipeline: Median Imputation -> Scaling
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Favorite Category Sub-Pipeline: Top 15 Grouper -> OneHotEncoder
    category_pipeline = Pipeline([
        ("grouper", TopCategoriesGrouper(top_n=15)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 3. State Sub-Pipeline: Direct OneHotEncoder
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


def split_data_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Performs a strict Time-Based Chronological Split based on customer arrival
    (first_purchase_timestamp).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0!"

    print(f"\n--- 2. Executing Chronological Split ({int(train_ratio*100)}% Train / {int(val_ratio*100)}% Val / {int(test_ratio*100)}% Test) ---")

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

        # Quick validation of the new preprocessor
        print("\n--- Testing Pipeline Fit on X_train only ---")
        preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        print(f"✅ X_train transformed shape: {X_train_trans.shape}")
        print(f"✅ X_test transformed shape:  {X_test_trans.shape}")
        print("✅ Pipeline successfully fit on X_train and applied to X_test with zero leakage!")