import sys
from pathlib import Path
from typing import Tuple, List
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_features_and_target(df_customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separates the target from predictive features, explicitly dropping
    recency and timestamps to prevent target data leakage.
    """
    print("\n--- 1. Separating Features (X) & Target (y) ---")

    # Define the target
    y = df_customers["is_churned"].copy()

    # Explicit list of numerical predictors
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

    X = df_customers[numerical_features + categorical_features].copy()

    # Simplify favorite_category to top 15 categories + 'other' to prevent sparse explosion
    top_categories = X["favorite_category"].value_counts().nlargest(15).index
    X["favorite_category"] = X["favorite_category"].apply(lambda c: c if c in top_categories else "other")

    print(f"Total Customer Records: {len(X):,}")
    print(f"Numerical Features ({len(numerical_features)}): {', '.join(numerical_features[:4])}...")
    print(f"Categorical Features ({len(categorical_features)}): {', '.join(categorical_features)}")
    print(f"Target Distribution: {y.value_counts(normalize=True).to_dict()}")

    return X, y, numerical_features, categorical_features


def build_preprocessor_pipeline(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Creates a robust Scikit-Learn ColumnTransformer:
    - Scales numerical features (StandardScaler)
    - One-hot encodes categorical features (OneHotEncoder)
    """
    print("\n--- 2. Building ColumnTransformer Preprocessing Pipeline ---")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs an 80/20 Stratified Train/Test Split.
    Stratification ensures both sets have the exact same 59.16% churn ratio.
    """
    print(f"\n--- 3. Executing Stratified Train/Test Split ({int((1-test_size)*100)}/{int(test_size*100)}) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    print(f"Training Set:   {len(X_train):,} samples (Churn Rate: {y_train.mean()*100:.2f}%)")
    print(f"Testing Set:    {len(X_test):,} samples (Churn Rate: {y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


def main():
    customers_path = Path("data/processed/customer_features.parquet")
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)

    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Test the preprocessor transformation on X_train
    print("\n--- 4. Validating Preprocessor Matrix Output ---")
    X_train_transformed = preprocessor.fit_transform(X_train)
    print(f"✅ Transformed Feature Matrix Shape: {X_train_transformed.shape}")
    print(f"   (Expanded from {len(X.columns)} raw columns to {X_train_transformed.shape[1]} one-hot encoded columns)")


if __name__ == "__main__":
    main()