import sys
from pathlib import Path
from typing import Dict, Any
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.model_pipeline import (
    prepare_features_and_target,
    build_preprocessor_pipeline,
    split_data
)


def evaluate_model(model_name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Fits a Scikit-Learn pipeline on training data and evaluates performance
    on the unseen testing set across standard classification metrics.
    """
    print(f"\n🚀 Training {model_name}...")
    pipeline.fit(X_train, y_train)

    # Predict binary classes (0 or 1)
    y_pred = pipeline.predict(X_test)

    # Predict churn probabilities for ROC-AUC
    # Some models like Dummy or hard classifiers handle probabilities differently
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = 0.50

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc
    }

    print(f"   • Accuracy:  {metrics['Accuracy']*100:.2f}%")
    print(f"   • Precision: {metrics['Precision']*100:.2f}%")
    print(f"   • Recall:    {metrics['Recall']*100:.2f}%")
    print(f"   • F1-Score:  {metrics['F1-Score']*100:.2f}%")
    print(f"   • ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    return metrics


def run_baseline_benchmarks(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    Constructs and runs all 3 baseline pipelines:
    1. Dummy Classifier (The Floor)
    2. Logistic Regression (The Linear Benchmark)
    3. Random Forest (The Ensemble Non-Linear Benchmark)
    """
    print("=" * 65)
    print("🥊 RUNNING BASELINE MODEL BENCHMARKS")
    print("=" * 65)

    # Define the 3 benchmark candidate models
    models = {
        "1. Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "2. Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "3. Random Forest (100 Trees)": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1  # Uses all available CPU cores for speed
        )
    }

    results = []
    for name, clf in models.items():
        # Preprocessor is rebuilt cleanly for each pipeline to guarantee zero contamination
        preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
        
        # Assemble the full end-to-end pipeline: Preprocess -> Classify
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        metrics = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("📊 BASELINE COMPARISON SCOREBOARD (Unseen Test Set)")
    print("=" * 65)
    
    # Format for clean display
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    display_df["ROC-AUC"] = display_df["ROC-AUC"].round(4)
    
    print(display_df.to_string(index=False))
    print("=" * 65)

    return results_df


def main():
    customers_path = Path("data/processed/customer_features.parquet")
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)

    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results_df = run_baseline_benchmarks(X_train, X_test, y_train, y_test, num_cols, cat_cols)


if __name__ == "__main__":
    main()