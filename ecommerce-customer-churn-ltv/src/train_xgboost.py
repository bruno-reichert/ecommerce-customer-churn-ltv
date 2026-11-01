import sys
from pathlib import Path
from typing import Dict, Any
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
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

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_tune_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    num_cols: list, 
    cat_cols: list
) -> Pipeline:
    """
    Builds an end-to-end XGBoost Pipeline, performs Cross-Validated
    Hyperparameter Optimization, evaluates on the unseen test set,
    and persists the best model artifact to disk.
    """
    print("=" * 65)
    print("⚡ PHASE 4.3: XGBOOST MODELING & HYPERPARAMETER TUNING")
    print("=" * 65)

    # 1. Build Preprocessor
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)

    # 2. Base XGBoost Classifier
    # scale_pos_weight: ratio of negative to positive class (~40.8% / ~59.2% ≈ 0.70)
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_base)
    ])

    # 3. Define the Hyperparameter Search Space
    param_distributions = {
        "classifier__n_estimators": [100, 150, 200, 250],
        "classifier__max_depth": [4, 5, 6, 7],
        "classifier__learning_rate": [0.03, 0.05, 0.1, 0.15],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 3, 5]
    }

    print("\n🔍 Running Randomized Search Cross-Validation (3 Folds, 12 Iterations)...")
    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\n🏆 Best Cross-Validation ROC-AUC: {search.best_score_:.4f}")
    print("📌 Optimal Hyperparameters Found:")
    for param, val in search.best_params_.items():
        print(f"   • {param.replace('classifier__', '')}: {val}")

    best_pipeline = search.best_estimator_

    # 4. Final Evaluation on Unseen Test Set
    print("\n" + "=" * 65)
    print("🎯 FINAL EVALUATION ON UNSEEN TEST SET (18,672 Customers)")
    print("=" * 65)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"   • Accuracy:       {acc*100:.2f}%")
    print(f"   • Precision:      {prec*100:.2f}%")
    print(f"   • Recall:         {rec*100:.2f}%")
    print(f"   • F1-Score:       {f1*100:.2f}%")
    print(f"   • ROC-AUC Score:  {auc:.4f}")

    # 5. Persist Best Model Artifact
    model_path = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\n💾 Saved production-ready model pipeline to: {model_path}")

    return best_pipeline


def main():
    customers_path = Path("data/processed/customer_features.parquet")
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)

    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_model = train_and_tune_xgboost(X_train, y_train, X_test, y_test, num_cols, cat_cols)


if __name__ == "__main__":
    main()