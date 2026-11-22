import sys
from pathlib import Path
# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RANDOM_SEED
)
from src.model_pipeline import (
    prepare_features_and_target,
    build_preprocessor_pipeline,
    split_data_chronological
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_tune_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    num_cols: list, 
    cat_cols: list
) -> Pipeline:
    """
    Builds an end-to-end XGBoost Pipeline, performs Cross-Validated
    Hyperparameter Optimization on the training set, and validates on the validation set.
    """
    print("=" * 65)
    print("⚡ XGBOOST MODELING & HYPERPARAMETER TUNING (Chronological Split)")
    print("=" * 65)

    # 1. Build Preprocessor
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)

    # 2. Base XGBoost Classifier
    # Target definition: 1 = churned (majority), 0 = repurchased (minority)
    # For predicting churn (class 1), class 1 is the majority.
    xgb_base = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_base)
    ])

    # 3. Define the Hyperparameter Search Space
    param_distributions = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 3, 5]
    }

    print("\n🔍 Running Randomized Search Cross-Validation (3 Folds, 10 Iterations)...")
    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="roc_auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\n🏆 Best Cross-Validation ROC-AUC: {search.best_score_:.4f}")
    print("📌 Optimal Hyperparameters Found:")
    for param, val in search.best_params_.items():
        print(f"   • {param.replace('classifier__', '')}: {val}")

    best_pipeline = search.best_estimator_

    # 4. Evaluation on Validation Set
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = best_pipeline.predict(X_val)

    val_auc = roc_auc_score(y_val, y_val_prob)
    val_pr_auc = average_precision_score(y_val, y_val_prob)

    print("\n" + "=" * 65)
    print("🎯 VALIDATION SET EVALUATION (Tuning & Diagnostics)")
    print("=" * 65)
    print(f"   • Validation ROC-AUC:    {val_auc:.4f}")
    print(f"   • Validation PR-AUC:     {val_pr_auc:.4f}")

    # 5. Persist Best Model Artifact
    model_path = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\n💾 Saved model pipeline to: {model_path}")

    return best_pipeline


def main():
    customers_path = DATA_PROCESSED / "customer_features.parquet"
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)
    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronological(X, y, timestamps)

    train_and_tune_xgboost(X_train, y_train, X_val, y_val, num_cols, cat_cols)


if __name__ == "__main__":
    main()