"""
XGBoost Modeling & Hyperparameter Optimization Module.
Evaluates the rare positive class: repurchased_in_window (1 = repeat buyer, 0 = churned).
Uses PR-AUC optimization to rank rare repeat purchases out-of-time.
"""

import sys
from pathlib import Path

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline_models import compute_cumulative_gains_and_lift
from src.config import MODELS_DIR, RANDOM_SEED
from src.model_pipeline import (
    build_preprocessor_pipeline,
    load_out_of_time_splits,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_tune_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: list, 
    cat_cols: list
) -> Pipeline:
    """
    Builds an end-to-end XGBoost Pipeline, performs Cross-Validated
    Hyperparameter Optimization targeting PR-AUC, and evaluates on Snapshot 2.
    """
    print("=" * 70)
    print("⚡ TUNED XGBOOST MODELING & HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # 1. Build Preprocessor
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)

    # 2. Base XGBoost Classifier
    # Positive class: 1 = repurchased (~1.63% in train)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    xgb_base = XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_base)
    ])

    # 3. Hyperparameter Search Space
    param_distributions = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.02, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__min_child_weight": [1, 3, 5]
    }

    print("\n🔍 Running Randomized Search CV (3 Folds, 10 Iterations, Scoring: PR-AUC)...")
    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="average_precision",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\n🏆 Best Cross-Validation PR-AUC: {search.best_score_:.4f}")
    print("📌 Optimal Hyperparameters Found:")
    for param, val in search.best_params_.items():
        print(f"   • {param.replace('classifier__', '')}: {val}")

    tuned_pipeline = search.best_estimator_

    # 4. Out-of-Time Test Evaluation (Snapshot 2)
    y_test_prob = tuned_pipeline.predict_proba(X_test)[:, 1]
    prevalence = float(y_test.mean())

    test_auc = roc_auc_score(y_test, y_test_prob)
    test_pr = average_precision_score(y_test, y_test_prob)
    test_brier = brier_score_loss(y_test, y_test_prob)
    gains_10, lift_10 = compute_cumulative_gains_and_lift(y_test, y_test_prob, top_pct=0.10)
    gains_20, lift_20 = compute_cumulative_gains_and_lift(y_test, y_test_prob, top_pct=0.20)

    print("\n" + "=" * 70)
    print("🎯 OUT-OF-TIME EVALUATION ON SNAPSHOT 2 (N = {:,})".format(len(X_test)))
    print("=" * 70)
    print(f"   • ROC-AUC:      {test_auc:.4f}")
    print(f"   • PR-AUC:       {test_pr:.4f} (Prevalence Baseline: {prevalence:.4f})")
    print(f"   • Brier Score:  {test_brier:.4f}")
    print(f"   • Top 10% Lift: {lift_10:.2f}x (Captures {gains_10:.1f}% of repeat buyers)")
    print(f"   • Top 20% Lift: {lift_20:.2f}x (Captures {gains_20:.1f}% of repeat buyers)")

    # 5. Persist Best Model Artifact
    model_path = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"
    joblib.dump(tuned_pipeline, model_path)
    print(f"\n💾 Saved tuned pipeline artifact to: {model_path}")

    return tuned_pipeline


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols = load_out_of_time_splits()
    train_and_tune_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, num_cols, cat_cols)


if __name__ == "__main__":
    main()