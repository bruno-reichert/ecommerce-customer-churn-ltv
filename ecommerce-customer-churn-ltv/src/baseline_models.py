import sys
from pathlib import Path
from typing import Any

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Headless rendering backend
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED, FIGURES_DIR, RANDOM_SEED
from src.model_pipeline import (
    build_preprocessor_pipeline,
    prepare_features_and_target,
    split_data_chronological,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_baseline_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[dict[str, Any], np.ndarray]:
    """
    Fits a Scikit-Learn pipeline on chronological training data and evaluates
    performance on the unseen test set across classification, ranking (ROC/PR),
    and probabilistic calibration metrics.
    """
    print(f"\n🚀 Training {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Probabilistic scoring
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
    else:
        y_prob = np.full(len(y_test), y_train.mean())
        roc_auc = 0.50
        pr_auc = float(y_test.mean())
        brier = brier_score_loss(y_test, y_prob)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier
    }

    print(f"   • ROC-AUC:     {metrics['ROC-AUC']:.4f}")
    print(f"   • PR-AUC:      {metrics['PR-AUC']:.4f}")
    print(f"   • Brier Score: {metrics['Brier Score']:.4f} (Lower is better)")
    return metrics, y_prob


def run_baseline_benchmarks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_cols: list,
    cat_cols: list
) -> pd.DataFrame:
    """
    Constructs and runs 3 baseline pipelines:
    1. Dummy Classifier (The Floor)
    2. Logistic Regression (The Linear Benchmark)
    3. Random Forest (The Ensemble Non-Linear Benchmark)
    Plots a multi-model Calibration Curve on the held-out test set.
    """
    print("=" * 65)
    print("🥊 RUNNING BASELINE MODEL BENCHMARKS (Chronological Split)")
    print("=" * 65)

    models = {
        "1. Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "2. Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "3. Random Forest (100 Trees)": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    results = []
    probabilities = {}

    for name, clf in models.items():
        preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        metrics, y_prob = evaluate_baseline_model(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(metrics)
        probabilities[name] = y_prob

    results_df = pd.DataFrame(results)

    # ==========================================
    # Plot Baseline Calibration Curves
    # ==========================================
    print("\n📈 Generating Baseline Calibration Curves...")
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration (Diagonal)")

    colors = ["#7f8c8d", "#2980b9", "#27ae60"]
    for (name, y_prob), color in zip(probabilities.items(), colors):
        if name.startswith("1. Dummy"):
            continue  # Dummy has no continuous probability spread
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name, color=color)

    plt.title("Probability Calibration Curves (Baseline Models - Test Set)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Mean Predicted Probability of Churn", fontsize=11, fontweight="bold")
    plt.ylabel("Fraction of True Churners (Positives)", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    cal_path = FIGURES_DIR / "09_baseline_calibration_curve.png"
    plt.savefig(cal_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved calibration curve figure: {cal_path}")

    # Display clean scoreboard
    print("\n" + "=" * 70)
    print("📊 BASELINE COMPARISON SCOREBOARD (Unseen Chronological Test Set)")
    print("=" * 70)
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    display_df["ROC-AUC"] = display_df["ROC-AUC"].round(4)
    display_df["PR-AUC"] = display_df["PR-AUC"].round(4)
    display_df["Brier Score"] = display_df["Brier Score"].round(4)
    print(display_df.to_string(index=False))
    print("=" * 70)

    return results_df


def main():
    customers_path = DATA_PROCESSED / "customer_features.parquet"
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)
    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    X_train, _X_val, X_test, y_train, _y_val, y_test = split_data_chronological(X, y, timestamps)

    run_baseline_benchmarks(X_train, X_test, y_train, y_test, num_cols, cat_cols)


if __name__ == "__main__":
    main()