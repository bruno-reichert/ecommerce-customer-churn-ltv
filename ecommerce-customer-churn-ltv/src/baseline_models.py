"""
Baseline Model Benchmarks & Probability Calibration Module.
Evaluates the rare positive class: repurchased_in_window (1 = repeat buyer, 0 = churned).
Reports ROC-AUC, PR-AUC against the prevalence baseline, Brier Score, and Cumulative Lift.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Headless rendering backend
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, RANDOM_SEED
from src.model_pipeline import build_preprocessor_pipeline, load_out_of_time_splits

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_cumulative_gains_and_lift(y_true: pd.Series, y_prob: np.ndarray, top_pct: float = 0.10) -> Tuple[float, float]:
    """
    Computes Cumulative Gains (% of true positives captured in top X%)
    and Lift (ratio of capture rate vs random selection).
    """
    df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob})
    df = df.sort_values(by="y_prob", ascending=False).reset_index(drop=True)

    n_total = len(df)
    n_positives = df["y_true"].sum()

    if n_positives == 0:
        return 0.0, 1.0

    cutoff_idx = int(n_total * top_pct)
    top_slice = df.iloc[:cutoff_idx]
    positives_captured = top_slice["y_true"].sum()

    gains_pct = (positives_captured / n_positives) * 100.0
    random_expected = n_positives * top_pct
    lift = positives_captured / random_expected if random_expected > 0 else 1.0

    return gains_pct, lift


def evaluate_model_repurchase(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Fits model on training snapshot and evaluates on out-of-time test snapshot.
    Evaluates probability of repurchase (the rare positive event).
    """
    print(f"\n🚀 Training {model_name}...")
    pipeline.fit(X_train, y_train)

    if hasattr(pipeline, "predict_proba"):
        # Positive class: 1 = Repurchased
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = np.full(len(y_test), y_train.mean())

    prevalence_baseline = float(y_test.mean())
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    gains_10, lift_10 = compute_cumulative_gains_and_lift(y_test, y_prob, top_pct=0.10)
    gains_20, lift_20 = compute_cumulative_gains_and_lift(y_test, y_prob, top_pct=0.20)

    metrics = {
        "Model": model_name,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Prevalence Baseline": prevalence_baseline,
        "Brier Score": brier,
        "Top 10% Gains": gains_10,
        "Top 10% Lift": lift_10,
        "Top 20% Gains": gains_20,
        "Top 20% Lift": lift_20
    }

    print(f"   • ROC-AUC:      {roc_auc:.4f}")
    print(f"   • PR-AUC:       {pr_auc:.4f} (Prevalence Baseline: {prevalence_baseline:.4f})")
    print(f"   • Brier Score:  {brier:.4f}")
    print(f"   • Top 10% Lift: {lift_10:.2f}x (Captures {gains_10:.1f}% of repeat buyers)")
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
    Runs 3 baseline models evaluated on predicting repeat purchase:
    1. Dummy Classifier (Prior prevalence)
    2. Logistic Regression (Linear benchmark with class weighting)
    3. Random Forest (100 Trees with class weighting)
    """
    print("=" * 70)
    print("🥊 RUNNING REPURCHASE BASELINE BENCHMARKS (Out-of-Time Snapshot 2)")
    print("=" * 70)

    models = {
        "1. Dummy (Prior Mean)": DummyClassifier(strategy="prior"),
        "2. Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "3. Random Forest (100 Trees)": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
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
        metrics, y_prob = evaluate_model_repurchase(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(metrics)
        probabilities[name] = y_prob

    results_df = pd.DataFrame(results)

    # ==========================================
    # Plot Probability Calibration Curves
    # ==========================================
    print("\n📈 Generating Baseline Calibration Curves (Repurchase Probability)...")
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration (Diagonal)")

    colors = ["#7f8c8d", "#2980b9", "#27ae60"]
    for (name, y_prob), color in zip(probabilities.items(), colors):
        if name.startswith("1. Dummy"):
            continue
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name, color=color)

    plt.title("Repurchase Probability Calibration (Test Snapshot 2)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Mean Predicted Probability of Repurchase", fontsize=11, fontweight="bold")
    plt.ylabel("Fraction of True Repurchasers", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    cal_path = FIGURES_DIR / "09_baseline_calibration_curve.png"
    plt.savefig(cal_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved calibration curve: {cal_path}")

    # Display clean scoreboard
    print("\n" + "=" * 78)
    print("📊 BASELINE MODEL SCOREBOARD (Target: repurchased_in_window = 1)")
    print("=" * 78)
    display_df = results_df.copy()
    display_df["ROC-AUC"] = display_df["ROC-AUC"].round(4)
    display_df["PR-AUC"] = display_df["PR-AUC"].round(4)
    display_df["Prevalence Baseline"] = display_df["Prevalence Baseline"].round(4)
    display_df["Brier Score"] = display_df["Brier Score"].round(4)
    display_df["Top 10% Gains"] = display_df["Top 10% Gains"].round(1).astype(str) + "%"
    display_df["Top 10% Lift"] = display_df["Top 10% Lift"].round(2).astype(str) + "x"
    display_df["Top 20% Lift"] = display_df["Top 20% Lift"].round(2).astype(str) + "x"

    cols = ["Model", "ROC-AUC", "PR-AUC", "Prevalence Baseline", "Brier Score", "Top 10% Gains", "Top 10% Lift", "Top 20% Lift"]
    print(display_df[cols].to_string(index=False))
    print("=" * 78)

    return results_df


def main():
    X_train, _X_val, X_test, y_train, _y_val, y_test, num_cols, cat_cols = load_out_of_time_splits()
    run_baseline_benchmarks(X_train, X_test, y_train, y_test, num_cols, cat_cols)


if __name__ == "__main__":
    main()