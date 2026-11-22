import sys
from pathlib import Path
# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


# Headless rendering backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    FIGURES_DIR,
    COUPON_COST,
    BASE_WIN_BACK_RATE
)
from src.model_pipeline import (
    prepare_features_and_target,
    split_data_chronological
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"


def find_optimal_business_threshold(y_val: pd.Series, y_val_prob: np.ndarray, avg_churner_ltv: float) -> Tuple[float, float]:
    """
    Simulates commercial net profit across thresholds strictly on the VALIDATION SET.
    Returns (optimal_threshold, max_validation_net_profit).
    """
    thresholds = np.linspace(0.10, 0.99, 90)
    net_values = []

    # Net value preserved per True Positive (saved churner)
    ltv_saved_unit = avg_churner_ltv * BASE_WIN_BACK_RATE

    for t in thresholds:
        preds = (y_val_prob >= t).astype(int)
        # Honest credit: only actual churners who were targeted and saved
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))

        gross_value = tp * ltv_saved_unit
        total_campaign_cost = (tp + fp) * COUPON_COST
        net_profit = gross_value - total_campaign_cost
        net_values.append(net_profit)

    best_idx = int(np.argmax(net_values))
    optimal_threshold = float(thresholds[best_idx])
    max_val_savings = float(net_values[best_idx])

    return optimal_threshold, max_val_savings


def evaluate_on_unseen_test(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    avg_churner_ltv: float
) -> Dict[str, Any]:
    """
    1. Tunes threshold on validation set.
    2. Evaluates performance on unseen test set using the tuned threshold.
    3. Produces evaluation plots.
    """
    print("\n" + "=" * 65)
    print("🎯 PRODUCTION MODEL EVALUATION (Tuned on Val, Tested on Test)")
    print("=" * 65)

    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # 1. Tune Threshold on Validation Set ONLY
    optimal_thresh, val_profit = find_optimal_business_threshold(y_val, y_val_prob, avg_churner_ltv)
    print(f"📌 Optimal Business Threshold (Tuned on Validation Set): {optimal_thresh:.2f}")
    print(f"   (Validation Net Profit Preserved at {optimal_thresh:.2f}: R$ {val_profit:,.2f})")

    # 2. Apply Locked-In Threshold to Unseen Test Set
    y_test_pred = (y_test_prob >= optimal_thresh).astype(int)

    test_auc = roc_auc_score(y_test, y_test_prob)
    test_pr_auc = average_precision_score(y_test, y_test_prob)

    # Calculate commercial impact on test set with locked threshold
    tp_test = np.sum((y_test_pred == 1) & (y_test == 1))
    fp_test = np.sum((y_test_pred == 1) & (y_test == 0))
    test_targeted = tp_test + fp_test
    test_gross = tp_test * avg_churner_ltv * BASE_WIN_BACK_RATE
    test_cost = test_targeted * COUPON_COST
    test_net = test_gross - test_cost
    test_roi = (test_net / test_cost * 100) if test_cost > 0 else 0.0

    print(f"\n📊 UNSEEN TEST SET METRICS (Locked Threshold = {optimal_thresh:.2f}):")
    print(f"   • Test ROC-AUC:            {test_auc:.4f}")
    print(f"   • Test PR-AUC:             {test_pr_auc:.4f}")
    print(f"   • Customers Targeted:      {test_targeted:,} / {len(y_test):,}")
    print(f"   • Campaign Cost:           R$ {test_cost:,.2f}")
    print(f"   • Honest Gross Preserved:  R$ {test_gross:,.2f}")
    print(f"   • Net Profit Preserved:    R$ {test_net:,.2f}")
    print(f"   • Campaign ROI:            {test_roi:.1f}%")

    # ==========================================
    # 3. 4-Panel Evaluation Plot
    # ==========================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_pct = cm / cm.sum() * 100
    annot_matrix = np.array([
        [f"{cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"{cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"{cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
    ])
    sns.heatmap(
        cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=False, ax=ax1,
        xticklabels=["Predicted Active", "Predicted Churn"],
        yticklabels=["Actual Active", "Actual Churn"],
        annot_kws={"fontsize": 12, "fontweight": "bold"}
    )
    ax1.set_title(f"Test Confusion Matrix (Threshold = {optimal_thresh:.2f})", pad=15)

    # Panel 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    ax2.plot(fpr, tpr, color="#2980b9", linewidth=2.5, label=f"XGBoost (AUC = {test_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", label="Random Baseline (0.50)")
    ax2.set_title("Test Set ROC Curve", pad=15)
    ax2.set_xlabel("False Positive Rate", labelpad=10, fontweight="bold")
    ax2.set_ylabel("True Positive Rate", labelpad=10, fontweight="bold")
    ax2.legend(loc="lower right")

    # Panel 3: Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
    ax3.plot(rec, prec, color="#27ae60", linewidth=2.5, label=f"XGBoost (PR-AUC = {test_pr_auc:.4f})")
    ax3.axhline(y_test.mean(), color="#e74c3c", linestyle=":", label=f"Baseline ({y_test.mean()*100:.1f}%)")
    ax3.set_title("Test Set Precision-Recall Curve", pad=15)
    ax3.set_xlabel("Recall", labelpad=10, fontweight="bold")
    ax3.set_ylabel("Precision", labelpad=10, fontweight="bold")
    ax3.legend(loc="lower left")

    # Panel 4: Validation Set Threshold Optimization Curve
    val_thresholds = np.linspace(0.10, 0.99, 90)
    val_nets = []
    for t in val_thresholds:
        preds = (y_val_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        val_nets.append((tp * avg_churner_ltv * BASE_WIN_BACK_RATE) - ((tp + fp) * COUPON_COST))

    ax4.plot(val_thresholds, val_nets, color="#8e44ad", linewidth=2.5)
    ax4.axvline(optimal_thresh, color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Val Optimal: {optimal_thresh:.2f}")
    ax4.set_title("Validation Set Threshold Optimization", pad=15)
    ax4.set_xlabel("Decision Threshold", labelpad=10, fontweight="bold")
    ax4.set_ylabel("Net Preserved Value (R$)", fontweight="bold")
    ax4.legend(loc="lower center")

    plt.suptitle("XGBoost Evaluation: Validation-Tuned Threshold Applied to Unseen Test Set", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "10_model_evaluation_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved updated evaluation dashboard: {save_path}")

    return {
        "optimal_threshold": optimal_thresh,
        "test_roc_auc": round(test_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
        "test_targeted": int(test_targeted),
        "test_campaign_cost": round(test_cost, 2),
        "test_gross_preserved": round(test_gross, 2),
        "test_net_profit": round(test_net, 2),
        "test_roi_pct": round(test_roi, 2)
    }


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model artifact not found at {MODEL_PATH}! Run src/train_xgboost.py first.")
        return

    customers_path = DATA_PROCESSED / "customer_features.parquet"
    df_customers = pd.read_parquet(customers_path)

    # Average LTV of customers who churned
    avg_churner_ltv = float(df_customers[df_customers["is_churned"] == 1]["total_spend"].mean())

    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronological(X, y, timestamps)

    pipeline = joblib.load(MODEL_PATH)
    evaluate_on_unseen_test(pipeline, X_val, y_val, X_test, y_test, avg_churner_ltv)


if __name__ == "__main__":
    main()