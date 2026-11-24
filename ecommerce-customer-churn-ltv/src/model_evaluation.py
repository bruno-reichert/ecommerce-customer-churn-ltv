import sys
from pathlib import Path
from typing import Any

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

# Headless rendering backend
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    BASE_WIN_BACK_RATE,
    COUPON_COST,
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
)
from src.model_pipeline import prepare_features_and_target, split_data_chronological

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"


def find_optimal_business_threshold(y_val: pd.Series, y_val_prob: np.ndarray, avg_churner_ltv: float) -> tuple[float, float]:
    """
    Simulates commercial net profit across thresholds strictly on the VALIDATION SET.
    Returns (optimal_threshold, max_validation_net_profit).
    """
    thresholds = np.linspace(0.10, 0.99, 90)
    net_values = []
    ltv_saved_unit = avg_churner_ltv * BASE_WIN_BACK_RATE

    for t in thresholds:
        preds = (y_val_prob >= t).astype(int)
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
) -> dict[str, Any]:
    """
    Evaluates final XGBoost on held-out test set:
    1. Locks threshold from validation set.
    2. Reports test ROC-AUC, PR-AUC, Brier score, and commercial net profit.
    3. Produces a 5-panel executive dashboard including Calibration Curve.
    """
    print("\n" + "=" * 65)
    print("🎯 PRODUCTION MODEL EVALUATION (Tuned on Val, Tested on Test)")
    print("=" * 65)

    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # 1. Tune Threshold on Validation Set ONLY
    optimal_thresh, _val_profit = find_optimal_business_threshold(y_val, y_val_prob, avg_churner_ltv)
    print(f"📌 Optimal Business Threshold (Tuned on Validation Set): {optimal_thresh:.2f}")

    # 2. Apply Locked-In Threshold to Unseen Test Set
    y_test_pred = (y_test_prob >= optimal_thresh).astype(int)
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_pr_auc = average_precision_score(y_test, y_test_prob)
    test_brier = brier_score_loss(y_test, y_test_prob)

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
    print(f"   • Test Brier Score:        {test_brier:.4f}")
    print(f"   • Customers Targeted:      {test_targeted:,} / {len(y_test):,}")
    print(f"   • Campaign Cost:           R$ {test_cost:,.2f}")
    print(f"   • Honest Gross Preserved:  R$ {test_gross:,.2f}")
    print(f"   • Net Profit Preserved:    R$ {test_net:,.2f}")
    print(f"   • Campaign ROI:            {test_roi:.1f}%")

    # ==========================================
    # 3. 5-Panel Evaluation Dashboard
    # ==========================================
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1:])  # Span 2 columns for threshold curve

    # Panel 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_pct = cm / cm.sum() * 100
    annot_matrix = np.array([
        [f"{cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"{cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"{cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
    ])
    sns.heatmap(
        cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=False, ax=ax1,
        xticklabels=["Pred Active", "Pred Churn"],
        yticklabels=["Actual Active", "Actual Churn"],
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    ax1.set_title(f"Test Confusion Matrix (@ {optimal_thresh:.2f})", pad=12)

    # Panel 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    ax2.plot(fpr, tpr, color="#2980b9", linewidth=2.5, label=f"XGBoost (AUC={test_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", label="Random Baseline")
    ax2.set_title("Test Set ROC Curve", pad=12)
    ax2.set_xlabel("False Positive Rate", labelpad=8, fontweight="bold")
    ax2.set_ylabel("True Positive Rate", labelpad=8, fontweight="bold")
    ax2.legend(loc="lower right")

    # Panel 3: Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
    ax3.plot(rec, prec, color="#27ae60", linewidth=2.5, label=f"XGBoost (PR-AUC={test_pr_auc:.4f})")
    ax3.axhline(y_test.mean(), color="#e74c3c", linestyle=":", label=f"Baseline ({y_test.mean()*100:.1f}%)")
    ax3.set_title("Test Set Precision-Recall Curve", pad=12)
    ax3.set_xlabel("Recall", labelpad=8, fontweight="bold")
    ax3.set_ylabel("Precision", labelpad=8, fontweight="bold")
    ax3.legend(loc="lower left")

    # Panel 4: Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=8, strategy="uniform")
    ax4.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax4.plot(prob_pred, prob_true, marker="s", color="#8e44ad", linewidth=2, label="Tuned XGBoost")
    ax4.set_title(f"Probability Calibration (Brier={test_brier:.4f})", pad=12)
    ax4.set_xlabel("Mean Predicted Probability", labelpad=8, fontweight="bold")
    ax4.set_ylabel("Fraction of True Churners", labelpad=8, fontweight="bold")
    ax4.legend(loc="lower right")
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Panel 5: Validation Set Threshold Optimization Curve
    val_thresholds = np.linspace(0.10, 0.99, 90)
    val_nets = []
    for t in val_thresholds:
        preds = (y_val_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        val_nets.append((tp * avg_churner_ltv * BASE_WIN_BACK_RATE) - ((tp + fp) * COUPON_COST))

    ax5.plot(val_thresholds, val_nets, color="#d35400", linewidth=2.5)
    ax5.axvline(optimal_thresh, color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Val Optimal Threshold: {optimal_thresh:.2f}")
    ax5.set_title("Validation Set Decision Threshold Optimization (Net Preserved R$)", pad=12)
    ax5.set_xlabel("Decision Threshold", labelpad=8, fontweight="bold")
    ax5.set_ylabel("Validation Net Profit (R$)", labelpad=8, fontweight="bold")
    ax5.legend(loc="lower center")
    ax5.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("XGBoost Evaluation: Out-of-Time Generalization, Calibration & Commercial ROI", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "10_model_evaluation_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved 5-panel evaluation dashboard: {save_path}")

    return {
        "optimal_threshold": optimal_thresh,
        "test_roc_auc": round(test_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
        "test_brier_score": round(test_brier, 4),
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
    avg_churner_ltv = float(df_customers[df_customers["is_churned"] == 1]["total_spend"].mean())

    X, y, timestamps, _num_cols, _cat_cols = prepare_features_and_target(df_customers)
    _X_train, X_val, X_test, _y_train, y_val, y_test = split_data_chronological(X, y, timestamps)

    pipeline = joblib.load(MODEL_PATH)
    evaluate_on_unseen_test(pipeline, X_val, y_val, X_test, y_test, avg_churner_ltv)


if __name__ == "__main__":
    main()