import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np

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

from src.model_pipeline import (
    prepare_features_and_target,
    split_data
)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/best_xgboost_churn_pipeline.joblib")


def plot_evaluation_dashboard(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Generates a 4-panel executive model evaluation dashboard:
    1. Annotated Confusion Matrix
    2. Receiver Operating Characteristic (ROC) Curve
    3. Precision-Recall Curve
    4. Business Cost/Benefit Threshold Simulation
    """
    print("\n--- Generating 4-Panel Model Evaluation Dashboard ---")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # ==========================================
    # 1. Confusion Matrix
    # ==========================================
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum() * 100

    # Annotate with both raw count and percentage
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
    ax1.set_title("Confusion Matrix (Unseen Test Set: 18,672 Customers)", pad=15)

    # ==========================================
    # 2. ROC Curve
    # ==========================================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    ax2.plot(fpr, tpr, color="#2980b9", linewidth=2.5, label=f"Tuned XGBoost (AUC = {auc_score:.4f})")
    ax2.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", label="Random Baseline (AUC = 0.5000)")
    ax2.set_title("Receiver Operating Characteristic (ROC) Curve", pad=15)
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=10, fontweight="bold")
    ax2.set_ylabel("True Positive Rate (Recall)", labelpad=10, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)

    # ==========================================
    # 3. Precision-Recall Curve
    # ==========================================
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    ax3.plot(recalls, precisions, color="#27ae60", linewidth=2.5, label=f"XGBoost (PR-AUC = {avg_precision:.4f})")
    ax3.axhline(y_test.mean(), color="#e74c3c", linestyle=":", label=f"No-Skill Baseline ({y_test.mean()*100:.1f}%)")
    ax3.set_title("Precision-Recall Curve", pad=15)
    ax3.set_xlabel("Recall (Fraction of Lost Customers Caught)", labelpad=10, fontweight="bold")
    ax3.set_ylabel("Precision (Accuracy of Churn Alarms)", labelpad=10, fontweight="bold")
    ax3.legend(loc="lower left", frameon=True)

    # ==========================================
    # 4. Business Cost/Benefit Threshold Simulation
    # ==========================================
    # Business Assumptions:
    # - Average Churned Customer Value (Lost LTV) = R$ 165.20
    # - Retention Campaign Intervention Cost (Discount Voucher / Outreach) = R$ 15.00
    # - Success rate of saving an identified at-risk customer = 25% (Industry benchmark)
    
    LTV_SAVED_VALUE = 165.20 * 0.25 # Net value preserved per True Positive = R$ 41.30
    INTERVENTION_COST = 15.00        # Cost per predicted churner (TP + FP)

    thresholds = np.linspace(0.10, 0.90, 81)
    net_values = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        
        gross_value = tp * LTV_SAVED_VALUE
        total_campaign_cost = (tp + fp) * INTERVENTION_COST
        net_profit_saved = gross_value - total_campaign_cost
        net_values.append(net_profit_saved)

    best_idx = np.argmax(net_values)
    best_threshold = thresholds[best_idx]
    max_savings = net_values[best_idx]

    ax4.plot(thresholds, net_values, color="#8e44ad", linewidth=2.5)
    ax4.axvline(best_threshold, color="#e74c3c", linestyle="--", linewidth=2, 
                label=f"Optimal Business Threshold: {best_threshold:.2f}\n(Net Preserved: R$ {max_savings:,.0f})")
    ax4.axvline(0.50, color="#7f8c8d", linestyle=":", label="Default Threshold: 0.50")
    ax4.set_title("Commercial ROI Simulation: Net Value Preserved vs. Threshold", pad=15)
    ax4.set_xlabel("Churn Probability Decision Threshold", labelpad=10, fontweight="bold")
    ax4.set_ylabel("Net Preserved Value (R$)", fontweight="bold")
    ax4.legend(loc="lower center", frameon=True)

    plt.suptitle("XGBoost Production Evaluation: Technical Precision & Commercial Impact", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "10_model_evaluation_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Evaluation Dashboard: {save_path}")

    # Print classification report
    print("\n📋 CLASSIFICATION REPORT (Default 0.50 Threshold):")
    print(classification_report(y_test, y_pred, target_names=["Active (0)", "Churned (1)"]))

    print(f"💡 COMMERCIAL STRATEGY RECOMMENDATION:")
    print(f"   • Standard ML Threshold: 0.50 (Assumes false positives and false negatives cost the same).")
    print(f"   • Optimal Commercial Threshold: {best_threshold:.2f}")
    print(f"   • Executing retention campaigns at threshold {best_threshold:.2f} maximizes net profit preserved at R$ {max_savings:,.2f} on the test cohort alone!")


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Trained model not found at {MODEL_PATH}! Run src/train_xgboost.py first.")
        return

    customers_path = Path("data/processed/customer_features.parquet")
    df_customers = pd.read_parquet(customers_path)

    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    _, X_test, _, y_test = split_data(X, y)

    print(f"Loading champion model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    plot_evaluation_dashboard(y_test, y_pred, y_prob)


if __name__ == "__main__":
    main()