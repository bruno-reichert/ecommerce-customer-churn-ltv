import json
import sys
from pathlib import Path
from typing import Any

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

# Headless rendering backend
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    BASE_WIN_BACK_RATE,
    COUPON_COST,
    DATA_PROCESSED,
    FIGURES_DIR,
    METRICS_JSON_PATH,
    MODELS_DIR,
    SENSITIVITY_COUPON_COSTS,
    SENSITIVITY_WIN_BACK_RATES,
)
from src.model_evaluation import find_optimal_business_threshold
from src.model_pipeline import prepare_features_and_target, split_data_chronological

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"


def run_honest_financial_simulation(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_customers: pd.DataFrame,
    test_indices: pd.Index
) -> dict[str, Any]:
    """
    Executes an honest commercial retention simulation:
    1. Tunes threshold on Validation Set.
    2. Runs simulation STRICTLY on held-out Test Set.
    3. Uses honest credit: only actual churners who were targeted are credited.
    4. Evaluates sensitivity grid (varying win-back rate and coupon cost).
    5. Exports all figures and writes reports/metrics.json.
    """
    print("=" * 65)
    print("💰 HONEST COMMERCIAL RETENTION SIMULATION (Held-out Test Cohort)")
    print("=" * 65)

    # 1. Align customer attributes (spend, persona) for test set
    test_cust = df_customers.loc[test_indices].copy().reset_index(drop=True)
    avg_churner_ltv = float(test_cust[y_test == 1]["total_spend"].mean())

    # 2. Predict churn probabilities
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # 3. Find optimal threshold on VALIDATION set
    optimal_thresh, val_profit = find_optimal_business_threshold(y_val, y_val_prob, avg_churner_ltv)
    print(f"📌 Locked Decision Threshold (Tuned on Val): {optimal_thresh:.2f}")

    test_cust["churn_prob"] = y_test_prob
    test_cust["is_churned"] = y_test.values
    test_cust["targeted"] = (test_cust["churn_prob"] >= optimal_thresh).astype(int)

    # ==========================================
    # 4. Macro Strategies Comparison (Test Set Only)
    # ==========================================
    n_test = len(test_cust)

    # Strategy 1: Status Quo
    sq_cost, sq_gross, sq_net, sq_roi = 0.0, 0.0, 0.0, 0.0

    # Strategy 2: Blanket Spray-and-Pray (Target all test customers)
    blanket_cost = n_test * COUPON_COST
    # Honest blanket gross: only actual churners can be saved
    blanket_saved_spend = (test_cust["is_churned"] * test_cust["total_spend"]).sum()
    blanket_gross = blanket_saved_spend * BASE_WIN_BACK_RATE
    blanket_net = blanket_gross - blanket_cost
    blanket_roi = (blanket_net / blanket_cost * 100) if blanket_cost > 0 else 0.0

    # Strategy 3: ML-Targeted Precision
    ml_targeted_count = int(test_cust["targeted"].sum())
    ml_cost = ml_targeted_count * COUPON_COST
    # HONEST CREDIT: Only targeted customers who ACTUALLY churned are saved!
    ml_saved_spend = (test_cust["targeted"] * test_cust["is_churned"] * test_cust["total_spend"]).sum()
    ml_gross = ml_saved_spend * BASE_WIN_BACK_RATE
    ml_net = ml_gross - ml_cost
    ml_roi = (ml_net / ml_cost * 100) if ml_cost > 0 else 0.0

    # Sanity Invariant: ML gross can NEVER exceed blanket gross
    assert ml_gross <= blanket_gross + 1e-5, (
        f"FATAL: ML Gross (R$ {ml_gross:,.2f}) exceeds Blanket Gross (R$ {blanket_gross:,.2f})!"
    )

    macro_df = pd.DataFrame([
        {
            "Strategy": "1. Status Quo (Do Nothing)",
            "Customers Targeted": 0,
            "Targeted %": "0.0%",
            "Campaign Spend (R$)": f"R$ {sq_cost:,.2f}",
            "Gross Preserved (R$)": f"R$ {sq_gross:,.2f}",
            "Net Profit (R$)": f"R$ {sq_net:,.2f}",
            "ROI (%)": f"{sq_roi:.1f}%"
        },
        {
            "Strategy": "2. Blanket Marketing (All Test Customers)",
            "Customers Targeted": n_test,
            "Targeted %": "100.0%",
            "Campaign Spend (R$)": f"R$ {blanket_cost:,.2f}",
            "Gross Preserved (R$)": f"R$ {blanket_gross:,.2f}",
            "Net Profit (R$)": f"R$ {blanket_net:,.2f}",
            "ROI (%)": f"{blanket_roi:.1f}%"
        },
        {
            "Strategy": f"3. ML-Targeted Precision (@ {optimal_thresh:.2f})",
            "Customers Targeted": ml_targeted_count,
            "Targeted %": f"{(ml_targeted_count / n_test)*100:.1f}%",
            "Campaign Spend (R$)": f"R$ {ml_cost:,.2f}",
            "Gross Preserved (R$)": f"R$ {ml_gross:,.2f}",
            "Net Profit (R$)": f"R$ {ml_net:,.2f}",
            "ROI (%)": f"{ml_roi:.1f}%"
        }
    ])

    print(f"\n📊 MACRO STRATEGY SCORECARD (Unseen Test Set: N = {n_test:,}):")
    print(macro_df.to_string(index=False))

    # ==========================================
    # 5. Sensitivity Analysis Grid (Illustrative)
    # ==========================================
    print("\n--- Running Sensitivity Analysis (Varying Win-Back Rates & Costs) ---")
    sensitivity_rows = []
    for rate in SENSITIVITY_WIN_BACK_RATES:
        for cost in SENSITIVITY_COUPON_COSTS:
            sc_cost = ml_targeted_count * cost
            sc_gross = ml_saved_spend * rate
            sc_net = sc_gross - sc_cost
            sc_roi = (sc_net / sc_cost * 100) if sc_cost > 0 else 0.0
            sensitivity_rows.append({
                "Win-Back Rate": f"{int(rate*100)}%",
                "Coupon Cost": f"R$ {cost:.2f}",
                "Campaign Spend (R$)": round(sc_cost, 2),
                "Gross Preserved (R$)": round(sc_gross, 2),
                "Net Profit (R$)": round(sc_net, 2),
                "ROI (%)": round(sc_roi, 1)
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    print("\n📋 SENSITIVITY TABLE (Illustrative - Test Set):")
    print(sensitivity_df.to_string(index=False))

    # ==========================================
    # 6. Plot Sensitivity Heatmap
    # ==========================================
    pivot_roi = sensitivity_df.pivot(index="Coupon Cost", columns="Win-Back Rate", values="ROI (%)")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_roi, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "ROI (%)"})
    plt.title("Illustrative Campaign ROI (%) Sensitivity Matrix (Test Set)", pad=15, fontweight="bold")
    plt.tight_layout()
    sens_path = FIGURES_DIR / "15_financial_simulation_roi.png"
    plt.savefig(sens_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved sensitivity heatmap: {sens_path}")

    # ==========================================
    # 7. Collect & Export Metrics to reports/metrics.json
    # ==========================================
    metrics_data = {
        "dataset": {
            "total_customers_eligible": len(df_customers),
            "test_customers": n_test,
            "test_churn_rate_pct": round(float(y_test.mean() * 100), 2)
        },
        "threshold_tuning": {
            "optimal_threshold": round(optimal_thresh, 2),
            "validation_net_profit": round(val_profit, 2)
        },
        "test_macro_strategies": {
            "blanket": {
                "targeted": n_test,
                "cost": round(blanket_cost, 2),
                "gross_preserved": round(blanket_gross, 2),
                "net_profit": round(blanket_net, 2),
                "roi_pct": round(blanket_roi, 1)
            },
            "ml_targeted": {
                "targeted": ml_targeted_count,
                "cost": round(ml_cost, 2),
                "gross_preserved": round(ml_gross, 2),
                "net_profit": round(ml_net, 2),
                "roi_pct": round(ml_roi, 1)
            }
        },
        "assumptions": {
            "base_coupon_cost": COUPON_COST,
            "base_win_back_rate": BASE_WIN_BACK_RATE,
            "simulation_nature": "illustrative"
        }
    }

    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n💾 Saved all quoted metrics dynamically to: {METRICS_JSON_PATH}")

    return metrics_data


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model artifact not found at {MODEL_PATH}!")
        return

    customers_path = DATA_PROCESSED / "customer_features.parquet"
    df_customers = pd.read_parquet(customers_path)

    X, y, timestamps, _num_cols, _cat_cols = prepare_features_and_target(df_customers)

    # Sort indices chronologically to get test slice indices
    sorted_indices = timestamps.sort_values().index
    n_total = len(sorted_indices)
    val_end = int(n_total * 0.85)  # 70% train + 15% val = 85%
    test_indices = sorted_indices[val_end:]

    _X_train, X_val, X_test, _y_train, y_val, y_test = split_data_chronological(X, y, timestamps)

    pipeline = joblib.load(MODEL_PATH)
    run_honest_financial_simulation(
        pipeline, X_val, y_val, X_test, y_test, df_customers, test_indices
    )


if __name__ == "__main__":
    main()