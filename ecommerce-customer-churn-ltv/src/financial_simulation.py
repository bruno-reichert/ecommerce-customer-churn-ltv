"""
Commercial Retention & Illustrative Scenario Simulation Module.
Evaluates campaign economics strictly on held-out Snapshot 2 (Test: T0 = 2018-03-01).
Implements:
1. Top-k% ML Targeting vs. Random Targeting at the exact same N (Incremental Value).
2. Blanket Marketing vs. Status Quo.
3. Break-Even Win-Back Rate Table across outreach costs.
4. Descriptive Persona Payoff Breakdown on the test cohort.
5. 2-Panel Executive Dashboard (Incremental Value & Break-Even Curve).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

# Headless rendering backend
import matplotlib
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
)
from src.model_pipeline import (
    load_out_of_time_splits,
    prepare_features_and_target,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_xgboost_churn_pipeline.joblib"


def run_honest_scenario_simulation(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_full: pd.DataFrame
) -> Dict[str, Any]:
    """
    Simulates commercial retention strategies on Snapshot 2 (N = 55,525).
    Evaluates Top-10% and Top-20% ML targeting against Random Selection at identical N.
    """
    print("=" * 75)
    print("💰 COMMERCIAL RETENTION SCENARIO SIMULATION (Out-of-Time Snapshot 2)")
    print("=" * 75)

    n_total = len(df_test_full)
    # Predict continuous probability of repurchase (rare event)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    sim_df = df_test_full.copy().reset_index(drop=True)
    sim_df["repurchase_prob"] = y_prob
    sim_df["y_true"] = y_test.values

    # Sort descending by model score
    sim_df = sim_df.sort_values(by="repurchase_prob", ascending=False).reset_index(drop=True)

    # Total pool stats
    total_spend_potential = (sim_df["y_true"] * sim_df["total_spend"]).sum()
    total_repurchasers = int(sim_df["y_true"].sum())
    prevalence = total_repurchasers / n_total

    print(f"Test Cohort Size:            {n_total:,}")
    print(f"Total True Repurchasers:     {total_repurchasers:,} ({prevalence*100:.2f}%)")
    print(f"Gross At-Risk Spend Pool:    R$ {total_spend_potential:,.2f}")

    # ==========================================
    # 1. Macro Strategies Scoreboard (Top-k% vs. Random at Same N)
    # ==========================================
    tiers = [
        ("Top 10% ML-Targeted", 0.10),
        ("Top 20% ML-Targeted", 0.20),
        ("Blanket Marketing (100%)", 1.00)
    ]

    strategy_rows = []
    # Status Quo
    strategy_rows.append({
        "Strategy": "1. Status Quo (Do Nothing)",
        "Customers Targeted": 0,
        "Targeted %": "0.0%",
        "Campaign Cost (R$)": 0.0,
        "Gross Preserved (R$)": 0.0,
        "Net Profit (R$)": 0.0,
        "ROI (%)": 0.0,
        "Incremental vs Random (R$)": 0.0
    })

    for label, pct in tiers:
        k = int(n_total * pct)
        top_slice = sim_df.iloc[:k]

        # ML Campaign Economics
        ml_cost = k * COUPON_COST
        ml_saved_spend = (top_slice["y_true"] * top_slice["total_spend"]).sum()
        ml_gross = ml_saved_spend * BASE_WIN_BACK_RATE
        ml_net = ml_gross - ml_cost
        ml_roi = (ml_net / ml_cost * 100.0) if ml_cost > 0 else 0.0

        # Random Selection Baseline at the EXACT SAME N (k customers)
        random_expected_gross = (k / n_total) * total_spend_potential * BASE_WIN_BACK_RATE
        random_net = random_expected_gross - ml_cost
        incremental_net = ml_net - random_net if pct < 1.0 else 0.0

        strategy_rows.append({
            "Strategy": label,
            "Customers Targeted": k,
            "Targeted %": f"{pct*100:.0f}%",
            "Campaign Cost (R$)": round(ml_cost, 2),
            "Gross Preserved (R$)": round(ml_gross, 2),
            "Net Profit (R$)": round(ml_net, 2),
            "ROI (%)": round(ml_roi, 1),
            "Incremental vs Random (R$)": round(incremental_net, 2)
        })

    strat_df = pd.DataFrame(strategy_rows)
    print("\n📊 STRATEGY SCOREBOARD (Assumptions: R$ 15.00 voucher, 25% win-back):")
    display_strat = strat_df.copy()
    for col in ["Campaign Cost (R$)", "Gross Preserved (R$)", "Net Profit (R$)", "Incremental vs Random (R$)"]:
        display_strat[col] = display_strat[col].apply(lambda x: f"R$ {x:,.2f}")
    display_strat["ROI (%)"] = display_strat["ROI (%)"].astype(str) + "%"
    print(display_strat.to_string(index=False))

    # ==========================================
    # 2. Break-Even Win-Back Rate Table
    # ==========================================
    print("\n--- Break-Even Analysis (Win-Back Rate Required for Net Zero Profit) ---")
    breakeven_rows = []
    top_10k = int(n_total * 0.10)
    top_10_slice = sim_df.iloc[:top_10k]
    top_10_spend_pool = (top_10_slice["y_true"] * top_10_slice["total_spend"]).sum()

    for cost in SENSITIVITY_COUPON_COSTS:
        # Break-Even Rate = Total Cost / Target Spend Pool
        be_blanket = (n_total * cost) / total_spend_potential if total_spend_potential > 0 else 1.0
        be_top10 = (top_10k * cost) / top_10_spend_pool if top_10_spend_pool > 0 else 1.0

        breakeven_rows.append({
            "Voucher Unit Cost": f"R$ {cost:.2f}",
            "Blanket Break-Even Win-Back": f"{be_blanket*100:.2f}%",
            "Top 10% ML Break-Even Win-Back": f"{be_top10*100:.2f}%",
            "Natural Prevalence Rate": f"{prevalence*100:.2f}%"
        })

    be_df = pd.DataFrame(breakeven_rows)
    print("\n📋 BREAK-EVEN WIN-BACK TABLE:")
    print(be_df.to_string(index=False))

    # ==========================================
    # 3. Descriptive Persona Payoff (Test Snapshot)
    # ==========================================
    persona_rows = []
    if "persona" in top_10_slice.columns:
        print("\n--- Descriptive Persona Retention Breakdown (Top 10% Targeted Slice) ---")
        for persona, group in top_10_slice.groupby("persona"):
            n_p = len(group)
            p_cost = n_p * COUPON_COST
            p_saved_spend = (group["y_true"] * group["total_spend"]).sum()
            p_gross = p_saved_spend * BASE_WIN_BACK_RATE
            p_net = p_gross - p_cost
            p_roi = (p_net / p_cost * 100.0) if p_cost > 0 else 0.0
            persona_rows.append({
                "Persona": persona,
                "Targeted in Top 10%": n_p,
                "Targeted Share %": (n_p / top_10k) * 100.0,
                "Spend Preserved (R$)": round(p_gross, 2),
                "Net Profit (R$)": round(p_net, 2),
                "ROI (%)": round(p_roi, 1)
             })

        persona_df = pd.DataFrame(persona_rows).sort_values(by="Net Profit (R$)", ascending=False)
        print(persona_df.to_string(index=False))
    else:
        print("\n⚠️ Persona column not present in snapshot. Run src/segmentation.py to populate personas.")
    # ==========================================
    # 4. Generate 2-Panel Executive Figure 15
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: ML Net Profit vs. Random Targeting Baseline
    x_labels = ["Top 10%", "Top 20%"]
    ml_nets = [strat_df.loc[1, "Net Profit (R$)"], strat_df.loc[2, "Net Profit (R$)"]]
    rand_nets = [
        (0.10 * total_spend_potential * BASE_WIN_BACK_RATE) - (0.10 * n_total * COUPON_COST),
        (0.20 * total_spend_potential * BASE_WIN_BACK_RATE) - (0.20 * n_total * COUPON_COST)
    ]

    x = np.arange(len(x_labels))
    width = 0.35
    ax1.bar(x - width/2, ml_nets, width, label="Tuned ML Targeting", color="#2980b9")
    ax1.bar(x + width/2, rand_nets, width, label="Random Targeting (Same N)", color="#7f8c8d", alpha=0.7)

    ax1.set_title("Net Preserved Value: Model vs. Random Control Baseline", pad=15, fontweight="bold")
    ax1.set_ylabel("Net Preserved Profit (R$)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Break-Even Win-Back Rate Comparison
    be_costs = [10.0, 15.0, 20.0]
    be_top10_vals = [((top_10k * c) / top_10_spend_pool) * 100 for c in be_costs]
    be_blanket_vals = [((n_total * c) / total_spend_potential) * 100 for c in be_costs]

    ax2.plot(be_costs, be_top10_vals, marker="o", linewidth=2.5, color="#27ae60", label="Top 10% ML Break-Even %")
    ax2.plot(be_costs, be_blanket_vals, marker="s", linewidth=2.5, color="#e74c3c", linestyle="--", label="Blanket Break-Even %")
    ax2.axhline(prevalence * 100, color="#2c3e50", linestyle=":", label=f"Natural Baseline ({prevalence*100:.2f}%)")

    ax2.set_title("Break-Even Win-Back Rate Required vs. Voucher Cost", pad=15, fontweight="bold")
    ax2.set_xlabel("Voucher Unit Cost (R$)", fontweight="bold")
    ax2.set_ylabel("Required Win-Back Rate (%)", fontweight="bold")
    ax2.set_xticks(be_costs)
    ax2.set_xticklabels([f"R$ {c:.2f}" for c in be_costs])
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Commercial Retention Simulation: Incremental Lift & Scenario Sensitivity", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "15_financial_simulation_roi.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n💾 Saved updated scenario dashboard: {save_path}")

    # ==========================================
    # 5. Export to reports/metrics.json
    # ==========================================
    metrics_export = {
        "dataset_snapshot_2_test": {
            "total_customers": n_total,
            "repurchased_count": total_repurchasers,
            "prevalence_rate_pct": round(prevalence * 100, 2),
            "at_risk_spend_pool_brl": round(total_spend_potential, 2)
        },
        "simulation_strategies": strategy_rows,
        "breakeven_analysis": breakeven_rows,
        "persona_payoff_top10": persona_rows
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"💾 Updated metrics repository: {METRICS_JSON_PATH}")

    return metrics_export


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model artifact not found at {MODEL_PATH}! Run src/train_xgboost.py first.")
        return

    test_snap_path = DATA_PROCESSED / "customer_features_test.parquet"
    df_test_full = pd.read_parquet(test_snap_path)

    # Attach persona from customer_features.parquet if present
    cust_path = DATA_PROCESSED / "customer_features.parquet"
    if cust_path.exists():
        cust_df = pd.read_parquet(cust_path)
        if "persona" in cust_df.columns:
            df_test_full = df_test_full.merge(
                cust_df[["customer_unique_id", "persona"]],
                on="customer_unique_id",
                how="left"
            )
            df_test_full["persona"] = df_test_full["persona"].fillna("Unassigned")

    X_test, y_test, _, _ = prepare_features_and_target(df_test_full)

    pipeline = joblib.load(MODEL_PATH)
    run_honest_scenario_simulation(pipeline, X_test, y_test, df_test_full)


if __name__ == "__main__":
    main()