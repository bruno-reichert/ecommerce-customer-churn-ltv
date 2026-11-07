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

from src.model_pipeline import prepare_features_and_target

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/best_xgboost_churn_pipeline.joblib")


def run_financial_simulation(df_customers: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Executes an end-to-end commercial simulation:
    1. Scores all customers using the trained XGBoost model.
    2. Simulates targeted retention campaigns at optimal threshold (0.44).
    3. Calculates Gross Preserved LTV, Campaign Cost, Net Profit, and ROI by Persona.
    4. Compares ML-targeted strategy against Blanket Marketing.
    """
    print("=" * 65)
    print("💰 PHASE 5.3: COMMERCIAL RETENTION & ROI SIMULATION")
    print("=" * 65)

    df = df_customers.copy()

    # 1. Feature Extraction & Scoring
    X, y, _, _ = prepare_features_and_target(df)
    print(f"Scoring all {len(X):,} customers through XGBoost pipeline...")
    df["churn_prob"] = pipeline.predict_proba(X)[:, 1]

    # Optimal commercial threshold identified in Phase 4.4
    OPTIMAL_THRESHOLD = 0.40
    COUPON_COST = 15.00       # Cost per outreach/voucher (R$)
    WIN_BACK_RATE = 0.25      # Conservative 25% retention success rate

    # Flag for intervention
    df["trigger_campaign"] = (df["churn_prob"] >= OPTIMAL_THRESHOLD).astype(int)

    # ==========================================
    # 2. Financial Simulation by Persona
    # ==========================================
    print("\n--- Simulating Retention Impact by Customer Persona ---")
    
    persona_financials = []
    for persona, group in df.groupby("persona"):
        total_in_persona = len(group)
        targeted = group["trigger_campaign"].sum()
        avg_ltv = group["total_spend"].mean()

        campaign_cost = targeted * COUPON_COST
        saved_customers = targeted * WIN_BACK_RATE
        gross_preserved_ltv = saved_customers * avg_ltv
        net_profit = gross_preserved_ltv - campaign_cost
        roi_pct = (net_profit / campaign_cost * 100) if campaign_cost > 0 else 0.0

        persona_financials.append({
            "Persona": persona,
            "Total Customers": total_in_persona,
            "Targeted (At-Risk)": targeted,
            "Targeted %": (targeted / total_in_persona) * 100,
            "Avg LTV (R$)": avg_ltv,
            "Campaign Cost (R$)": campaign_cost,
            "Gross Preserved (R$)": gross_preserved_ltv,
            "Net Profit (R$)": net_profit,
            "ROI (%)": roi_pct
        })

    fin_df = pd.DataFrame(persona_financials)

    print("\n📊 COMMERCIAL PERFORMANCE BY PERSONA:")
    display_df = fin_df.copy()
    display_df["Avg LTV (R$)"] = display_df["Avg LTV (R$)"].round(2)
    display_df["Campaign Cost (R$)"] = display_df["Campaign Cost (R$)"].apply(lambda x: f"R$ {x:,.0f}")
    display_df["Gross Preserved (R$)"] = display_df["Gross Preserved (R$)"].apply(lambda x: f"R$ {x:,.0f}")
    display_df["Net Profit (R$)"] = display_df["Net Profit (R$)"].apply(lambda x: f"R$ {x:,.0f}")
    display_df["ROI (%)"] = display_df["ROI (%)"].round(1).astype(str) + "%"
    display_df["Targeted %"] = display_df["Targeted %"].round(1).astype(str) + "%"

    print(display_df[["Persona", "Total Customers", "Targeted (At-Risk)", "Targeted %", "Avg LTV (R$)", "Net Profit (R$)", "ROI (%)"]].to_string(index=False))

    # ==========================================
    # 3. Macro Strategy Comparison (The 3 Scenarios)
    # ==========================================
    print("\n" + "=" * 65)
    print("🥊 MACRO STRATEGY COMPARISON: BLANKET vs. ML-TARGETED")
    print("=" * 65)

    # Strategy B: Blanket Marketing (Send coupon to all 93k customers)
    blanket_targeted = len(df)
    blanket_cost = blanket_targeted * COUPON_COST
    # Only actually churning customers can be saved (55,232 churners)
    actual_churners = df["is_churned"].sum()
    blanket_gross = actual_churners * WIN_BACK_RATE * df["total_spend"].mean()
    blanket_net = blanket_gross - blanket_cost
    blanket_roi = (blanket_net / blanket_cost) * 100

    # Strategy C: ML-Targeted Retention (Our model)
    ml_targeted = df["trigger_campaign"].sum()
    ml_cost = fin_df["Campaign Cost (R$)"].sum()
    ml_gross = fin_df["Gross Preserved (R$)"].sum()
    ml_net = fin_df["Net Profit (R$)"].sum()
    ml_roi = (ml_net / ml_cost) * 100

    strategy_comparison = pd.DataFrame([
        {
            "Strategy": "1. Status Quo (Do Nothing)",
            "Customers Targeted": 0,
            "Campaign Spend": "R$ 0",
            "Gross Revenue Preserved": "R$ 0",
            "Net Profit Preserved": "R$ 0",
            "ROI": "0.0%"
        },
        {
            "Strategy": "2. Blanket Spray-and-Pray (All Users)",
            "Customers Targeted": blanket_targeted,
            "Campaign Spend": f"R$ {blanket_cost:,.0f}",
            "Gross Revenue Preserved": f"R$ {blanket_gross:,.0f}",
            "Net Profit Preserved": f"R$ {blanket_net:,.0f}",
            "ROI": f"{blanket_roi:.1f}%"
        },
        {
            "Strategy": "3. ML-Targeted Precision (XGBoost @ 0.44)",
            "Customers Targeted": ml_targeted,
            "Campaign Spend": f"R$ {ml_cost:,.0f}",
            "Gross Revenue Preserved": f"R$ {ml_gross:,.0f}",
            "Net Profit Preserved": f"R$ {ml_net:,.0f}",
            "ROI": f"{ml_roi:.1f}%"
        }
    ])

    print(strategy_comparison.to_string(index=False))

    # ==========================================
    # 4. Generate Visual Financial Dashboard
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Net Profit by Persona
    colors = ["#2ecc71", "#2980b9", "#f39c12", "#e74c3c"]
    bars1 = ax1.bar(fin_df["Persona"], fin_df["Net Profit (R$)"], color=colors, alpha=0.85, width=0.55)
    ax1.set_title("Net Revenue Preserved by Customer Persona (R$)", pad=15)
    ax1.set_ylabel("Net Profit Preserved (R$)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"R$ {h:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )

    # Subplot 2: ROI % by Persona
    bars2 = ax2.bar(fin_df["Persona"], fin_df["ROI (%)"], color=colors, alpha=0.85, width=0.55)
    ax2.set_title("Campaign Return on Investment (ROI %) by Persona", pad=15)
    ax2.set_ylabel("ROI (%)", fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )

    plt.suptitle("Commercial Impact: Monetizing Churn Mitigation with Machine Learning", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "15_financial_simulation_roi.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n💾 Saved Executive Financial Dashboard: {save_path}")

    return fin_df


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}!")
        return

    customers_path = Path("data/processed/customer_features.parquet")
    df_customers = pd.read_parquet(customers_path)

    print(f"Loading champion model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)

    run_financial_simulation(df_customers, pipeline)


if __name__ == "__main__":
    main()