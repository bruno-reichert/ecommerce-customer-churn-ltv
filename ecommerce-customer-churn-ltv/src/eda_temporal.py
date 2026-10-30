import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# 1. Force the headless, file-only rendering backend (no Tkinter required)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_monthly_growth_and_gmv(df_orders: pd.DataFrame) -> None:
    """
    Analyzes monthly order volume and Gross Merchandise Value (GMV).
    Highlights seasonality, including the massive Black Friday 2017 anomaly.
    """
    print("📈 Generating Monthly Growth & GMV plot...")

    df = df_orders.copy()
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    
    monthly_stats = df.groupby("order_month").agg(
        order_count=("order_id", "count"),
        total_gmv=("total_payment_value", "sum")
    ).reset_index()

    monthly_stats["month_str"] = monthly_stats["order_month"].astype(str)
    monthly_stats = monthly_stats[
        (monthly_stats["month_str"] >= "2017-01") & 
        (monthly_stats["month_str"] <= "2018-08")
    ].reset_index(drop=True)

    monthly_stats["gmv_thousands"] = monthly_stats["total_gmv"] / 1000.0

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Bar chart for Order Volume
    color_bar = "#3498db"
    ax1.bar(
        monthly_stats["month_str"], 
        monthly_stats["order_count"], 
        color=color_bar, 
        alpha=0.6, 
        width=0.55,
        label="Monthly Orders"
    )
    ax1.set_xlabel("Purchase Month", labelpad=12, fontweight="bold")
    ax1.set_ylabel("Order Count", color="#2980b9", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.tick_params(axis="x", rotation=45)
    
    # Headroom to keep annotations below the title
    ax1.set_ylim(0, 9000)
    ax1.grid(False)

    # Twin axis for GMV line
    ax2 = ax1.twinx()
    color_line = "#e74c3c"
    ax2.plot(
        monthly_stats["month_str"], 
        monthly_stats["gmv_thousands"], 
        color=color_line, 
        linewidth=2.5, 
        marker="o", 
        markersize=6,
        label="GMV (R$ Thousands)"
    )
    ax2.set_ylabel("Gross Merchandise Value (R$ Thousands)", color=color_line, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, 1450) # Matching proportional headroom
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Annotate Black Friday Spike (November 2017)
    bf_idx = monthly_stats[monthly_stats["month_str"] == "2017-11"].index[0]
    bf_orders = monthly_stats.loc[bf_idx, "order_count"]
    bf_gmv = monthly_stats.loc[bf_idx, "gmv_thousands"]
    
    ax1.annotate(
        f"Black Friday 2017\n{bf_orders:,} Orders\n(R$ {bf_gmv:,.0f}k)",
        xy=(bf_idx, bf_orders),
        xytext=(bf_idx - 2.0, bf_orders + 700),
        arrowprops=dict(facecolor="#2c3e50", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1c40f", alpha=0.8, edgecolor="#d4ac0d")
    )

    plt.title("Olist Monthly Order Growth & Gross Merchandise Value (2017 - 2018)", pad=28)
    fig.tight_layout()
    
    save_path = FIGURES_DIR / "01_monthly_gmv_orders.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved figure: {save_path}")


def plot_cohort_retention_heatmap(df_orders: pd.DataFrame) -> None:
    """
    Computes and plots an e-commerce Cohort Retention Heatmap.
    Tracks monthly cohorts from first purchase to month n.
    """
    print("\n📊 Computing Cohort Retention Matrix...")
    df = df_orders[["customer_unique_id", "order_purchase_timestamp"]].copy()
    
    # 1. Determine order month
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")

    # 2. Determine cohort month (customer's first purchase)
    df["cohort_month"] = df.groupby("customer_unique_id")["order_purchase_timestamp"].transform("min").dt.to_period("M")

    # Filter to main operational window (Jan 2017 to Dec 2017) to observe long maturation
    df = df[(df["cohort_month"] >= "2017-01") & (df["cohort_month"] <= "2017-12")]

    # 3. Calculate time offset (period distance in months)
    df["cohort_index"] = (df["order_month"].astype(int) - df["cohort_month"].astype(int))

    # 4. Group by cohort and index to count unique active customers
    cohort_data = df.groupby(["cohort_month", "cohort_index"])["customer_unique_id"].nunique().reset_index()

    # Pivot into a matrix: Cohort Month vs. Month Offset
    cohort_matrix = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_unique_id")

    # 5. Calculate retention percentages (Month 0 = 100%)
    cohort_sizes = cohort_matrix.iloc[:, 0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100

    # 6. Plot Heatmap (Excluding month 0 which is trivially 100%)
    plt.figure(figsize=(14, 8))
    
    # Mask month 0 for visualization contrast since subsequent months are < 2%
    display_matrix = retention_matrix.iloc[:, 1:13] # Next 12 months

    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Retention Rate (%)"},
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0 # Scale focused on 0% - 1% to highlight subtle repeat buying
    )

    plt.title("Olist Monthly Cohort Retention Matrix (2017 Cohorts)", pad=18)
    plt.xlabel("Months After First Purchase (Maturation)", labelpad=12, fontweight="bold")
    plt.ylabel("Acquisition Cohort Month", labelpad=12, fontweight="bold")
    plt.tight_layout()

    save_path = FIGURES_DIR / "02_cohort_retention_heatmap.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved figure: {save_path}")


def main():
    orders_path = Path("data/processed/orders_clean.parquet")
    if not orders_path.exists():
        print(f"❌ Clean orders file not found at {orders_path}. Run src/run_merge_test.py first!")
        return

    print(f"Loading {orders_path} for temporal EDA...")
    df_orders = pd.read_parquet(orders_path)
    
    plot_monthly_growth_and_gmv(df_orders)
    plot_cohort_retention_heatmap(df_orders)
    print("\n🎉 Phase 2.1 Visualizations Generated Successfully in reports/figures/!")


if __name__ == "__main__":
    main()