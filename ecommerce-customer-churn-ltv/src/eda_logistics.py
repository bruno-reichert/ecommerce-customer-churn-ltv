import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Force headless rendering backend
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_delivery_delay_vs_satisfaction(df_orders: pd.DataFrame) -> None:
    """
    Analyzes the relationship between delivery punctuality and review scores.
    Categorizes delivery performance into SLA buckets and evaluates customer backlash.
    """
    print("📦 Generating Delivery Delay vs. Review Score plot...")
    df = df_orders.copy()

    # Define operational SLA buckets based on delay relative to estimated date
    def categorize_delivery(delay_days: float) -> str:
        if delay_days <= -5:
            return "1. > 5 Days Early"
        elif delay_days <= 0:
            return "2. On Time (0-5d Early)"
        elif delay_days <= 3:
            return "3. 1-3 Days Late"
        elif delay_days <= 7:
            return "4. 4-7 Days Late"
        else:
            return "5. > 7 Days Late"

    df["delivery_bucket"] = df["delivery_delay_days"].apply(categorize_delivery)

    # Calculate average review score and 1-star percentage per bucket
    delay_metrics = df.groupby("delivery_bucket").agg(
        order_count=("order_id", "count"),
        mean_review_score=("review_score", "mean"),
        one_star_pct=("review_score", lambda x: (x == 1.0).mean() * 100)
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for Average Review Score
    bar_colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#c0392b"]
    bars = ax1.bar(
        delay_metrics["delivery_bucket"],
        delay_metrics["mean_review_score"],
        color=bar_colors,
        alpha=0.85,
        width=0.55
    )
    ax1.set_ylabel("Average Review Score (1 to 5 Stars)", color="#2c3e50", fontweight="bold")
    ax1.set_ylim(0, 5.5)
    ax1.set_xlabel("Delivery Performance vs. Estimated Date", labelpad=12, fontweight="bold")

    # Add direct score labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.2f} ★",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold",
            fontsize=11
        )

    # Twin axis for 1-star review share percentage
    ax2 = ax1.twinx()
    ax2.plot(
        delay_metrics["delivery_bucket"],
        delay_metrics["one_star_pct"],
        color="#7f8c8d",
        linewidth=2.5,
        marker="s",
        markersize=7,
        linestyle="--",
        label="% 1-Star Reviews"
    )
    ax2.set_ylabel("Share of 1-Star Reviews (%)", color="#7f8c8d", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    # Label the 1-star percentages
    for i, row in delay_metrics.iterrows():
        ax2.annotate(
            f"{row['one_star_pct']:.1f}%",
            xy=(row["delivery_bucket"], row["one_star_pct"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#2c3e50",
            fontweight="bold",
            fontsize=10
        )

    plt.title("The Logistics Cliff: Impact of Delivery Delays on Customer Satisfaction", pad=20)
    fig.tight_layout()

    save_path = FIGURES_DIR / "03_delay_vs_review_score.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved figure: {save_path}")


def plot_regional_logistics_disparities(df_orders: pd.DataFrame) -> None:
    """
    Examines regional disparities in delivery speed and freight costs
    across Brazilian Macro-Regions.
    """
    print("\n🌎 Generating Regional Logistics Disparities plot...")
    df = df_orders.copy()

    # Map Brazilian States (UF) to official Macro-Regions
    region_map = {
        # Southeast (Sudeste)
        "SP": "Southeast", "RJ": "Southeast", "MG": "Southeast", "ES": "Southeast",
        # South (Sul)
        "RS": "South", "PR": "South", "SC": "South",
        # Midwest (Centro-Oeste)
        "DF": "Midwest", "GO": "Midwest", "MT": "Midwest", "MS": "Midwest",
        # Northeast (Nordeste)
        "BA": "Northeast", "PE": "Northeast", "CE": "Northeast", "MA": "Northeast",
        "PB": "Northeast", "RN": "Northeast", "AL": "Northeast", "PI": "Northeast", "SE": "Northeast",
        # North (Norte)
        "AM": "North", "PA": "North", "RO": "North", "TO": "North", "AC": "North", "AP": "North", "RR": "North"
    }

    df["macro_region"] = df["customer_state"].map(region_map).fillna("Unknown")

    # Group by region
    regional_stats = df.groupby("macro_region").agg(
        order_count=("order_id", "count"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        avg_freight_value=("total_freight_value", "mean"),
        avg_review_score=("review_score", "mean")
    ).loc[["Southeast", "South", "Midwest", "Northeast", "North"]].reset_index()

    # Plot 2 subplots side by side: Transit Days & Freight Cost
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    palette = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    # 1. Average Delivery Days
    bars1 = ax1.bar(
        regional_stats["macro_region"],
        regional_stats["avg_delivery_days"],
        color=palette,
        alpha=0.85,
        width=0.55
    )
    ax1.set_title("Average Delivery Time by Region (Days)", pad=15)
    ax1.set_ylabel("Lead Time (Days)", fontweight="bold")
    ax1.set_ylim(0, 32)
    ax1.tick_params(axis="x", rotation=20)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}d",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )

    # 2. Average Freight Cost (R$)
    bars2 = ax2.bar(
        regional_stats["macro_region"],
        regional_stats["avg_freight_value"],
        color=palette,
        alpha=0.85,
        width=0.55
    )
    ax2.set_title("Average Freight Cost by Region (R$)", pad=15)
    ax2.set_ylabel("Freight (R$)", fontweight="bold")
    ax2.set_ylim(0, 50)
    ax2.tick_params(axis="x", rotation=20)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"R$ {h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )

    plt.suptitle("Brazilian Regional Disparities: Delivery Speed & Logistics Surcharge", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "04_regional_logistics_disparities.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {save_path}")


def main():
    orders_path = Path("data/processed/orders_clean.parquet")
    if not orders_path.exists():
        print(f"❌ File not found at {orders_path}!")
        return

    print(f"Loading {orders_path}...")
    df_orders = pd.read_parquet(orders_path)

    plot_delivery_delay_vs_satisfaction(df_orders)
    plot_regional_logistics_disparities(df_orders)
    print("\n🎉 Phase 2.2 Logistics Visualizations Generated Successfully!")


if __name__ == "__main__":
    main()