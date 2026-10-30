import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Headless rendering backend
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


def plot_unit_economics_and_payments(df_orders: pd.DataFrame) -> None:
    """
    Analyzes ticket size distributions, payment methods, and installment usage.
    """
    print("💳 Generating Unit Economics & Payment Behavior plots...")
    df = df_orders.copy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- SUBPLOT 1: Ticket Size Distribution (Basket Value) ---
    # Most e-commerce order values are heavily right-skewed; cap at R$ 500 for visual clarity
    capped_values = df[df["total_payment_value"] <= 500]["total_payment_value"]
    median_val = df["total_payment_value"].median()
    mean_val = df["total_payment_value"].mean()

    sns.histplot(capped_values, bins=35, kde=True, color="#2980b9", ax=ax1, alpha=0.6)
    ax1.axvline(median_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: R$ {median_val:.2f}")
    ax1.axvline(mean_val, color="#f39c12", linestyle=":", linewidth=2, label=f"Mean: R$ {mean_val:.2f}")
    ax1.set_title("Basket Size Distribution (Under R$ 500)", pad=15)
    ax1.set_xlabel("Order Payment Value (R$)", labelpad=10, fontweight="bold")
    ax1.set_ylabel("Order Frequency", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    # --- SUBPLOT 2: Payment Method Market Share ---
    # Clean up and count payment methods
    payment_counts = df["primary_payment_type"].value_counts(normalize=True) * 100
    payment_labels = [p.replace("_", " ").title() for p in payment_counts.index]
    palette_pie = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    bars = ax2.bar(payment_labels, payment_counts.values, color=palette_pie, alpha=0.85, width=0.55)
    ax2.set_title("Payment Method Share (%)", pad=15)
    ax2.set_ylabel("Share of Orders (%)", fontweight="bold")
    ax2.set_ylim(0, 85)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )

    # --- SUBPLOT 3: Installment Culture (Parcelamento) ---
    # Focus on credit card orders with 1 to 10 installments
    cc_orders = df[df["primary_payment_type"] == "credit_card"].copy()
    installment_counts = cc_orders["max_installments"].value_counts(normalize=True).sort_index()
    # Filter 1 to 10 installments (covers ~98% of cases)
    installment_counts = installment_counts.loc[1:10] * 100

    ax3.bar(installment_counts.index.astype(str), installment_counts.values, color="#34495e", alpha=0.8, width=0.6)
    ax3.set_title("Credit Card Installments Breakdown", pad=15)
    ax3.set_xlabel("Number of Installments (Parcels)", labelpad=10, fontweight="bold")
    ax3.set_ylabel("Share of Credit Card Orders (%)", fontweight="bold")
    ax3.set_ylim(0, 60)

    # Highlight 1x (Upfront) and 10x (Max Free Installments)
    ax3.annotate(
        f"1x (Upfront):\n{installment_counts[1]:.1f}%",
        xy=(0, installment_counts[1]),
        xytext=(0.8, installment_counts[1] - 5),
        arrowprops=dict(facecolor="#2980b9", shrink=0.1, width=1, headwidth=5),
        fontweight="bold", fontsize=9
    )
    if 10 in installment_counts:
        ax3.annotate(
            f"10x Peak:\n{installment_counts[10]:.1f}%",
            xy=(9, installment_counts[10]),
            xytext=(7.5, installment_counts[10] + 8),
            arrowprops=dict(facecolor="#e74c3c", shrink=0.1, width=1, headwidth=5),
            fontweight="bold", fontsize=9
        )

    plt.suptitle("Olist Unit Economics: Basket Values & Financing Behavior", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()

    save_path = FIGURES_DIR / "05_unit_economics_breakdown.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {save_path}")


def plot_correlation_heatmap(df_customers: pd.DataFrame) -> None:
    """
    Computes and plots a correlation matrix between operational friction,
    customer spend, satisfaction, and churn probability.
    """
    print("\n🔥 Generating Correlation Matrix Heatmap...")
    df = df_customers.copy()

    # Select numerical features of interest
    feature_cols = [
        "total_orders",
        "total_spend",
        "avg_delivery_days",
        "max_delivery_delay",
        "has_experienced_late_delivery",
        "avg_freight_ratio",
        "avg_review_score",
        "has_negative_review",
        "avg_installments",
        "recency_days",
        "is_churned"
    ]

    # Friendly human-readable names for the plot
    renamed_cols = [
        "Total Orders",
        "Total Spend (LTV)",
        "Avg Delivery Days",
        "Max Delay Days",
        "Late Delivery Flag",
        "Freight Ratio",
        "Avg Review Score",
        "Negative Review Flag",
        "Avg Installments",
        "Recency (Days)",
        "Churn Status (1=Yes)"
    ]

    corr = df[feature_cols].corr()
    corr.columns = renamed_cols
    corr.index = renamed_cols

    # Mask the upper triangle to create a clean lower-triangle correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.6,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient (r)"}
    )

    plt.title("Correlation Matrix: Operational Friction vs. Customer Value & Churn", pad=20)
    plt.tight_layout()

    save_path = FIGURES_DIR / "06_correlation_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {save_path}")


def main():
    orders_path = Path("data/processed/orders_clean.parquet")
    customers_path = Path("data/processed/customer_features.parquet")

    if not orders_path.exists() or not customers_path.exists():
        print("❌ Required parquet files not found! Run src/run_merge_test.py first.")
        return

    print("Loading processed datasets...")
    df_orders = pd.read_parquet(orders_path)
    df_customers = pd.read_parquet(customers_path)

    plot_unit_economics_and_payments(df_orders)
    plot_correlation_heatmap(df_customers)
    print("\n🎉 Phase 2.3 Financial & Unit Economics Visualizations Generated!")


if __name__ == "__main__":
    main()