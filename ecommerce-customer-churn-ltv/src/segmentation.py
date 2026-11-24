"""
Customer Segmentation & RFM Profiling Module.
DESCRIPTIVE / EXPLORATORY ONLY:
Personas are derived from Recency, Frequency, and Monetary spend to help
commercial teams organize marketing campaigns. They are strictly descriptive
and are NEVER fed as predictive features into supervised churn classifiers.
"""

import sys
from pathlib import Path

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Headless rendering backend
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_PROCESSED, FIGURES_DIR, RANDOM_SEED

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def extract_rfm_table(df_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts core RFM dimensions from the customer analytical dataset
    solely for exploratory clustering.
    """
    print("\n--- 1. Extracting Core RFM Metrics (Descriptive Segmentation) ---")
    rfm = df_customers[[
        "customer_unique_id",
        "recency_days",
        "total_orders",
        "total_spend"
    ]].copy()

    rfm = rfm.rename(columns={
        "recency_days": "recency",
        "total_orders": "frequency",
        "total_spend": "monetary"
    })
    print(f"Extracted RFM records for {len(rfm):,} unique customers.")
    return rfm


def audit_rfm_distributions(rfm: pd.DataFrame) -> pd.DataFrame:
    """Computes percentiles and skewness for RFM metrics."""
    print("\n--- 2. RFM Statistical Distribution & Skewness Audit ---")
    metrics = ["recency", "frequency", "monetary"]
    stats_df = rfm[metrics].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
    stats_df["skewness"] = rfm[metrics].skew()
    print(stats_df[["mean", "std", "min", "50%", "75%", "90%", "99%", "max", "skewness"]])
    return stats_df


def preprocess_rfm_for_clustering(rfm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Transforms RFM features for K-Means:
    1. Log-transformation (np.log1p) to normalize heavy monetary/recency tails.
    2. StandardScaler to bring all 3 dimensions to mean=0, variance=1.
    """
    print("\n--- 3. Preprocessing RFM Features for K-Means Clustering ---")
    df = rfm.copy()

    df["log_recency"] = np.log1p(df["recency"])
    df["log_frequency"] = np.log1p(df["frequency"])
    df["log_monetary"] = np.log1p(df["monetary"])

    raw_skew = rfm[["recency", "frequency", "monetary"]].skew()
    log_skew = df[["log_recency", "log_frequency", "log_monetary"]].skew()

    print("\n📊 SKEWNESS REDUCTION COMPARISON:")
    print(f"{'Feature':<12} | {'Raw Skewness':<14} | {'Log-Transformed Skewness':<25}")
    print("-" * 58)
    print(f"{'Recency':<12} | {raw_skew['recency']:<14.2f} | {log_skew['log_recency']:<25.2f}")
    print(f"{'Frequency':<12} | {raw_skew['frequency']:<14.2f} | {log_skew['log_frequency']:<25.2f}")
    print(f"{'Monetary':<12} | {raw_skew['monetary']:<14.2f} | {log_skew['log_monetary']:<25.2f}")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["log_recency", "log_frequency", "log_monetary"]])

    # Diagnostic plot: Raw vs. Log-Transformed
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    sns.histplot(rfm["recency"], bins=30, kde=True, ax=axes[0, 0], color="#3498db")
    axes[0, 0].set_title("Raw Recency (Days)")
    sns.histplot(rfm["frequency"], bins=15, ax=axes[0, 1], color="#e67e22")
    axes[0, 1].set_title("Raw Frequency (Orders)")
    sns.histplot(rfm["monetary"].clip(upper=1000), bins=30, kde=True, ax=axes[0, 2], color="#2ecc71")
    axes[0, 2].set_title("Raw Monetary (Capped R$ 1,000)")

    sns.histplot(df["log_recency"], bins=30, kde=True, ax=axes[1, 0], color="#2980b9")
    axes[1, 0].set_title("Log-Transformed Recency [ln(1+R)]")
    sns.histplot(df["log_frequency"], bins=15, ax=axes[1, 1], color="#d35400")
    axes[1, 1].set_title("Log-Transformed Frequency [ln(1+F)]")
    sns.histplot(df["log_monetary"], bins=30, kde=True, ax=axes[1, 2], color="#27ae60")
    axes[1, 2].set_title("Log-Transformed Monetary [ln(1+M)]")

    plt.suptitle("RFM Feature Distributions: Raw vs. Log-Transformed (Normalizing Tails)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_path = FIGURES_DIR / "07_rfm_transformation_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved diagnostic plot: {save_path}")

    return df, scaled_features, scaler


def assign_and_profile_personas(
    rfm: pd.DataFrame, 
    scaled_matrix: np.ndarray, 
    df_customers: pd.DataFrame, 
    k: int = 4
) -> pd.DataFrame:
    """
    Fits K-Means model (k=4) to produce descriptive customer personas for business reporting.
    """
    print(f"\n--- 4. Fitting Descriptive K-Means Model (k={k}) ---")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=15)
    rfm["cluster"] = kmeans.fit_predict(scaled_matrix)

    cluster_profiles = rfm.groupby("cluster").agg(
        customer_count=("customer_unique_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
        total_revenue=("monetary", "sum")
    ).reset_index()

    cluster_profiles["customer_pct"] = (cluster_profiles["customer_count"] / len(rfm)) * 100
    cluster_profiles["revenue_pct"] = (cluster_profiles["total_revenue"] / rfm["monetary"].sum()) * 100

    print("\n📋 CLUSTER CENTROID PROFILES (Real-World Averages):")
    print(cluster_profiles.to_string(index=False))

    # Descriptive Persona Mapping based on Spend and Recency centroids
    sorted_by_spend = cluster_profiles.sort_values(by="avg_monetary", ascending=False)["cluster"].tolist()
    persona_map = {
        sorted_by_spend[0]: "Champions / Loyal VIPs",
        sorted_by_spend[1]: "At-Risk High Spenders",
        sorted_by_spend[2]: "Active / Recent Buyers",
        sorted_by_spend[3]: "Lost / Hibernating Shoppers"
    }

    rfm["persona"] = rfm["cluster"].map(persona_map)
    df_customers["cluster"] = rfm["cluster"]
    df_customers["persona"] = rfm["persona"]

    final_summary = rfm.groupby("persona").agg(
        customers=("customer_unique_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        revenue=("monetary", "sum")
    ).reset_index()
    final_summary["cust_share"] = (final_summary["customers"] / len(rfm)) * 100
    final_summary["rev_share"] = (final_summary["revenue"] / rfm["monetary"].sum()) * 100

    print("\n🏷️ DESCRIPTIVE COMMERCIAL PERSONA PROFILES:")
    print(final_summary[["persona", "customers", "cust_share", "recency", "frequency", "monetary", "rev_share"]].to_string(index=False))

    # --- PLOT PERSONA DASHBOARD ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    palette = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    shares_df = final_summary.melt(id_vars="persona", value_vars=["cust_share", "rev_share"], var_name="Metric", value_name="Percentage")
    shares_df["Metric"] = shares_df["Metric"].map({"cust_share": "% of Customers", "rev_share": "% of Revenue"})
    sns.barplot(data=shares_df, x="persona", y="Percentage", hue="Metric", ax=ax1, palette=["#34495e", "#f39c12"])
    ax1.set_title("Customer Share vs. Revenue Contribution", pad=15)
    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    sns.barplot(data=final_summary, x="persona", y="recency", ax=ax2, palette=palette)
    ax2.set_title("Average Recency (Days Since Last Order)", pad=15)
    ax2.set_ylabel("Days", fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    sns.barplot(data=final_summary, x="persona", y="monetary", ax=ax3, palette=palette)
    ax3.set_title("Average Spend (LTV in R$)", pad=15)
    ax3.set_ylabel("Spend (R$)", fontweight="bold")
    ax3.tick_params(axis="x", rotation=15)

    sns.barplot(data=final_summary, x="persona", y="frequency", ax=ax4, palette=palette)
    ax4.set_title("Average Lifetime Orders", pad=15)
    ax4.set_ylabel("Orders Count", fontweight="bold")
    ax4.tick_params(axis="x", rotation=15)

    plt.suptitle("Descriptive Customer Personas: 4 Strategic RFM Clusters", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "09_rfm_customer_personas.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n💾 Saved Persona Dashboard: {save_path}")

    # Re-save customer_features.parquet with cluster & persona columns attached
    export_path = DATA_PROCESSED / "customer_features.parquet"
    df_customers.to_parquet(export_path, index=False)
    print(f"💾 Updated customer dataset with descriptive personas: {export_path}")

    return df_customers


def main():
    customers_path = DATA_PROCESSED / "customer_features.parquet"
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)
    rfm = extract_rfm_table(df_customers)
    audit_rfm_distributions(rfm)
    _df_transformed, scaled_matrix, _scaler = preprocess_rfm_for_clustering(rfm)
    assign_and_profile_personas(rfm, scaled_matrix, df_customers, k=4)


if __name__ == "__main__":
    main()