import sys
from pathlib import Path
from typing import Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd
import numpy as np

# Headless rendering backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

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


def extract_rfm_table(df_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and standardizes the core RFM features from the customer analytical dataset.
    """
    print("\n--- 1. Extracting Core RFM Metrics ---")
    
    rfm = df_customers[[
        "customer_unique_id",
        "recency_days",
        "total_orders",
        "total_spend"
    ]].copy()

    # Clean rename without deprecated inplace flag
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


def preprocess_rfm_for_clustering(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Transforms RFM features for K-Means:
    1. Log-transformation (np.log1p) to compress extreme tails and normalize distributions.
    2. StandardScaler to bring all 3 dimensions to mean=0, variance=1.
    """
    print("\n--- 3. Preprocessing RFM Features for K-Means Clustering ---")
    df = rfm.copy()

    # 1. Log-Transformation: np.log1p(x) calculates ln(1 + x)
    # Adding 1 protects against log(0) since recency can be 0.0 days
    df["log_recency"] = np.log1p(df["recency"])
    df["log_frequency"] = np.log1p(df["frequency"])
    df["log_monetary"] = np.log1p(df["monetary"])

    # Audit the impact of Log Transformation on Skewness
    raw_skew = rfm[["recency", "frequency", "monetary"]].skew()
    log_skew = df[["log_recency", "log_frequency", "log_monetary"]].skew()

    print("\n📊 SKEWNESS REDUCTION COMPARISON:")
    print(f"{'Feature':<12} | {'Raw Skewness':<14} | {'Log-Transformed Skewness':<25}")
    print("-" * 58)
    print(f"{'Recency':<12} | {raw_skew['recency']:<14.2f} | {log_skew['log_recency']:<25.2f}")
    print(f"{'Frequency':<12} | {raw_skew['frequency']:<14.2f} | {log_skew['log_frequency']:<25.2f}")
    print(f"{'Monetary':<12} | {raw_skew['monetary']:<14.2f} | {log_skew['log_monetary']:<25.2f}")

    # 2. Standardization via StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["log_recency", "log_frequency", "log_monetary"]])

    print(f"\n✅ Standardized feature matrix created: {scaled_features.shape}")
    print(f"Mean across dimensions:   {scaled_features.mean(axis=0).round(4)}")
    print(f"Variance across dimensions: {scaled_features.var(axis=0).round(4)}")

    # 3. Plot Diagnostic Distributions: Raw vs. Log-Transformed
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Top Row: Raw Distributions
    sns.histplot(rfm["recency"], bins=30, kde=True, ax=axes[0, 0], color="#3498db")
    axes[0, 0].set_title("Raw Recency (Days)")
    
    sns.histplot(rfm["frequency"], bins=15, ax=axes[0, 1], color="#e67e22")
    axes[0, 1].set_title("Raw Frequency (Orders)")
    
    sns.histplot(rfm["monetary"].clip(upper=1000), bins=30, kde=True, ax=axes[0, 2], color="#2ecc71")
    axes[0, 2].set_title("Raw Monetary (Capped R$ 1,000)")

    # Bottom Row: Log-Transformed Distributions
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

def find_optimal_k(scaled_matrix: np.ndarray, k_range: range = range(2, 8)) -> None:
    """
    Evaluates K-Means clustering for k in k_range using:
    1. Elbow Method (Inertia / Within-Cluster Sum of Squares).
    2. Silhouette Score (Cluster separation & cohesion).
    """
    print("\n--- 4. Determining Optimal k (Elbow Method & Silhouette Analysis) ---")

    inertias = []
    silhouette_scores = []
    
    # Subsample 15,000 points for fast, statistically sound silhouette calculation
    np.random.seed(42)
    sample_indices = np.random.choice(len(scaled_matrix), size=15000, replace=False)
    sample_matrix = scaled_matrix[sample_indices]

    for k in k_range:
        print(f"Training K-Means with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_matrix)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score on sample
        score = silhouette_score(sample_matrix, kmeans.predict(sample_matrix))
        silhouette_scores.append(score)
        print(f"   • k={k} | Inertia: {kmeans.inertia_:,.1f} | Silhouette Score: {score:.4f}")

    # Plot Elbow Curve and Silhouette Curve side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Elbow Plot
    ax1.plot(list(k_range), inertias, marker="o", color="#2980b9", linewidth=2.5, markersize=7)
    ax1.set_title("The Elbow Method (Inertia)", pad=15)
    ax1.set_xlabel("Number of Clusters (k)", labelpad=10, fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontweight="bold")
    ax1.set_xticks(list(k_range))

    # 2. Silhouette Plot
    ax2.plot(list(k_range), silhouette_scores, marker="s", color="#27ae60", linewidth=2.5, markersize=7, linestyle="--")
    ax2.set_title("Silhouette Score vs. Number of Clusters", pad=15)
    ax2.set_xlabel("Number of Clusters (k)", labelpad=10, fontweight="bold")
    ax2.set_ylabel("Mean Silhouette Coefficient", fontweight="bold")
    ax2.set_xticks(list(k_range))

    plt.suptitle("K-Means Hyperparameter Optimization for Customer Segmentation", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()

    save_path = FIGURES_DIR / "08_kmeans_elbow_silhouette.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n💾 Saved optimization plots: {save_path}")

def assign_and_profile_personas(
    rfm: pd.DataFrame, 
    scaled_matrix: np.ndarray, 
    df_customers: pd.DataFrame, 
    k: int = 4
) -> pd.DataFrame:
    """
    Fits the final K-Means model with optimal k, profiles cluster centroids
    in real-world units, assigns business personas, and exports the enriched dataset.
    """
    print(f"\n--- 5. Fitting Final K-Means Model (k={k}) & Profiling Personas ---")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
    rfm["cluster"] = kmeans.fit_predict(scaled_matrix)

    # Calculate real-world averages per cluster
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

    print("\n📋 UNLABELED CLUSTER PROFILES (Real-World Averages):")
    print(cluster_profiles.to_string(index=False))

    # Dynamically assign personas based on Recency and Monetary profiles
    # 1. VIPs: Highest Monetary or Frequency
    # 2. At-Risk High Spenders: High Monetary, High Recency (Inactive)
    # 3. New/Active Shoppers: Low Recency, Moderate/Low Spend
    # 4. Lost/Hibernating: High Recency, Lowest Monetary Spend
    persona_map = {}
    for idx, row in cluster_profiles.iterrows():
        c_id = int(row["cluster"])
        r = row["avg_recency"]
        m = row["avg_monetary"]
        f = row["avg_frequency"]

        if f > 1.2 or (m > cluster_profiles["avg_monetary"].mean() and r < 200):
            persona_map[c_id] = "Champions / Loyal VIPs"
        elif m > cluster_profiles["avg_monetary"].mean() and r >= 200:
            persona_map[c_id] = "At-Risk High Spenders"
        elif r < cluster_profiles["avg_recency"].mean():
            persona_map[c_id] = "Active / Recent Buyers"
        else:
            persona_map[c_id] = "Lost / Hibernating Shoppers"

    # Fallback to ensure distinct naming if logic assigns duplicate names
    if len(set(persona_map.values())) < k:
        # Fallback ranking by monetary spend
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

    # Print final persona summary
    final_summary = rfm.groupby("persona").agg(
        customers=("customer_unique_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        revenue=("monetary", "sum")
    ).reset_index()
    final_summary["cust_share"] = (final_summary["customers"] / len(rfm)) * 100
    final_summary["rev_share"] = (final_summary["revenue"] / rfm["monetary"].sum()) * 100

    print("\n🏷️ FINAL COMMERCIAL PERSONA PROFILES:")
    print(final_summary[["persona", "customers", "cust_share", "recency", "frequency", "monetary", "rev_share"]].to_string(index=False))

    # --- PLOT PERSONA COMPARISON DASHBOARD ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    palette = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    # 1. Customer Share vs Revenue Share
    shares_df = final_summary.melt(id_vars="persona", value_vars=["cust_share", "rev_share"], var_name="Metric", value_name="Percentage")
    shares_df["Metric"] = shares_df["Metric"].map({"cust_share": "% of Customers", "rev_share": "% of Revenue"})
    sns.barplot(data=shares_df, x="persona", y="Percentage", hue="Metric", ax=ax1, palette=["#34495e", "#f39c12"])
    ax1.set_title("Customer Share vs. Revenue Contribution", pad=15)
    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    # 2. Recency Comparison
    sns.barplot(data=final_summary, x="persona", y="recency", ax=ax2, palette=palette)
    ax2.set_title("Average Recency (Days Since Last Order - Lower is Better)", pad=15)
    ax2.set_ylabel("Days", fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    # 3. Monetary Spend Comparison
    sns.barplot(data=final_summary, x="persona", y="monetary", ax=ax3, palette=palette)
    ax3.set_title("Average Monetary Spend (LTV in R$)", pad=15)
    ax3.set_ylabel("Spend (R$)", fontweight="bold")
    ax3.tick_params(axis="x", rotation=15)

    # 4. Frequency (Repeat Buying)
    sns.barplot(data=final_summary, x="persona", y="frequency", ax=ax4, palette=palette)
    ax4.set_title("Average Lifetime Orders (Frequency)", pad=15)
    ax4.set_ylabel("Orders Count", fontweight="bold")
    ax4.tick_params(axis="x", rotation=15)

    plt.suptitle("Olist Customer Segmentation: 4 Strategic RFM Personas", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = FIGURES_DIR / "09_rfm_customer_personas.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n💾 Saved Persona Dashboard: {save_path}")

    # Re-save customer_features.parquet with cluster & persona columns attached!
    export_path = Path("data/processed/customer_features.parquet")
    df_customers.to_parquet(export_path, index=False)
    print(f"💾 Updated customer dataset with persona columns: {export_path}")

    return df_customers

def main():
    customers_path = Path("data/processed/customer_features.parquet")
    if not customers_path.exists():
        print(f"❌ File not found at {customers_path}. Run src/run_merge_test.py first!")
        return

    print(f"Loading {customers_path}...")
    df_customers = pd.read_parquet(customers_path)

    rfm = extract_rfm_table(df_customers)
    audit_rfm_distributions(rfm)
    df_transformed, scaled_matrix, scaler = preprocess_rfm_for_clustering(rfm)
    
    # 1. Run optimization plot (can comment out if already run)
    # find_optimal_k(scaled_matrix, k_range=range(2, 8))

    # 2. Assign final personas with k=4
    assign_and_profile_personas(rfm, scaled_matrix, df_customers, k=4)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()