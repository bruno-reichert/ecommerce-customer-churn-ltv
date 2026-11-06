import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np

from src.model_pipeline import prepare_features_and_target

DATA_PROCESSED = Path("data/processed")
POWERBI_DIR = Path("data/powerbi")
POWERBI_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/best_xgboost_churn_pipeline.joblib")


def export_powerbi_star_schema():
    print("=" * 65)
    print("📊 EXPORTING POWER BI STAR SCHEMA DATA TABLES")
    print("=" * 65)

    # 1. Load Clean Parquet Datasets
    orders_path = DATA_PROCESSED / "orders_clean.parquet"
    cust_path = DATA_PROCESSED / "customer_features.parquet"

    if not orders_path.exists() or not cust_path.exists():
        print(f"❌ Parquet files not found in {DATA_PROCESSED}! Run src/run_merge_test.py first.")
        return

    print("Loading processed Parquet datasets...")
    df_orders = pd.read_parquet(orders_path)
    df_customers = pd.read_parquet(cust_path)

    # 2. Attach ML Predictions to Customers
    print(f"Loading champion XGBoost model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    
    X, _, _, _ = prepare_features_and_target(df_customers)
    print(f"Scoring all {len(X):,} customers for Power BI...")
    
    # Predict continuous churn probability and binary trigger at optimal 0.40 cutoff
    df_customers["churn_probability"] = pipeline.predict_proba(X)[:, 1].round(4)
    df_customers["trigger_retention"] = (df_customers["churn_probability"] >= 0.40).astype(int)

    # ==========================================
    # 3. Create Dimension Table: dim_customers
    # ==========================================
    print("\n--- 1. Building 'dim_customers.csv' ---")
    dim_customers = df_customers[[
        "customer_unique_id",
        "customer_state",
        "favorite_category",
        "persona",
        "is_churned",
        "churn_probability",
        "trigger_retention",
        "total_orders",
        "total_spend",
        "recency_days",
        "avg_review_score",
        "avg_delivery_days",
        "max_delivery_delay"
    ]].copy()

    # Friendly human-readable flags
    dim_customers["churn_status"] = dim_customers["is_churned"].map({1: "Churned", 0: "Active"})
    dim_customers["action_plan"] = dim_customers["trigger_retention"].map({
        1: "Send Retention Campaign (>= 40%)",
        0: "Organic Loyalty (No Action)"
    })

    cust_out = POWERBI_DIR / "dim_customers.csv"
    dim_customers.to_csv(cust_out, index=False)
    print(f"✅ Exported dim_customers: {len(dim_customers):,} rows -> {cust_out}")

    # ==========================================
    # 4. Create Fact Table: fact_orders
    # ==========================================
    print("\n--- 2. Building 'fact_orders.csv' ---")
    df_orders["order_month"] = df_orders["order_purchase_timestamp"].dt.strftime("%Y-%m")
    df_orders["order_date"] = df_orders["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")

    fact_orders = df_orders[[
        "order_id",
        "customer_unique_id",
        "order_date",
        "order_month",
        "customer_state",
        "clean_category",
        "total_items",
        "total_item_value",
        "total_freight_value",
        "total_payment_value",
        "primary_payment_type",
        "max_installments",
        "actual_delivery_days",
        "delivery_delay_days",
        "is_late_delivery",
        "freight_ratio",
        "review_score"
    ]].copy()

    fact_orders["is_late_delivery_label"] = fact_orders["is_late_delivery"].map({1: "Late", 0: "On-Time / Early"})

    orders_out = POWERBI_DIR / "fact_orders.csv"
    fact_orders.to_csv(orders_out, index=False)
    print(f"✅ Exported fact_orders: {len(fact_orders):,} rows -> {orders_out}")

    # ==========================================
    # 5. Create Monthly Summary: dim_monthly_summary
    # ==========================================
    print("\n--- 3. Building 'dim_monthly_summary.csv' ---")
    monthly_df = df_orders.groupby("order_month").agg(
        total_orders=("order_id", "count"),
        total_gmv=("total_payment_value", "sum"),
        avg_order_value=("total_payment_value", "mean"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        late_delivery_rate=("is_late_delivery", "mean"),
        avg_review_score=("review_score", "mean")
    ).reset_index()

    # Filter to main operational window
    monthly_df = monthly_df[
        (monthly_df["order_month"] >= "2017-01") & 
        (monthly_df["order_month"] <= "2018-08")
    ].reset_index(drop=True)

    monthly_df["late_delivery_rate"] = (monthly_df["late_delivery_rate"] * 100).round(2)
    monthly_df["total_gmv"] = monthly_df["total_gmv"].round(2)
    monthly_df["avg_order_value"] = monthly_df["avg_order_value"].round(2)
    monthly_df["avg_delivery_days"] = monthly_df["avg_delivery_days"].round(1)
    monthly_df["avg_review_score"] = monthly_df["avg_review_score"].round(2)

    monthly_out = POWERBI_DIR / "dim_monthly_summary.csv"
    monthly_df.to_csv(monthly_out, index=False)
    print(f"✅ Exported dim_monthly_summary: {len(monthly_df):,} months -> {monthly_out}")

    print("\n" + "=" * 65)
    print("🎉 ALL POWER BI STAR SCHEMA TABLES READY IN data/powerbi/")
    print("=" * 65)


if __name__ == "__main__":
    export_powerbi_star_schema()