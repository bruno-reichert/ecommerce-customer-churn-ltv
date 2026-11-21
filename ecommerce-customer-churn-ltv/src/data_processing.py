"""
Data processing and feature engineering module with strict temporal cutoff (T0).
Features are computed solely on or before T0.
Target (is_churned) is computed over the forward outcome window.
"""
import sys
from pathlib import Path
# Guarantee that the project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Tuple
import pandas as pd
import numpy as np

from src.config import (
    DATA_RAW,
    DATA_PROCESSED,
    T0_TIMESTAMP,
    OUTCOME_WINDOW_DAYS
)


def load_raw_tables(data_dir: Path = DATA_RAW) -> Dict[str, pd.DataFrame]:
    """Loads the core Olist CSV tables into a dictionary of DataFrames."""
    path = Path(data_dir)
    tables = {
        "orders": pd.read_csv(path / "olist_orders_dataset.csv"),
        "customers": pd.read_csv(path / "olist_customers_dataset.csv"),
        "items": pd.read_csv(path / "olist_order_items_dataset.csv"),
        "payments": pd.read_csv(path / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(path / "olist_order_reviews_dataset.csv"),
        "products": pd.read_csv(path / "olist_products_dataset.csv"),
        "translations": pd.read_csv(path / "product_category_name_translation.csv"),
    }
    print("✅ All 7 core CSV tables loaded successfully into memory.")
    return tables


def build_unified_orders_dataframe(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges relational tables into a single order-level DataFrame.
    Pre-aggregates payments, items, and reviews to guarantee zero row duplication.
    """
    print("\n--- 1. Pre-aggregating Order Items ---")
    items_agg = tables["items"].groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_item_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        main_product_id=("product_id", "first")
    ).reset_index()

    print("\n--- 2. Pre-aggregating Payments ---")
    payments_agg = tables["payments"].groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        primary_payment_type=("payment_type", "first")
    ).reset_index()

    print("\n--- 3. Pre-aggregating Reviews ---")
    reviews_agg = tables["reviews"].groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "count")
    ).reset_index()

    print("\n--- 4. Merging Category Translations ---")
    products = tables["products"].merge(
        tables["translations"], 
        on="product_category_name", 
        how="left"
    )
    products["clean_category"] = products["product_category_name_english"].fillna("other")

    items_with_cat = items_agg.merge(
        products[["product_id", "clean_category"]],
        left_on="main_product_id",
        right_on="product_id",
        how="left"
    ).drop(columns=["product_id"])

    print("\n--- 5. Master Merge into Orders ---")
    orders = tables["orders"][tables["orders"]["order_status"] == "delivered"].copy()

    orders = orders.merge(
        tables["customers"][["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="inner"
    )
    orders = orders.merge(items_with_cat, on="order_id", how="inner")
    orders = orders.merge(payments_agg, on="order_id", how="inner")
    orders = orders.merge(reviews_agg, on="order_id", how="left")

    orders["review_score"] = orders["review_score"].fillna(orders["review_score"].median())

    datetime_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in datetime_cols:
        orders[col] = pd.to_datetime(orders[col])

    print(f"🎉 Unified Orders Dataset: {len(orders):,} delivered orders.")
    return orders


def engineer_order_level_features(df_orders: pd.DataFrame) -> pd.DataFrame:
    """Computes operational, logistics, and friction metrics for each order."""
    df = df_orders.copy()
    print("\n--- Engineering Order-Level Operational Features ---")

    df["actual_delivery_days"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    )
    df["delivery_delay_days"] = (
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0
    )
    df["is_late_delivery"] = (df["delivery_delay_days"] > 0).astype(int)
    df["freight_ratio"] = df["total_freight_value"] / (df["total_item_value"] + 0.001)
    df["is_negative_review"] = (df["review_score"] <= 2).astype(int)

    df["actual_delivery_days"] = df["actual_delivery_days"].clip(lower=0)
    df["actual_delivery_days"] = df["actual_delivery_days"].fillna(df["actual_delivery_days"].median())
    df["delivery_delay_days"] = df["delivery_delay_days"].fillna(0)

    return df


def aggregate_to_customer_level_with_t0(
    df_orders: pd.DataFrame,
    t0_str: str = T0_TIMESTAMP,
    outcome_days: int = OUTCOME_WINDOW_DAYS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict Temporal Separation (Option B):
    - Observation Window (orders <= T0): used to build all customer features and recency.
    - Outcome Window (T0 < orders <= T0 + outcome_days): used strictly to label churn.
    """
    print(f"\n--- Splitting Data by Cutoff T0: {t0_str} ---")
    t0 = pd.to_datetime(t0_str)
    t_outcome_end = t0 + pd.Timedelta(days=outcome_days)

    # 1. Split orders into observation period vs outcome period
    obs_orders = df_orders[df_orders["order_purchase_timestamp"] <= t0].copy()
    outcome_orders = df_orders[
        (df_orders["order_purchase_timestamp"] > t0) &
        (df_orders["order_purchase_timestamp"] <= t_outcome_end)
    ].copy()

    print(f"Delivered orders on or before T0 (Feature Source): {len(obs_orders):,}")
    print(f"Delivered orders in Outcome Window (Label Source):   {len(outcome_orders):,}")

    # 2. Aggregate features SOLELY from observation orders (<= T0)
    customer_df = obs_orders.groupby("customer_unique_id").agg(
        last_purchase_timestamp=("order_purchase_timestamp", "max"),
        first_purchase_timestamp=("order_purchase_timestamp", "min"),
        total_orders=("order_id", "count"),
        total_items_bought=("total_items", "sum"),
        total_spend=("total_payment_value", "sum"),
        avg_order_value=("total_payment_value", "mean"),
        total_freight_paid=("total_freight_value", "sum"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        max_delivery_delay=("delivery_delay_days", "max"),
        has_experienced_late_delivery=("is_late_delivery", "max"),
        avg_freight_ratio=("freight_ratio", "mean"),
        avg_review_score=("review_score", "mean"),
        has_negative_review=("is_negative_review", "max"),
        avg_installments=("max_installments", "mean"),
        customer_state=("customer_state", "first"),
        favorite_category=("clean_category", lambda x: x.mode().iloc[0] if not x.empty else "other")
    ).reset_index()

    # Recency relative to T0 (NOT relative to dataset end)
    customer_df["recency_days"] = (
        (t0 - customer_df["last_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(1)

    customer_df["customer_relationship_days"] = (
        (customer_df["last_purchase_timestamp"] - customer_df["first_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(1)
    customer_df["is_repeat_buyer"] = (customer_df["total_orders"] > 1).astype(int)

    # 3. Formulate Honest Ground-Truth Label from Outcome Window
    # Repurchased = placed an order between T0 and T0 + outcome_days
    repurchased_ids = set(outcome_orders["customer_unique_id"].unique())
    customer_df["repurchased_in_window"] = customer_df["customer_unique_id"].isin(repurchased_ids).astype(int)
    # is_churned = 1 if did NOT repurchase
    customer_df["is_churned"] = 1 - customer_df["repurchased_in_window"]

    churn_count = customer_df["is_churned"].sum()
    active_count = customer_df["repurchased_in_window"].sum()
    total_customers = len(customer_df)

    print(f"\n🎉 Eligible Cohort (Customers with orders <= T0): {total_customers:,}")
    print(f"   • Churned (no order in next {outcome_days}d): {churn_count:,} ({churn_count / total_customers * 100:.2f}%)")
    print(f"   • Active/Retained (repurchased in next {outcome_days}d): {active_count:,} ({active_count / total_customers * 100:.2f}%)")

    return customer_df, df_orders


def export_processed_datasets(
    df_customers: pd.DataFrame, 
    df_orders: pd.DataFrame,
    output_dir: Path = DATA_PROCESSED
) -> None:
    """Exports processed datasets to Parquet."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cust_file = out_path / "customer_features.parquet"
    df_customers.to_parquet(cust_file, index=False)
    print(f"\n💾 Customer dataset saved: {cust_file} ({len(df_customers):,} rows)")

    orders_file = out_path / "orders_clean.parquet"
    df_orders.to_parquet(orders_file, index=False)
    print(f"💾 Clean orders dataset saved: {orders_file} ({len(df_orders):,} rows)")


if __name__ == "__main__":
    tables = load_raw_tables()
    df_orders = build_unified_orders_dataframe(tables)
    df_orders = engineer_order_level_features(df_orders)
    df_customers, df_orders = aggregate_to_customer_level_with_t0(df_orders)
    export_processed_datasets(df_customers, df_orders)