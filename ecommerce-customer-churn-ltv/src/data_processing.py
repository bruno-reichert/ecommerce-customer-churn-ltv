"""
Data processing and feature engineering module with strict temporal cutoff (T0).
Supports Option 2 Rolling Snapshots:
- Snapshot 1 (Train): orders <= T0_train, outcome window (T0_train, T0_train + 180d]
- Snapshot 2 (Test):  orders <= T0_test,  outcome window (T0_test,  T0_test  + 180d]
Features strictly mask any delivery or review event that occurred after the snapshot T0.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    OUTCOME_WINDOW_DAYS,
    T0_TEST_TIMESTAMP,
    T0_TRAIN_TIMESTAMP,
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
    Leaves reviews un-imputed to prevent global data leakage.
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

    print("\n--- 3. Pre-aggregating Reviews with Timestamps ---")
    reviews_df = tables["reviews"].copy()
    reviews_df["review_creation_date"] = pd.to_datetime(reviews_df["review_creation_date"])
    reviews_agg = reviews_df.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "count"),
        latest_review_date=("review_creation_date", "max")
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


def engineer_order_level_features_with_t0_censor(df_orders: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """
    Computes operational and friction metrics for orders placed on or before T0.
    CRITICAL ANTI-LEAKAGE FIX:
    - If delivered after T0: actual_delivery_days, delivery_delay_days, and is_late_delivery
      are masked to NaN (in transit at T0).
    - If review created after T0: review_score and is_negative_review are masked to NaN.
    """
    df = df_orders.copy()

    # 1. Freight Ratio (Known at purchase time)
    df["freight_ratio"] = df["total_freight_value"] / (df["total_item_value"] + 0.001)

    # 2. Delivery Time & Delay
    # Mask if delivery date occurred AFTER T0
    delivered_mask = (df["order_delivered_customer_date"].notnull()) & (df["order_delivered_customer_date"] <= t0)

    df["actual_delivery_days"] = np.where(
        delivered_mask,
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0,
        np.nan
    )
    df["delivery_delay_days"] = np.where(
        delivered_mask,
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0,
        np.nan
    )
    df["is_late_delivery"] = np.where(
        delivered_mask,
        (df["delivery_delay_days"] > 0).astype(float),
        np.nan
    )

    # Clip negative delivery days to 0 if package arrived on same day
    df["actual_delivery_days"] = df["actual_delivery_days"].clip(lower=0)

    # 3. Review Metrics (Mask if review was submitted AFTER T0)
    review_mask = (df["latest_review_date"].notnull()) & (df["latest_review_date"] <= t0)
    df["review_score_t0"] = np.where(review_mask, df["review_score"], np.nan)
    df["is_negative_review"] = np.where(
        review_mask,
        (df["review_score_t0"] <= 2).astype(float),
        np.nan
    )

    return df


def build_snapshot_cohort(
    df_orders: pd.DataFrame,
    t0_str: str,
    outcome_days: int = OUTCOME_WINDOW_DAYS
) -> pd.DataFrame:
    """
    Generates a single self-contained customer cohort snapshot at cutoff T0:
    - Features: strictly from orders <= T0, with post-T0 delivery/reviews masked.
    - Includes legitimate recency_days and customer_relationship_days.
    - Outcome: repurchased_in_window (1 if bought between T0 and T0 + outcome_days).
    """
    t0 = pd.to_datetime(t0_str)
    t_outcome_end = t0 + pd.Timedelta(days=outcome_days)

    obs_orders = df_orders[df_orders["order_purchase_timestamp"] <= t0].copy()
    outcome_orders = df_orders[
        (df_orders["order_purchase_timestamp"] > t0) &
        (df_orders["order_purchase_timestamp"] <= t_outcome_end)
    ].copy()

    # Apply anti-leakage masking to observation orders
    obs_orders = engineer_order_level_features_with_t0_censor(obs_orders, t0)

    # Aggregate to customer level
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
        avg_review_score=("review_score_t0", "mean"),
        has_negative_review=("is_negative_review", "max"),
        avg_installments=("max_installments", "mean"),
        customer_state=("customer_state", "first"),
        favorite_category=("clean_category", lambda x: x.mode().iloc[0] if not x.empty else "other")
    ).reset_index()

    # Legitimate recency features relative to this snapshot's T0
    customer_df["recency_days"] = (
        (t0 - customer_df["last_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(1)

    customer_df["customer_relationship_days"] = (
        (customer_df["last_purchase_timestamp"] - customer_df["first_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(1)

    customer_df["is_repeat_buyer"] = (customer_df["total_orders"] > 1).astype(int)

    # Outcome Label: Primary target is repurchased_in_window (the rare positive event)
    repurchased_ids = set(outcome_orders["customer_unique_id"].unique())
    customer_df["repurchased_in_window"] = customer_df["customer_unique_id"].isin(repurchased_ids).astype(int)
    customer_df["is_churned"] = 1 - customer_df["repurchased_in_window"]

    n_total = len(customer_df)
    n_repurchased = int(customer_df["repurchased_in_window"].sum())
    repurchase_rate = (n_repurchased / n_total) * 100 if n_total > 0 else 0.0

    print(f"Snapshot @ T0 = {t0.date()} (Outcome to {t_outcome_end.date()}):")
    print(f"   • Customers:           {n_total:,}")
    print(f"   • Repurchased (Event): {n_repurchased:,} ({repurchase_rate:.2f}%)")
    print(f"   • Churned (Dormant):   {n_total - n_repurchased:,} ({100 - repurchase_rate:.2f}%)")

    return customer_df


def export_snapshots(
    df_train_snapshot: pd.DataFrame,
    df_test_snapshot: pd.DataFrame,
    df_orders: pd.DataFrame,
    output_dir: Path = DATA_PROCESSED
) -> None:
    """Exports both snapshots and clean orders to Parquet."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot 1 (Train)
    train_file = out_path / "customer_features_train.parquet"
    df_train_snapshot.to_parquet(train_file, index=False)
    print(f"\n💾 Train Snapshot saved: {train_file} ({len(df_train_snapshot):,} rows)")

    # 2. Snapshot 2 (Test)
    test_file = out_path / "customer_features_test.parquet"
    df_test_snapshot.to_parquet(test_file, index=False)
    print(f"💾 Test Snapshot saved:  {test_file} ({len(df_test_snapshot):,} rows)")

    # 3. Main customer_features.parquet (points to test snapshot for backward compatibility)
    cust_file = out_path / "customer_features.parquet"
    df_test_snapshot.to_parquet(cust_file, index=False)
    print(f"💾 Default customer features: {cust_file} ({len(df_test_snapshot):,} rows)")

    # 4. Clean Orders
    orders_file = out_path / "orders_clean.parquet"
    df_orders.to_parquet(orders_file, index=False)
    print(f"💾 Clean orders saved:   {orders_file} ({len(df_orders):,} rows)")


if __name__ == "__main__":
    tables = load_raw_tables()
    df_orders = build_unified_orders_dataframe(tables)

    print("\n--- Generating Rolling Snapshot 1 (Train: T0 = 2017-09-01) ---")
    df_train_snapshot = build_snapshot_cohort(df_orders, T0_TRAIN_TIMESTAMP)

    print("\n--- Generating Rolling Snapshot 2 (Test: T0 = 2018-03-01) ---")
    df_test_snapshot = build_snapshot_cohort(df_orders, T0_TEST_TIMESTAMP)

    export_snapshots(df_train_snapshot, df_test_snapshot, df_orders)