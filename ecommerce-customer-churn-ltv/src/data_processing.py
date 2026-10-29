from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np


def load_raw_tables(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
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
    # A single order can have multiple items and freight fees
    items_agg = tables["items"].groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_item_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        main_product_id=("product_id", "first") # primary product for category mapping
    ).reset_index()
    print(f"Aggregated {len(tables['items']):,} item rows -> {len(items_agg):,} unique orders.")

    print("\n--- 2. Pre-aggregating Payments ---")
    # An order can have multiple payment installments or methods (e.g. voucher + credit card)
    payments_agg = tables["payments"].groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        primary_payment_type=("payment_type", "first")
    ).reset_index()
    print(f"Aggregated {len(tables['payments']):,} payment rows -> {len(payments_agg):,} unique orders.")

    print("\n--- 3. Pre-aggregating Reviews ---")
    # Take the average review score if multiple reviews exist for an order
    reviews_agg = tables["reviews"].groupby("order_id").agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "count")
    ).reset_index()
    print(f"Aggregated {len(tables['reviews']):,} review rows -> {len(reviews_agg):,} unique orders.")

    print("\n--- 4. Merging Category Translations ---")
    products = tables["products"].merge(
        tables["translations"], 
        on="product_category_name", 
        how="left"
    )
    products["clean_category"] = products["product_category_name_english"].fillna("other")

    # Map main product category to order items
    items_with_cat = items_agg.merge(
        products[["product_id", "clean_category"]],
        left_on="main_product_id",
        right_on="product_id",
        how="left"
    ).drop(columns=["product_id"])

    print("\n--- 5. Master Merge into Orders ---")
    # Filter for delivered orders to ensure clean metrics (canceled orders skew LTV)
    orders = tables["orders"][tables["orders"]["order_status"] == "delivered"].copy()
    print(f"Delivered orders to process: {len(orders):,} (out of {len(tables['orders']):,} total)")

    # Join customer mapping to get customer_unique_id
    orders = orders.merge(
        tables["customers"][["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="inner"
    )

    # Join pre-aggregated items, payments, and reviews
    orders = orders.merge(items_with_cat, on="order_id", how="inner")
    orders = orders.merge(payments_agg, on="order_id", how="inner")
    orders = orders.merge(reviews_agg, on="order_id", how="left")

    # Fill missing review scores with median (3.0 or 4.0)
    orders["review_score"] = orders["review_score"].fillna(orders["review_score"].median())

    # Convert timestamp columns to real datetime objects
    datetime_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in datetime_cols:
        orders[col] = pd.to_datetime(orders[col])

    print(f"\n🎉 Unified Orders Dataset Created: {len(orders):,} orders ready!")
    print(f"Unique human customers represented: {orders['customer_unique_id'].nunique():,}")
    return orders

def engineer_order_level_features(df_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Computes operational, logistics, and friction metrics for each order.
    """
    df = df_orders.copy()
    print("\n--- Engineering Order-Level Operational Features ---")

    # 1. Delivery Time (Days from purchase to customer doorstep)
    # Using total_seconds() / 86400 gives clean floating-point days (e.g. 5.3 days)
    df["actual_delivery_days"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    )

    # 2. Delivery Delay vs. Promised SLA
    # Positive value = package arrived LATE. Negative = arrived EARLY.
    df["delivery_delay_days"] = (
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0
    )
    # Boolean flag: 1 if delivery exceeded the promised deadline
    df["is_late_delivery"] = (df["delivery_delay_days"] > 0).astype(int)

    # 3. Freight Ratio (Shipping cost as a percentage of product value)
    # Add small epsilon (0.001) to prevent division by zero if item value is 0
    df["freight_ratio"] = df["total_freight_value"] / (df["total_item_value"] + 0.001)

    # 4. Negative Review Flag (1 if review_score <= 2, else 0)
    df["is_negative_review"] = (df["review_score"] <= 2).astype(int)

    # Handle slight edge cases where delivery timestamps were null or negative
    df["actual_delivery_days"] = df["actual_delivery_days"].clip(lower=0)
    # Impute missing delivery days with median
    df["actual_delivery_days"] = df["actual_delivery_days"].fillna(df["actual_delivery_days"].median())
    df["delivery_delay_days"] = df["delivery_delay_days"].fillna(0)

    print("✅ Order-level features created: actual_delivery_days, delivery_delay_days, is_late_delivery, freight_ratio, is_negative_review.")
    return df

def aggregate_to_customer_level(df_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Rolls up order-level data into a single, unified customer-level analytical dataset
    keyed by 'customer_unique_id'.
    """
    print("\n--- Aggregating to Unique Customer Level ---")

    # Reference date: The day after the latest purchase in the dataset (for recency calculations)
    max_purchase_date = df_orders["order_purchase_timestamp"].max()

    customer_df = df_orders.groupby("customer_unique_id").agg(
        # Recency: Days since most recent purchase
        last_purchase_timestamp=("order_purchase_timestamp", "max"),
        first_purchase_timestamp=("order_purchase_timestamp", "min"),
        
        # Frequency & Lifetime Volume
        total_orders=("order_id", "count"),
        total_items_bought=("total_items", "sum"),
        
        # Monetary Spend (Historical LTV)
        total_spend=("total_payment_value", "sum"),
        avg_order_value=("total_payment_value", "mean"),
        total_freight_paid=("total_freight_value", "sum"),
        
        # Operational Friction (Averages & Maxima)
        avg_delivery_days=("actual_delivery_days", "mean"),
        max_delivery_delay=("delivery_delay_days", "max"),
        has_experienced_late_delivery=("is_late_delivery", "max"), # 1 if ANY order arrived late
        avg_freight_ratio=("freight_ratio", "mean"),
        
        # Satisfaction & Payment
        avg_review_score=("review_score", "mean"),
        has_negative_review=("is_negative_review", "max"), # 1 if customer ever gave <= 2 stars
        avg_installments=("max_installments", "mean"),
        
        # Categorical / Location (Take mode or first)
        customer_state=("customer_state", "first"),
        favorite_category=("clean_category", lambda x: x.mode().iloc[0] if not x.empty else "other")
    ).reset_index()

    # Calculate Recency in Days
    customer_df["recency_days"] = (
        (max_purchase_date - customer_df["last_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(1)

    # Customer Lifetime Span (in days between first and last purchase)
    customer_df["customer_relationship_days"] = (
        (customer_df["last_purchase_timestamp"] - customer_df["first_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(1)

    # Repeat buyer indicator
    customer_df["is_repeat_buyer"] = (customer_df["total_orders"] > 1).astype(int)

    print(f"🎉 Customer-level dataset created: {len(customer_df):,} unique customers ready!")
    return customer_df

def label_churn_and_export(df_customers: pd.DataFrame, output_path: str = "data/processed/customer_features.parquet") -> pd.DataFrame:
    """
    Labels churn based on recency inactivity and saves the processed dataset to Parquet.
    In Brazilian non-contractual e-commerce, customers inactive for > 180 days 
    exhibit < 5% probability of organic repurchase.
    """
    df = df_customers.copy()
    print("\n--- Formulating Churn Target Label ---")

    # Inactivity threshold: 180 days (6 months)
    CHURN_THRESHOLD_DAYS = 180.0
    
    # 1 = Churned (Inactive for > 180 days), 0 = Active / Retained
    df["is_churned"] = (df["recency_days"] > CHURN_THRESHOLD_DAYS).astype(int)

    churn_rate = df["is_churned"].mean() * 100
    print(f"Churn Threshold: > {CHURN_THRESHOLD_DAYS} days of inactivity.")
    print(f"Overall Customer Churn Rate: {churn_rate:.2f}%")
    print(f"Active Customers: {len(df[df['is_churned'] == 0]):,} | Churned Customers: {len(df[df['is_churned'] == 1]):,}")

    # Export to Parquet (high-performance columnar format for Pandas/ML)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"\n💾 Analytical dataset saved to: {output_file.resolve()}")

    return df