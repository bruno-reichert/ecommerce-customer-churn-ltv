from pathlib import Path
from typing import Dict
import pandas as pd


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