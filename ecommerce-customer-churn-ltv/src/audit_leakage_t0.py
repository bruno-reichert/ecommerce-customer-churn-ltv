# src/audit_leakage_t0.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import T0_TIMESTAMP, DATA_RAW

def audit_post_t0_events():
    orders = pd.read_csv(DATA_RAW / "olist_orders_dataset.csv")
    reviews = pd.read_csv(DATA_RAW / "olist_order_reviews_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])

    t0 = pd.to_datetime(T0_TIMESTAMP)

    # Orders placed on or before T0
    pre_t0_orders = orders[orders["order_purchase_timestamp"] <= t0]

    # Delivered after T0
    delivered_after_t0 = pre_t0_orders[pre_t0_orders["order_delivered_customer_date"] > t0]

    # Reviews created after T0 for pre-T0 orders
    pre_t0_reviews = reviews.merge(pre_t0_orders[["order_id"]], on="order_id", how="inner")
    reviews_after_t0 = pre_t0_reviews[pre_t0_reviews["review_creation_date"] > t0]

    print("=" * 65)
    print("🕵️ T0 OPERATIONAL LEAKAGE AUDIT")
    print("=" * 65)
    print(f"Total orders placed <= T0 ({t0}):      {len(pre_t0_orders):,}")
    print(f"Delivered AFTER T0 (in transit at T0):  {len(delivered_after_t0):,} ({len(delivered_after_t0)/len(pre_t0_orders)*100:.2f}%)")
    print(f"Reviews created AFTER T0:              {len(reviews_after_t0):,} ({len(reviews_after_t0)/len(pre_t0_orders)*100:.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    audit_post_t0_events()