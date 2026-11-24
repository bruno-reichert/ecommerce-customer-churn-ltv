# src/audit_dates.py
from pathlib import Path

import pandas as pd


def audit_order_dates():
    orders_path = Path("data/raw/olist_orders_dataset.csv")
    if not orders_path.exists():
        print(f"❌ Orders CSV not found at {orders_path}")
        return

    df = pd.read_csv(orders_path, usecols=["order_status", "order_purchase_timestamp"])
    delivered = df[df["order_status"] == "delivered"].copy()
    delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])

    min_date = delivered["order_purchase_timestamp"].min()
    max_date = delivered["order_purchase_timestamp"].max()

    print("=" * 60)
    print("📅 OLIST DELIVERED ORDERS DATE RANGE AUDIT")
    print("=" * 60)
    print(f"First purchase date: {min_date}")
    print(f"Last purchase date:  {max_date}")
    print(f"Total span:          {(max_date - min_date).days} days")
    print("=" * 60)

if __name__ == "__main__":
    audit_order_dates()