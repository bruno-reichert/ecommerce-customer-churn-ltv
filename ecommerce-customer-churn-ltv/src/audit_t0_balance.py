# src/audit_t0_balance.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import DATA_RAW, OUTCOME_WINDOW_DAYS, T0_TIMESTAMP


def audit_t0_balance():
    orders = pd.read_csv(DATA_RAW / "olist_orders_dataset.csv")
    customers = pd.read_csv(DATA_RAW / "olist_customers_dataset.csv")

    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])

    df = delivered.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")

    t0 = pd.to_datetime(T0_TIMESTAMP)
    t_end = t0 + pd.Timedelta(days=OUTCOME_WINDOW_DAYS)

    # Split orders into Observation and Outcome periods
    obs_orders = df[df["order_purchase_timestamp"] <= t0]
    outcome_orders = df[(df["order_purchase_timestamp"] > t0) & (df["order_purchase_timestamp"] <= t_end)]

    # Eligible cohort: customers with at least one order on or before T0
    obs_customers = set(obs_orders["customer_unique_id"].unique())
    # Customers who repurchased in the outcome window
    repurchased_customers = set(outcome_orders["customer_unique_id"].unique())

    # Intersection: existing customers who repurchased
    retained_in_cohort = obs_customers.intersection(repurchased_customers)

    total_cohort = len(obs_customers)
    n_retained = len(retained_in_cohort)
    n_churned = total_cohort - n_retained

    print("=" * 65)
    print("🎯 OPTION B TEMPORAL SPLIT DIAGNOSTIC AUDIT")
    print("=" * 65)
    print(f"Cutoff Date (T0):              {t0}")
    print(f"Outcome Window End Date:       {t_end}")
    print(f"Eligible Customers (<= T0):    {total_cohort:,}")
    print(f"Repurchased in Window (Active): {n_retained:,} ({n_retained / total_cohort * 100:.2f}%)")
    print(f"Did Not Repurchase (Churned):   {n_churned:,} ({n_churned / total_cohort * 100:.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    audit_t0_balance()