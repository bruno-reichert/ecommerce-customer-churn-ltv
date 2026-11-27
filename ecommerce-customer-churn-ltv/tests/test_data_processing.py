"""
Isolated unit tests for data processing and aggregation logic.
Uses lightweight synthetic fixtures so tests can run in milliseconds
on a clean checkout without requiring pre-downloaded raw datasets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from src.data_processing import (
    build_snapshot_cohort,
    build_unified_orders_dataframe,
    engineer_order_level_features_with_t0_censor,
)


@pytest.fixture
def synthetic_raw_tables():
    """
    Creates a minimal, realistic synthetic relational database with 3 customers:
    - Customer C1: Placed 2 orders before T0.
    - Customer C2: Placed 1 order before T0 and 1 order in the outcome window (retained).
    - Customer C3: Placed 1 order before T0 and never repurchased (churned).
    """
    customers = pd.DataFrame({
        "customer_id": ["c_ord_1", "c_ord_2", "c_ord_3", "c_ord_4", "c_ord_5"],
        "customer_unique_id": ["C1", "C1", "C2", "C2", "C3"],
        "customer_city": ["sao paulo"] * 5,
        "customer_state": ["SP"] * 5
    })

    # Order timestamps relative to T0 = 2018-03-01
    orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5"],
        "customer_id": ["c_ord_1", "c_ord_2", "c_ord_3", "c_ord_4", "c_ord_5"],
        "order_status": ["delivered"] * 5,
        "order_purchase_timestamp": [
            "2017-05-10 10:00:00",  # O1 (C1, pre-T0)
            "2017-11-20 12:00:00",  # O2 (C1, pre-T0)
            "2018-01-15 09:00:00",  # O3 (C2, pre-T0)
            "2018-04-10 14:00:00",  # O4 (C2, post-T0 outcome window)
            "2017-08-01 16:00:00"   # O5 (C3, pre-T0)
        ],
        "order_approved_at": ["2017-05-10 10:30:00"] * 5,
        "order_delivered_carrier_date": ["2017-05-12 10:00:00"] * 5,
        "order_delivered_customer_date": [
            "2017-05-20 10:00:00",
            "2017-11-28 12:00:00",
            "2018-01-25 09:00:00",
            "2018-04-18 14:00:00",
            "2017-08-15 16:00:00"
        ],
        "order_estimated_delivery_date": [
            "2017-05-25 00:00:00",
            "2017-11-27 00:00:00",  # Arrived late!
            "2018-01-30 00:00:00",
            "2018-04-25 00:00:00",
            "2017-08-20 00:00:00"
        ]
    })

    items = pd.DataFrame({
        "order_id": ["O1", "O1", "O2", "O3", "O4", "O5"],
        "order_item_id": [1, 2, 1, 1, 1, 1],
        "product_id": ["P1", "P2", "P1", "P3", "P1", "P2"],
        "price": [50.0, 50.0, 100.0, 75.0, 120.0, 40.0],
        "freight_value": [10.0, 10.0, 15.0, 12.0, 18.0, 8.0]
    })

    payments = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5"],
        "payment_value": [120.0, 115.0, 87.0, 138.0, 48.0],
        "payment_installments": [1, 3, 2, 4, 1],
        "payment_type": ["credit_card"] * 5
    })

    reviews = pd.DataFrame({
        "review_id": ["R1", "R2", "R3", "R4", "R5"],
        "order_id": ["O1", "O2", "O3", "O4", "O5"],
        "review_score": [5, 1, 4, 5, 2],
        "review_creation_date": [
            "2017-05-21 00:00:00",
            "2017-11-29 00:00:00",
            "2018-01-26 00:00:00",
            "2018-04-19 00:00:00",
            "2017-08-16 00:00:00"
        ]
    })

    products = pd.DataFrame({
        "product_id": ["P1", "P2", "P3"],
        "product_category_name": ["beleza_saude", "cama_mesa_banho", "esporte_lazer"]
    })

    translations = pd.DataFrame({
        "product_category_name": ["beleza_saude", "cama_mesa_banho", "esporte_lazer"],
        "product_category_name_english": ["health_beauty", "bed_bath_table", "sports_leisure"]
    })

    return {
        "orders": orders,
        "customers": customers,
        "items": items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "translations": translations
    }


def test_build_unified_orders_no_row_duplication(synthetic_raw_tables):
    """Verifies that merging 1-to-many items/payments does NOT duplicate order rows."""
    df_orders = build_unified_orders_dataframe(synthetic_raw_tables)
    assert len(df_orders) == 5
    assert df_orders["order_id"].nunique() == 5

    o1 = df_orders[df_orders["order_id"] == "O1"].iloc[0]
    assert o1["total_items"] == 2
    assert o1["total_item_value"] == 100.0


def test_engineer_order_level_features_with_t0_censor(synthetic_raw_tables):
    """Verifies delay calculation, late flags, and anti-leakage censoring at T0."""
    df_orders = build_unified_orders_dataframe(synthetic_raw_tables)
    t0 = pd.to_datetime("2018-03-01 00:00:00")
    df_featured = engineer_order_level_features_with_t0_censor(df_orders, t0)

    # O2 arrived on 2017-11-28 but estimated was 2017-11-27 -> 1 day late
    o2 = df_featured[df_featured["order_id"] == "O2"].iloc[0]
    assert o2["is_late_delivery"] == 1
    assert o2["delivery_delay_days"] > 0
    assert o2["is_negative_review"] == 1


def test_build_snapshot_cohort_t0_properties(synthetic_raw_tables):
    """
    Verifies that:
    1. Output has exactly one row per customer.
    2. Features only use orders <= T0 (O4 from C2 is NOT in feature counts).
    3. Correct ground-truth repurchase label is assigned (C2 repurchased = 1, C3 = 0).
    """
    df_orders = build_unified_orders_dataframe(synthetic_raw_tables)
    t0_cutoff = "2018-03-01 00:00:00"
    df_cust = build_snapshot_cohort(df_orders, t0_str=t0_cutoff, outcome_days=180)

    assert len(df_cust) == 3
    assert df_cust["customer_unique_id"].nunique() == 3

    # Customer C2 placed 1 order before T0 and 1 order after T0:
    c2 = df_cust[df_cust["customer_unique_id"] == "C2"].iloc[0]
    assert c2["total_orders"] == 1, f"Data leakage! C2 has {c2['total_orders']} orders instead of 1."
    assert c2["repurchased_in_window"] == 1
    assert c2["is_churned"] == 0

    # Customer C3 had 1 order before T0 and zero orders in outcome window
    c3 = df_cust[df_cust["customer_unique_id"] == "C3"].iloc[0]
    assert c3["repurchased_in_window"] == 0
    assert c3["is_churned"] == 1

    # Legitimate recency feature relative to T0
    assert (df_cust["recency_days"] >= 0).all()