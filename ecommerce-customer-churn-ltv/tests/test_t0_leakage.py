# tests/test_t0_leakage.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import T0_TIMESTAMP, DATA_PROCESSED

def test_no_feature_uses_information_after_t0():
    """
    CRITICAL OPTION B GUARDRAIL:
    Verifies that every customer's last_purchase_timestamp used to compute features
    is strictly on or before T0, guaranteeing zero temporal leakage from the outcome window.
    """
    cust_path = DATA_PROCESSED / "customer_features.parquet"
    assert cust_path.exists(), f"Customer features missing at {cust_path}"

    df = pd.read_parquet(cust_path)
    t0 = pd.to_datetime(T0_TIMESTAMP)

    # Convert to datetime if stored as timestamp string or datetime
    last_purchases = pd.to_datetime(df["last_purchase_timestamp"])

    # Every single customer record must have last_purchase <= T0
    violations = df[last_purchases > t0]
    assert len(violations) == 0, (
        f"FATAL: Found {len(violations)} customers with feature timestamps after T0 ({t0})!"
    )

    # Recency relative to T0 must be non-negative
    assert (df["recency_days"] >= 0).all(), "Found negative recency_days relative to T0!"