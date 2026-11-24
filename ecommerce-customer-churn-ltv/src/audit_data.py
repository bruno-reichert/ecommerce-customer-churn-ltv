"""
Raw Data Ingestion & Relational Schema Audit.
Verifies the presence of all core Olist CSV tables, displays record counts,
and checks data licensing attribution (CC BY-NC-SA 4.0).
"""

import sys
from pathlib import Path

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW

EXPECTED_CSVS = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv"
]


def audit_raw_data():
    raw_dir = DATA_RAW
    csv_files = sorted(raw_dir.glob("*.csv"))

    print("=" * 70)
    print("📦 OLIST BRAZILIAN E-COMMERCE DATASET & LICENSE AUDIT")
    print("=" * 70)
    print("📌 Dataset Attribution: Olist & André Sionek (Kaggle)")
    print("📜 Data License:        CC BY-NC-SA 4.0 (Non-Commercial ShareAlike)")
    print("⚖️ Code License:        MIT License (Open Source)")
    print("=" * 70)

    if not csv_files:
        print("\n❌ No CSV files found in data/raw/!")
        print("Please follow the instructions in data/README.md to download the dataset via Kaggle CLI:")
        print("   kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw/ --unzip")
        return

    missing_csvs = [f for f in EXPECTED_CSVS if not (raw_dir / f).exists()]
    if missing_csvs:
        print(f"\n⚠️ Missing required CSV tables: {missing_csvs}")
    else:
        print("\n✅ All core relational CSV tables verified present.")

    total_rows = 0
    for file in csv_files:
        if file.name not in EXPECTED_CSVS:
            continue
        with open(file, "rb") as f:
            row_count = sum(1 for _ in f) - 1
        total_rows += row_count
        print(f"   • {file.name:<38} | Rows: {row_count:>10,}")

    print("-" * 70)
    print(f"✅ Total core relational rows ready for processing: {total_rows:,}")
    print("=" * 70)


if __name__ == "__main__":
    audit_raw_data()