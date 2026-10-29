import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import (
    load_raw_tables,
    build_unified_orders_dataframe,
    engineer_order_level_features,
    aggregate_to_customer_level,
    label_churn_and_export
)

def main():
    # 1. Load & Merge
    tables = load_raw_tables()
    df_orders = build_unified_orders_dataframe(tables)

    # 2. Engineer Order Features
    df_orders = engineer_order_level_features(df_orders)

    # 3. Aggregate to Customer Level
    df_customers = aggregate_to_customer_level(df_orders)

    # 4. Label Churn & Export to Parquet
    df_final = label_churn_and_export(df_customers)

if __name__ == "__main__":
    main()