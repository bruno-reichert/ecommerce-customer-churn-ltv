import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import (
    load_raw_tables,
    build_unified_orders_dataframe,
    engineer_order_level_features,
    aggregate_to_customer_level,
    export_processed_datasets
)

def main():
    tables = load_raw_tables()
    df_orders = build_unified_orders_dataframe(tables)
    df_orders = engineer_order_level_features(df_orders)
    df_customers = aggregate_to_customer_level(df_orders)
    export_processed_datasets(df_customers, df_orders)

if __name__ == "__main__":
    main()