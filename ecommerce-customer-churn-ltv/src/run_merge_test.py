from src.data_processing import load_raw_tables, build_unified_orders_dataframe

def main():
    tables = load_raw_tables()
    df_orders = build_unified_orders_dataframe(tables)

    print("\n" + "=" * 60)
    print("📋 UNIFIED ORDERS SCHEMA & INTEGRITY CHECK")
    print("=" * 60)
    print(df_orders.info())
    print("\n--- SAMPLE ROW ---")
    sample = df_orders[["order_id", "customer_unique_id", "total_item_value", "total_payment_value", "clean_category", "review_score"]].iloc[0]
    print(sample)

    # Sanity check: Ensure customer_unique_id has repeat buyers!
    customer_counts = df_orders["customer_unique_id"].value_counts()
    repeat_buyers = (customer_counts > 1).sum()
    print("\n" + "=" * 60)
    print(f"👤 Total Unique Customers:     {len(customer_counts):,}")
    print(f"🔁 Repeat Buyers Identified:   {repeat_buyers:,} ({(repeat_buyers / len(customer_counts)) * 100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()