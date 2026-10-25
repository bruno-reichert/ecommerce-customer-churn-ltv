from pathlib import Path
import pandas as pd

def audit_raw_data():
    raw_dir = Path("data/raw")
    csv_files = sorted(list(raw_dir.glob("*.csv")))
    
    if not csv_files:
        print("❌ No CSV files found in data/raw/!")
        print("Please download and unzip the Olist files into data/raw/.")
        return

    print("=" * 70)
    print("📦 OLIST BRAZILIAN E-COMMERCE DATASET AUDIT")
    print("=" * 70)
    
    total_rows = 0
    for file in csv_files:
        df = pd.read_csv(file, nrows=100) # Fast load for column schema check
        # Get true line count
        with open(file, "rb") as f:
            row_count = sum(1 for _ in f) - 1 # Subtract header
            
        total_rows += row_count
        print(f"\n📄 {file.name}")
        print(f"   • Rows: {row_count:,}")
        print(f"   • Columns ({len(df.columns)}): {', '.join(df.columns[:5])}...")

    print("\n" + "=" * 70)
    print(f"✅ Total relational rows ready for processing: {total_rows:,}")
    print("=" * 70)

if __name__ == "__main__":
    audit_raw_data()