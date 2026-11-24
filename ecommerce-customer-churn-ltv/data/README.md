# Olist Brazilian E-Commerce Dataset

This repository uses the public Brazilian E-Commerce dataset published by Olist on Kaggle.

---

## Kaggle Download Instructions

### Option 1: Automated Download via Kaggle CLI (Recommended)
1. Install the official Kaggle CLI:
   `pip install kaggle`

2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

3. Download and unpack all CSVs directly into `data/raw/`:
   `kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw/ --unzip`

### Option 2: Manual Download
1. Visit the Kaggle dataset page: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Click **Download (103 MB)**.
3. Unzip the archive and place all 8 CSV files into the `data/raw/` directory:
   - `olist_orders_dataset.csv`
   - `olist_customers_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_order_payments_dataset.csv`
   - `olist_order_reviews_dataset.csv`
   - `olist_products_dataset.csv`
   - `olist_geolocation_dataset.csv`
   - `product_category_name_translation.csv`

---

## Dataset Attribution & License Check
- **Original Publisher:** Olist and Andre Sionek
- **Platform:** Kaggle
- **Data License:** CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)
- **Code License:** MIT License