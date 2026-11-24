"""
Master Pipeline Runner for the Olist Churn & LTV Analytics Project.
Executes the full machine learning lifecycle sequentially:
1. Data Processing & Temporal Ground-Truth Formulation
2. Unsupervised RFM Customer Segmentation
3. Baseline Model Benchmarks & Calibration
4. XGBoost Training & Cross-Validated Hyperparameter Optimization
5. Out-of-Time Model Evaluation & Decision Threshold Locking
6. Global & Local Explainability with Tree-SHAP
7. Commercial Retention & Illustrative ROI Simulation
"""

import sys
import time
from pathlib import Path

# Guarantee project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"🚀 {title.upper()}")
    print("=" * 70)


def run_full_pipeline():
    start_total = time.time()
    print_header("Starting End-to-End Olist Analytics Pipeline")

    # Step 1: Data Processing
    print_header("Step 1/7: Processing Relational Tables & Building Features")
    from src.data_processing import (
        aggregate_to_customer_level_with_t0,
        build_unified_orders_dataframe,
        engineer_order_level_features,
        export_processed_datasets,
        load_raw_tables,
    )
    tables = load_raw_tables()
    df_orders = build_unified_orders_dataframe(tables)
    df_orders = engineer_order_level_features(df_orders)
    df_customers, df_orders = aggregate_to_customer_level_with_t0(df_orders)
    export_processed_datasets(df_customers, df_orders)

    # Step 2: Unsupervised RFM Segmentation
    print_header("Step 2/7: Customer RFM Clustering & Descriptive Personas")
    from src.segmentation import (
        assign_and_profile_personas,
        audit_rfm_distributions,
        extract_rfm_table,
        preprocess_rfm_for_clustering,
    )
    rfm = extract_rfm_table(df_customers)
    audit_rfm_distributions(rfm)
    _df_transformed, scaled_matrix, _scaler = preprocess_rfm_for_clustering(rfm)
    df_customers = assign_and_profile_personas(rfm, scaled_matrix, df_customers, k=4)

    # Prepare features and chronological split for modeling
    from src.model_pipeline import prepare_features_and_target, split_data_chronological
    X, y, timestamps, num_cols, cat_cols = prepare_features_and_target(df_customers)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_chronological(X, y, timestamps)

    # Step 3: Baseline Models & Calibration
    print_header("Step 3/7: Running Baseline Model Benchmarks")
    from src.baseline_models import run_baseline_benchmarks
    run_baseline_benchmarks(X_train, X_test, y_train, y_test, num_cols, cat_cols)

    # Step 4: XGBoost Modeling & Tuning
    print_header("Step 4/7: Training & Tuning XGBoost Pipeline")
    from src.train_xgboost import train_and_tune_xgboost
    pipeline = train_and_tune_xgboost(X_train, y_train, X_val, y_val, num_cols, cat_cols)

    # Step 5: Model Evaluation & Threshold Optimization
    print_header("Step 5/7: Out-of-Time Evaluation & Calibration")
    from src.model_evaluation import evaluate_on_unseen_test
    avg_churner_ltv = float(df_customers[df_customers["is_churned"] == 1]["total_spend"].mean())
    evaluate_on_unseen_test(pipeline, X_val, y_val, X_test, y_test, avg_churner_ltv)

    # Step 6: SHAP Explainability
    print_header("Step 6/7: Computing Tree-SHAP Feature Attribution")
    from src.shap_explainability import run_global_shap_analysis
    run_global_shap_analysis(sample_size=2500)

    # Step 7: Commercial Retention Simulation
    print_header("Step 7/7: Executing Commercial Retention & ROI Simulation")
    from src.financial_simulation import run_honest_financial_simulation
    sorted_indices = timestamps.sort_values().index
    val_end = int(len(sorted_indices) * 0.85)
    test_indices = sorted_indices[val_end:]
    run_honest_financial_simulation(
        pipeline, X_val, y_val, X_test, y_test, df_customers, test_indices
    )

    elapsed_total = time.time() - start_total
    print_header(f"All 7 Pipeline Steps Completed Successfully in {elapsed_total:.1f}s!")


if __name__ == "__main__":
    run_full_pipeline()