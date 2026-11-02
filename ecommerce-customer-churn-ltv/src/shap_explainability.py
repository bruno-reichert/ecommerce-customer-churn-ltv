import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np
import shap

# Headless rendering backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model_pipeline import (
    prepare_features_and_target,
    split_data
)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/best_xgboost_churn_pipeline.joblib")


def clean_feature_names(raw_names: np.ndarray) -> list:
    """
    Cleans Scikit-Learn ColumnTransformer prefixes to make
    feature labels clean and executive-ready for plots.
    e.g., 'num__max_delivery_delay' -> 'Max Delivery Delay'
          'cat__persona_At-Risk High Spenders' -> 'Persona: At-Risk High Spenders'
    """
    clean_names = []
    for name in raw_names:
        # Strip prefixes
        clean = name.replace("num__", "").replace("cat__", "")
        clean = clean.replace("customer_state_", "State: ")
        clean = clean.replace("favorite_category_", "Category: ")
        clean = clean.replace("persona_", "Persona: ")
        clean = clean.replace("_", " ").title()
        clean_names.append(clean)
    return clean_names


def run_global_shap_analysis(sample_size: int = 3000):
    """
    Computes Tree-SHAP values on a representative test sample
    and exports Global Feature Importance and Summary Beeswarm plots.
    """
    print("=" * 65)
    print("🧠 PHASE 5.1: GLOBAL FEATURE EXPLAINABILITY WITH SHAP")
    print("=" * 65)

    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}! Run src/train_xgboost.py first.")
        return

    # 1. Load Data & Split
    customers_path = Path("data/processed/customer_features.parquet")
    df_customers = pd.read_parquet(customers_path)
    X, y, num_cols, cat_cols = prepare_features_and_target(df_customers)
    _, X_test, _, y_test = split_data(X, y)

    # 2. Load Pipeline Artifact & Dissect Components
    print(f"\nLoading model pipeline from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["classifier"]

    # 3. Transform X_test through preprocessor and extract clean feature names
    print("Transforming test set through preprocessor...")
    X_test_transformed = preprocessor.transform(X_test)
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(raw_feature_names)

    # 4. Take a representative sample for fast, crisp SHAP calculation
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test_transformed), size=min(sample_size, len(X_test_transformed)), replace=False)
    X_sample = X_test_transformed[sample_indices]
    
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    # 5. Initialize TreeExplainer and calculate SHAP values
    print(f"Calculating SHAP values for {len(X_sample):,} customers using TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample_df)

    # ==========================================
    # Plot 1: SHAP Summary Beeswarm Plot
    # ==========================================
    print("\n📊 Generating SHAP Summary Beeswarm Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values.values,
        X_sample_df,
        max_display=15,
        show=False
    )
    plt.title("SHAP Global Feature Impact on Churn (Direction & Magnitude)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("SHAP Value (Impact on Model Churn Probability)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    beeswarm_path = FIGURES_DIR / "11_shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Beeswarm Plot: {beeswarm_path}")

    # ==========================================
    # Plot 2: Global Mean Absolute Feature Importance (Bar)
    # ==========================================
    print("📊 Generating SHAP Feature Importance Bar Plot...")
    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values.values,
        X_sample_df,
        plot_type="bar",
        max_display=15,
        color="#2980b9",
        show=False
    )
    plt.title("Top 15 Most Important Drivers of Customer Churn (Mean |SHAP Value|)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mean |SHAP Value| (Average Absolute Impact)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    bar_path = FIGURES_DIR / "12_shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Feature Importance Bar Plot: {bar_path}")

    # 6. Extract Top 5 Drivers for Terminal Summary
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_driver_indices = np.argsort(mean_abs_shap)[::-1][:5]

    print("\n" + "=" * 65)
    print("🏆 TOP 5 STRATEGIC DRIVERS OF OLIST CUSTOMER CHURN:")
    print("=" * 65)
    for rank, idx in enumerate(top_driver_indices, start=1):
        print(f"   {rank}. {feature_names[idx]:<30} (Mean Impact: {mean_abs_shap[idx]:.4f})")
    print("=" * 65)


def main():
    run_global_shap_analysis(sample_size=3000)


if __name__ == "__main__":
    main()