"""
Central configuration for the Olist Churn & Retention ML pipeline.
All temporal boundaries, commercial simulation constants, and random seeds are defined here.
"""

from pathlib import Path

# ==========================================
# 1. Project Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_POWERBI = BASE_DIR / "data" / "powerbi"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"

# ==========================================
# 2. Temporal Observation Windows (Option 2: Rolling Snapshots)
# ==========================================
# Forward outcome window in days
OUTCOME_WINDOW_DAYS = 180

# Training Snapshot: Feature cutoff T0_train, outcome window T0_train to T0_train + 180d
T0_TRAIN_TIMESTAMP = "2017-09-01 00:00:00"

# Testing Snapshot: Feature cutoff T0_test, outcome window T0_test to T0_test + 180d
T0_TEST_TIMESTAMP = "2018-03-01 00:00:00"

# Backward compatibility alias
T0_TIMESTAMP = T0_TEST_TIMESTAMP

# ==========================================
# 3. Model & Experiment Reproducibility
# ==========================================
RANDOM_SEED = 42

# Time-based splitting proportions (chronological by first_purchase_timestamp)
# Must sum to 1.0 (70% Train, 15% Validation, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# 4. Commercial Financial Simulation
# ==========================================
# Cost of retention outreach (coupon / discount / SMS voucher) per targeted customer (in R$)
COUPON_COST = 15.00

# Base realistic win-back rate for retail churn campaigns (conservative benchmark: 25%)
BASE_WIN_BACK_RATE = 0.25

# Sensitivity analysis range for win-back rates
SENSITIVITY_WIN_BACK_RATES = [0.10, 0.25, 0.40]

# Sensitivity analysis range for intervention coupon costs (R$)
SENSITIVITY_COUPON_COSTS = [10.00, 15.00, 20.00]