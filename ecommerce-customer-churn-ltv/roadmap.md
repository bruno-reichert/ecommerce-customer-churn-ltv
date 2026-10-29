# 🗺️ Roadmap: E-Commerce Customer Churn & LTV Predictive Analytics
### Strategic Customer Retention, Segmentation & Machine Learning on Brazilian E-Commerce Data (Olist)

---

## 🛠️ The Data Science & Analytics Tech Stack
* **Language & Runtime:** Python 3.11+
* **Environment & Package Manager:** `uv`
* **Data Wrangling & Manipulation:** `pandas`, `numpy`
* **Visualization & Business Intelligence:** `matplotlib`, `seaborn`
* **Unsupervised Machine Learning:** `scikit-learn` (K-Means Clustering, PCA, Silhouette Analysis)
* **Supervised Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
* **Model Explainability & Business Interpretability:** `shap`
* **Interactive Exploration & Presentation:** Jupyter Notebooks (`notebook` / `jupyterlab`) + Streamlit Dashboard (Optional)
* **Code Quality & CI:** `pytest`, `ruff`

---

## 📅 Multi-Part Action Plan

### 📋 Phase 0: Project Setup & Data Acquisition
> **Goal:** Set up reproducible virtual environment via `uv`, acquire the raw Olist relational dataset, and establish directory conventions.

- [X] **0.1 Environment Initialization:**
  - Initialize project with `uv` and configure `pyproject.toml` / `requirements.txt`.
  - Install core scientific stack (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, `jupyterlab`).
- [X] **0.2 Data Download & Schema Audit:**
  - Download the 8 relational CSV files from the Kaggle Brazilian E-Commerce Public Dataset by Olist.
  - Store raw files in `data/raw/` with a relational diagram check (Primary Keys & Foreign Keys).

---

### 🧹 Phase 1: Relational Data Engineering & Feature Synthesis (Pandas)
> **Goal:** Transform 8 fragmented relational tables into a unified, analytical customer-level dataset (`data/processed/customer_features.parquet`).

- [X] **1.1 Relational Merging & Primary Key Resolution:**
  - Resolve the difference between `customer_id` (per order) and `customer_unique_id` (the actual human being across multiple purchases).
  - Join orders, items, payments, reviews, and geolocation data without Cartesian duplication.
- [X] **1.2 Operational Feature Engineering:**
  - *Logistics features:* Calculate `delivery_delay_days` (difference between actual delivery and estimated delivery date), `freight_ratio` (freight cost vs. item price).
  - *Payment behavior:* Average installments, payment type preferences (Boleto vs. Cartão).
  - *Review metrics:* Historical review scores and whether the customer left negative feedback.
- [ ] **1.3 Churn Definition & Ground Truth Formulation:**
  - Establish an objective e-commerce churn window based on Brazilian retail purchasing cycles (e.g., customers with no repeat purchases within a defined historical inactivity cutoff).

---

### 📊 Phase 2: Exploratory Data Analysis (EDA) & Commercial Insights (Seaborn & Matplotlib)
> **Goal:** Discover commercial patterns, geographic logistics bottlenecks, and customer behavior trends to drive business decisions.

- [ ] **2.1 Cohort & Temporal Analysis:**
  - Monthly order volume, gross merchandise value (GMV), and seasonality spikes (e.g., Black Friday 2017).
- [ ] **2.2 Logistics Impact Analysis:**
  - Correlation between delivery delays / freight costs and customer satisfaction (review scores).
  - Regional logistics disparities (Southeast vs. North/Northeast delivery lead times).
- [ ] **2.3 Financial & Unit Economics Breakdown:**
  - Basket size distributions, installment usage across income/geographic regions, and repeat purchase rates.

---

### 🧩 Phase 3: Customer Segmentation & Lifetime Value (RFM + K-Means)
> **Goal:** Categorize customers into actionable strategic tiers based on Recency, Frequency, and Monetary behavior.

- [ ] **3.1 RFM Calculation:**
  - Calculate **R** (days since last purchase), **F** (total number of orders), and **M** (total monetary spend) for every unique customer.
- [ ] **3.2 Preprocessing for Clustering:**
  - Log-transformation to handle heavy right-skewed monetary distributions; feature standardization via `StandardScaler`.
- [ ] **3.3 K-Means Optimization:**
  - Determine optimal $k$ clusters using the **Elbow Method** (Inertia) and **Silhouette Scores**.
- [ ] **3.4 Commercial Persona Profiling:**
  - Assign business names to clusters (e.g., *Champions / VIPs*, *Loyal Customers*, *At-Risk High Spenders*, *Lost / Hibernating*).
  - Formulate targeted marketing strategies per cluster (e.g., cross-sell, reactivation discounts).

---

### 🤖 Phase 4: Supervised Churn Predictive Modeling (Scikit-Learn & XGBoost)
> **Goal:** Build, evaluate, and tune a machine learning classifier to predict customer churn probability.

- [ ] **4.1 Pipeline Design & Data Splitting:**
  - Time-based or stratified train/test split to prevent temporal data leakage.
  - Preprocessing pipeline: `OneHotEncoder` for categoricals, `StandardScaler` for continuous features.
- [ ] **4.2 Baseline Models:**
  - Train standard baselines: Dummy Classifier, Logistic Regression, Random Forest.
- [ ] **4.3 Gradient Boosting & Hyperparameter Tuning:**
  - Train **XGBoost Classifier** with class imbalance mitigation (e.g., `scale_pos_weight`).
  - Optimize hyperparameters using `GridSearchCV` or `RandomizedSearchCV` cross-validation.
- [ ] **4.4 Model Evaluation:**
  - Assess performance using metrics tailored for commercial imbalance: **ROC-AUC**, **Precision-Recall Curve**, **F1-Score**, and **Confusion Matrix**.
  - Set optimal classification threshold based on the business cost of false positives (unnecessary discount coupons) vs. false negatives (lost customers).

---

### 💡 Phase 5: Business Interpretability & LTV Financial Impact (SHAP)
> **Goal:** Translate mathematical model coefficients into actionable executive decisions.

- [ ] **5.1 Global Feature Importance with SHAP:**
  - Generate SHAP Summary Beeswarm plots to identify what drives churn company-wide (e.g., does freight cost outweigh delivery delay?).
- [ ] **5.2 Individual Customer Explainability (Waterfall Plots):**
  - Show how the model explains risk for a specific high-LTV customer.
- [ ] **5.3 Financial Simulation & ROI Calculation:**
  - Calculate the monetary impact: *"If retention campaigns save 15% of at-risk customers in Cluster A, what is the net revenue preserved for Olist?"*

---

### 📦 Phase 6: Executive Deliverables & Portfolio Showcase
> **Goal:** Package the repository with clean code, reproducible notebooks, and an executive README.

- [ ] **6.1 Refactoring & Quality:**
  - Organize reusable pipeline functions in `src/` (`data_loader.py`, `features.py`, `models.py`).
  - Add unit tests with `pytest` verifying data transformations and schema invariants.
- [ ] **6.2 Executive Presentation & README:**
  - Summary of business problem, methodology, data schema diagram, top model performance metrics, and strategic recommendations for retail executives.