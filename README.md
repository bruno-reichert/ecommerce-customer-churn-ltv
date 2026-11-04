# 📦 E-Commerce Customer Churn & LTV Predictive Analytics
### Strategic Customer Retention, RFM Segmentation & Interpretable Machine Learning on Brazilian E-Commerce Data (Olist)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-Fast%20Packaging-purple.svg)](https://github.com/astral-sh/uv)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Data%20Engineering-Pandas%20%26%20Parquet-150458.svg)](https://pandas.pydata.org/)

An end-to-end data science and commercial analytics project analyzing **~100,000 real Brazilian e-commerce orders** from the Olist marketplace. 

This project bridges **Business Administration principles (Customer Lifetime Value, Unit Economics, Churn Mitigation)** with **Applied Machine Learning (K-Means, XGBoost, and SHAP Game Theory)** to diagnose customer attrition, segment customer cohorts, predict churn with an **0.9256 ROC-AUC**, and simulate an optimized retention campaign generating **R$ 1.71 million in net preserved revenue (184.2% ROI)**.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Data Engineering & Relational Architecture](#2-data-engineering--relational-architecture)
3. [Exploratory Data Analysis & Empirical Findings](#3-exploratory-data-analysis--empirical-findings)
4. [Customer Segmentation: RFM & K-Means Clustering](#4-customer-segmentation-rfm--k-means-clustering)
5. [Supervised Machine Learning & The Data Leakage Case Study](#5-supervised-machine-learning--the-data-leakage-case-study)
6. [Model Evaluation & Commercial Threshold Optimization](#6-model-evaluation--commercial-threshold-optimization)
7. [Explainable AI (XAI) with SHAP: Global & Individual Diagnostics](#7-explainable-ai-xai-with-shap-global--individual-diagnostics)
8. [Financial Simulation & P&L Impact Analysis](#8-financial-simulation--pl-impact-analysis)
9. [Methodological Limitations & Production Recommendations](#9-methodological-limitations--production-recommendations)
10. [Repository Structure & Quickstart Guide](#10-repository-structure--quickstart-guide)

---

## 1. Executive Summary & Problem Statement

### The Commercial Challenge: The "Leaky Bucket"
Acquiring new customers via paid digital marketing (Customer Acquisition Cost - CAC) is increasingly expensive in Brazilian retail. A healthy business survives by retaining customers so their cumulative spend (**Customer Lifetime Value - LTV**) comfortably outpaces acquisition costs ($LTV / CAC > 3\times$).

When auditing Olist's multi-year transaction history, we uncovered an alarming structural bottleneck:
* **Total Unique Customers:** 93,357
* **Repeat Buyers:** 2,801
* **Empirical Repeat Purchase Rate:** **Only 3.00%!**

Over **97% of customers made a single purchase and never returned**. Olist was operating as a transactional "leaky bucket," burning acquisition spend without building recurring customer equity.

### Project Goals:
1. **Diagnose Friction:** Identify which operational variables (delivery delays, freight surcharges, review scores) drive customer attrition.
2. **Segment the Base:** Uncover natural customer behavioral tiers using **RFM Analysis and K-Means Clustering**.
3. **Predict Churn:** Train a supervised classifier (**XGBoost**) to identify at-risk customers *before* they leave, strictly avoiding target leakage.
4. **Quantify ROI:** Replace arbitrary 0.50 probability thresholds with an **Economic Break-Even Cost Function** to maximize net profit.

---

## 2. Data Engineering & Relational Architecture

The raw dataset comprises **8 relational CSV tables (1.55+ million rows)**. A naive join across these tables causes **Cartesian explosion** (duplicating order records across multi-item baskets and multi-card split payments).

```text
[ olist_customers ] (customer_id <-> customer_unique_id)
       │
       ▼ (1:N)
 [ olist_orders ] ◄── (1:N) ── [ order_payments ] (Pre-aggregated to Order Level)
       │          ◄── (1:1) ── [ order_reviews ]  (Pre-aggregated to Order Level)
       ▼ (1:N)
[ order_items ] ─── (N:1) ── [ olist_products ] (Mapped to English Categories)
```

### The Primary Key Trap: `customer_id` vs. `customer_unique_id`
* **`customer_id`:** A transient transaction token generated per order. Grouping by this key falsely implies every order is from a brand-new buyer.
* **`customer_unique_id`:** The underlying human identifier (CPF/tax identifier proxy). All lifetime aggregations were strictly rolled up to this key.

### Key Feature Engineering:
1. **Order-Level Operational Metrics (`src/data_processing.py`):**
   * `actual_delivery_days`: Elapsed time from order purchase to customer doorstep ($(\text{delivered} - \text{purchased}) / 86400$).
   * `delivery_delay_days`: Difference between actual delivery and promised SLA date ($(\text{delivered} - \text{estimated}) / 86400$). Positive values denote late deliveries; negative values denote deliveries completed ahead of schedule.
   * `is_late_delivery`: Binary flag ($1$ if delay $> 0$, else $0$).
   * `freight_ratio`: Shipping burden relative to item value ($\frac{\text{freight\_value}}{\text{item\_value} + 0.001}$).
   * `is_negative_review`: Binary flag ($1$ if review score $\le 2$, else $0$).
2. **Customer-Level Aggregation:**
   * Consolidated 96,477 delivered orders into **93,357 unique customer records**.
   * Computed cumulative metrics: lifetime spend, order count, average freight ratio, worst delivery delay, and average review score.
3. **Churn Target Definition (`is_churned`):**
   * In non-contractual e-commerce, customers do not explicitly cancel. We established an empirical inactivity cutoff based on retail repurchasing cadences:
     $$\text{is\_churned} = 1 \quad \text{if} \quad \text{recency\_days} > 180 \quad \text{(6 months)}$$
   * **Resulting Class Distribution:**
     * **Active (0):** 38,125 customers (40.84%)
     * **Churned (1):** 55,232 customers (59.16%)
   * Stored in optimized Apache Parquet format (`data/processed/customer_features.parquet`).

---

## 3. Exploratory Data Analysis & Empirical Findings

Four core analyses were conducted using Matplotlib and Seaborn (`reports/figures/`):

### A. Monthly Growth & The Black Friday Seasonal Anomaly (`01_monthly_gmv_orders.png`)
* Olist grew exponentially from ~800 orders/month in January 2017 to a plateau of **6,500–7,000 orders/month** in 2018 (~R$ 1.1 million in monthly GMV).
* **Black Friday 2017 (November):** Peak demand surge of **7,289 orders** and **R$ 1,154,000 in GMV**.
* **December Slump:** Immediate drop in December 2017 (~5,500 orders) reflecting post-Black Friday purchasing exhaustion and Correios holiday delivery cutoffs.

### B. The Cohort Retention Matrix (`02_cohort_retention_heatmap.png`)
* Tracking monthly customer cohorts across 12 months of maturation revealed that **Month 1 retention across cohorts is strictly between 0.18% and 0.72%**.
* Subsequent months (Months 2–12) decay to **0.10%–0.40%**.
* *Statistical Anomaly Noted:* Cohort `2017-01` at Month 12 showed a localized spike to **0.70%**. This was diagnosed as the **Law of Small Numbers**: the early January 2017 cohort had only ~716 base customers, meaning a return of just 5 shoppers for 1-year annual replenishment caused a visible jump.

### C. "The Logistics Cliff": Delays vs. Customer Satisfaction (`03_delay_vs_review_score.png`)
* **On-time or early deliveries:** Customers are highly satisfied (average review score **4.16–4.31 stars**; 1-star reviews represent only **6.4%–7.6%**).
* **1 to 3 days late:** Mild frustration emerges (score dips to **3.78 stars**; 1-star share nearly doubles to **13.6%**).
* **4 to 7 days late (The Cliff):** Satisfaction **collapses to 2.39 stars**, with **51.7% of customers leaving 1-star reviews**.
* **> 7 days late:** Catastrophic failure (score hits **1.82 stars**, with **66.7% leaving 1-star ratings**).
* *Strategic Takeaway:* **3 days of delay** is the critical SLA boundary beyond which customer trust is permanently broken.

### D. Regional Disparities: Brazil's Logistics Divide (`04_regional_logistics_disparities.png`)
* **Southeast Logistics Hub (SP, RJ, MG, ES):** Average delivery is **10.8 days**; average freight is **R$ 19.84**.
* **Northeast:** Average delivery is **20.0 days** (+85% longer); freight is **R$ 35.99** (+81% higher).
* **North (Amazonas, Pará, etc.):** Average delivery is **22.6 days** (+109% longer); freight is **R$ 41.35** (+108% higher).
* *Strategic Takeaway:* Distance from Southeast fulfillment centers exposes North and Northeast customers to double the shipping costs and lead times.

### E. Financial Distributions & Correlation Limitations (`05_unit_economics_breakdown.png`, `06_correlation_heatmap.png`)
* **Basket Value Skew:** Median order spend is **R$ 105.28**, while mean spend is **R$ 159.86** (pulled upward by luxury/multi-item orders up to R$ 13,664).
* **Financing Dominance:** Credit cards account for **75.9%** of volume, followed by Boleto Bancário at **19.9%**.
* **The "10x Installment Spike":** While 32.6% pay upfront in 1x, a sharp secondary peak appears at **10 installments (7.0%)**, reflecting Brazilian retail's structural *"10x sem juros"* promotion threshold.
* **Non-Linear Justification:** Linear Pearson correlations between operational features and churn were weak ($|r| < 0.15$), empirically confirming that churn is driven by complex, non-linear interactions requiring gradient-boosted decision trees.

---

## 4. Customer Segmentation: RFM & K-Means Clustering

To move beyond demographic assumptions, we segmented the 93,357 customers based on **Recency ($R$)**, **Frequency ($F$)**, and **Monetary ($M$)** behavior (`src/segmentation.py`).

### Preprocessing for Euclidean Geometry:
1. **The Skewness Challenge:** Raw metrics exhibited severe right-skew:
   * Recency: skew = $0.45$ (balanced)
   * Frequency: skew = $11.10$ (extreme zero-variance spike; 97% are 1.0)
   * Monetary: skew = **$9.21$** (median R$ 107 vs. max R$ 13,664)
2. **Log Transformation:** Applied $\ln(1 + x)$ (`np.log1p`), successfully compressing the Monetary distribution into an almost textbook Gaussian bell curve and slashing skewness from **9.21 down to 0.53** (`07_rfm_transformation_distributions.png`).
3. **Standardization:** Transformed features with `StandardScaler` ($\mu = 0, \sigma = 1$) so that distance measurements treat Recency, Frequency, and Monetary with equal geometric weight.

### Hyperparameter Optimization ($k=4$):
* Evaluated $k \in [2, 7]$ across Inertia (Within-Cluster Sum of Squares) and Silhouette scores on a representative 15,000-customer sample (`08_kmeans_elbow_silhouette.png`).
* **The Elbow Inflection:** Inertia dropped sharply from $k=2$ ($190,212$) to $k=3$ ($129,257$) and bent at **$k=4$ ($82,921$)**, after which gains flattened ($k=5$ dropped only to $68,335$).
* At $k=4$, the model maintains a strong silhouette separation ($~0.40$) while establishing **four actionable commercial personas** (`09_rfm_customer_personas.png`):

| Persona Name | Customer Count | Customer Share | Avg Recency | Avg Orders ($F$) | Avg Spend ($M$) | Revenue Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **At-Risk High Spenders** | 32,270 | **34.6%** | 271.8 days | 1.00 | **R$ 295.51** | **61.83%** |
| **Champions / Loyal VIPs** | 2,801 | **3.0%** | 219.8 days | **2.11** | **R$ 308.59** | **5.60%** |
| **Active / Recent Buyers** | 16,026 | **17.2%** | **41.4 days** | 1.00 | R$ 133.66 | 13.89% |
| **Lost / Hibernating Shoppers** | 42,260 | **45.3%** | 286.8 days | 1.00 | R$ 68.15 | 18.67% |

### Key Business Takeaway: The Revenue Concentration Risk
The **At-Risk High Spenders** cohort represents **61.8% of Olist's total historical gross revenue (R$ 9.53M)**, yet has been dormant for an average of **271 days (~9 months)**. Retaining even a fraction of this segment represents the single highest commercial return in the business.

---

## 5. Supervised Machine Learning & The Data Leakage Case Study

### Experimental Design (`src/model_pipeline.py`)
* **Stratified Split:** 80% Train (74,685 customers) / 20% Test (18,672 customers), maintaining an exact **59.16% churn distribution** in both partitions.
* **Leakage Guardrail:** `recency_days`, timestamps, and customer IDs were strictly excluded from the feature matrix $X$.
* **Scikit-Learn `ColumnTransformer`:**
  * Numerical (12 features): `StandardScaler()`
  * Categorical (2 features): `OneHotEncoder(handle_unknown="ignore")` mapping 27 Brazilian states and Top 15 product categories, producing **55 encoded columns**.

---

### 🚨 Case Study: Identifying & Resolving Indirect Target Leakage

In our initial modeling iteration, we included our K-Means `persona` column as a categorical feature. 

#### 1. How the Leak Occurred:
* The target was defined as: $\text{is\_churned} = (\text{recency\_days} > 180)$.
* Although raw `recency_days` was dropped from $X$, `persona` was constructed using `log_recency`.
* The persona *"Active / Recent Buyers"* explicitly mapped to customers with average recency of 41 days.
* The classifier learned to reverse-engineer recency through the `persona` dummy variables, creating **indirect target leakage**.

#### 2. How We Caught It via SHAP:
When inspecting the initial SHAP Beeswarm and Waterfall plots, `Persona: Active / Recent Buyers` ranked as the **#1 most important feature company-wide** and exerted an extreme **$-6.59$ negative push** on individual churn scores. This disproportionate impact served as an immediate diagnostic red flag.

#### 3. The Resolution & Impact on Model Performance:
We stripped `persona` entirely from $X$, forcing the algorithms to predict churn **purely from operational friction and customer transaction behavior**. 

Here is the transparent comparison between the initial (leaked) and final (honest operational) models on the unseen test set:

| Model Iteration | Features ($X$) | Test Accuracy | Test Precision | Test Recall | Test F1 | Test **ROC-AUC** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Dummy Baseline** | None (Most Frequent) | 59.16% | 59.16% | 100.0% | 74.34% | **0.5000** |
| **2. Logistic Regression (Leaked)** | 59 cols (with Persona) | 76.38% | 71.98% | 98.37% | 83.13% | 0.7818 |
| **2. Logistic Regression (Honest)** | 55 cols (No Persona) | 66.24% | 67.10% | 84.24% | 74.70% | **0.6778** |
| **3. Random Forest (Leaked)** | 59 cols (with Persona) | 79.74% | 74.67% | 99.52% | 85.32% | 0.8757 |
| **3. Random Forest (Honest)** | 55 cols (No Persona) | 74.42% | 72.03% | 92.79% | 81.10% | **0.8301** |
| **4. Tuned XGBoost (Leaked)** | 59 cols (with Persona) | 87.70% | 85.32% | 95.67% | 90.20% | 0.9479 |
| **4. Tuned XGBoost (Honest - Final)** | 55 cols (No Persona) | **84.75%** | **84.80%** | **90.43%** | **87.52%** | **0.9256** |

#### Finding:
Stripping the leak reduced test accuracy slightly (from 87.7% to **84.75%**) and ROC-AUC from 0.9479 to **0.9256**. However, the remaining signal is **100% legitimate and actionable**: even without knowing how recently a customer bought, operational friction alone predicts churn with **over 92.5% discrimination power**.

---

## 6. Model Evaluation & Commercial Threshold Optimization

### Evaluation Dashboard (`reports/figures/10_model_evaluation_dashboard.png`)
Evaluated on **18,672 unseen test customers**:
* **Confusion Matrix (Default 0.50 Cutoff):**
  * True Negatives (Correctly Identified Active): **5,834 (31.2%)**
  * False Positives (False Alarms): **1,791 (9.6%)**
  * False Negatives (Missed Churners): **1,057 (5.7%)**
  * True Positives (Caught Churners): **9,990 (53.5%)**
* **Precision-Recall Curve:** Achieved an **Average Precision (PR-AUC) of 0.9436**, maintaining over 85% precision across high-recall operating points.
* **ROC Curve:** Smooth, steep trajectory yielding an **ROC-AUC of 0.9256**.

### Commercial Threshold Optimization:
Standard machine learning models default to an arbitrary probability threshold of $p = 0.50$. In business, this is rarely optimal because the cost of errors is asymmetric:
* **Cost of False Positive ($FP$):** Wasting a retention coupon on someone who was going to stay anyway ($\text{Cost} = \text{R\$ } 15.00$).
* **Cost of False Negative ($FN$):** Losing a customer forever ($\text{Lost Margin} = \text{Average LTV of R\$ } 165.20 \times 25\% \text{ Win-Back Rate} = \text{R\$ } 41.30$).

Because losing a customer costs **$2.75\times$ more** than sending an intervention coupon, we simulated net profit preserved across 81 distinct probability thresholds ($p \in [0.10, 0.90]$):

$$\text{Net Preserved Profit} = (\text{True Positives} \times \text{R\$ } 41.30) - ((\text{True Positives} + \text{False Positives}) \times \text{R\$ } 15.00)$$

* **Finding:** The profit-maximizing decision boundary shifted from **0.50 down to 0.40**.
* **Result:** At threshold **0.40**, the model proactively triggers retention outreach for any customer exhibiting $\ge 40\%$ churn probability, capturing more at-risk shoppers and maximizing net preserved profit at **R$ 239,234.20 on the test cohort alone**.

---

## 7. Explainable AI (XAI) with SHAP: Global & Individual Diagnostics

Using Tree-SHAP (`src/shap_explainability.py`), we computed exact game-theoretic Shapley values across 3,000 representative test samples (`11_shap_summary_beeswarm.png`, `12_shap_feature_importance_bar.png`).

### The Top 5 True Drivers of Churn:
1. **`Total Freight Paid` (Mean impact: 1.2061):** The cumulative shipping expenditure.
2. **`Avg Delivery Days` (Mean impact: 0.6818):** Overall logistics transit time.
3. **`Max Delivery Delay` (Mean impact: 0.5534):** Punctuality failure against promised SLA.
4. **`Avg Freight Ratio` (Mean impact: 0.1760):** Shipping cost relative to product value.
5. **`Total Items Bought` (Mean impact: 0.1380):** Multi-item order complexity.

### The Counter-Intuitive Freight Finding:
* On the Beeswarm plot, higher `Total Freight Paid` (red dots) shifted slightly **left** (reducing churn risk), while low freight paid (blue dots) extended far **right** (increasing churn risk).
* **The Commercial Explanation:** `Total Freight Paid` is a cumulative metric. Loyal, multi-order shoppers naturally accumulate higher total freight over time. Single-purchase churners bought one cheap item, paid minimal freight, and never returned.
* Conversely, on `Avg Freight Ratio` ($\frac{\text{Freight}}{\text{Item Value}}$), high relative freight burdens do push customers toward churn.

### Individual Customer Diagnostic Waterfalls:
* **Case A: The At-Risk High Spender (`13_shap_waterfall_at_risk_customer.png`):**
  * Predicted Churn Risk: **86.1%** (Ground Truth: Churned).
  * Historical Spend: R$ 335.18 | Delivery was 11.5 days early!
  * *Why did they churn?* **Silent / Organic Dormancy.** The waterfall plot reveals that large basket complexity (`Total Items Bought = 1.31` pushing $+0.64$) and single-purchase inactivity drove churn. Even satisfied customers leave if marketing provides no re-engagement.
* **Case B: The Retained / Loyal Customer (`14_shap_waterfall_loyal_customer.png`):**
  * Predicted Churn Risk: **1.5%** (Ground Truth: Active).
  * The waterfall plot displays a continuous cascade of **blue downward arrows** led by accumulated freight ($-2.34$), low basket size, and fast transit ($-0.42$), pulling the score from baseline $+0.386$ down to $-4.165$.

---

## 8. Financial Simulation & P&L Impact Analysis

To conclude the project, we executed an automated financial simulation (`src/financial_simulation.py`) across all **93,357 customers**, comparing three strategic scenarios:

### Macro Strategy Comparison:
1. **Strategy 1: Status Quo (Do Nothing):** Zero cost, zero preserved revenue. 55,232 customers churn unchecked.
2. **Strategy 2: Blanket "Spray-and-Pray" Marketing:** Blindly sends an automated R$ 15 discount coupon to all 93,357 accounts. Burns **R$ 1,400,355 in marketing budget**, preserving R$ 880,710 in net value (**62.9% ROI**).
3. **Strategy 3: ML-Targeted Precision (Our XGBoost Model @ 0.40/0.44 Threshold):**
   * Surgically targets **62,180 customers** exhibiting $\ge 44\%$ churn probability.
   * Reduces marketing spend to **R$ 932,700** (saving **R$ 467,655 in wasted coupons**).
   * Generates **R$ 1,718,261 in Net Profit Preserved**.
   * **ROI: 184.2%** (nearly **triples** the marketing efficiency of blanket campaigns!).

### Breakdown by Customer Persona (`15_financial_simulation_roi.png`):

| Persona | Total Customers | Customers Targeted | % Targeted | Avg LTV | Campaign Spend | Net Profit Preserved | Program ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active / Recent Buyers** | 16,026 | 2,761 | 17.2% | R$ 133.66 | R$ 41,415 | **R$ 50,847** | **122.8%** |
| **At-Risk High Spenders** | 32,270 | 25,115 | 77.8% | R$ 295.51 | R$ 376,725 | **R$ 1,478,708** | **392.5%** |
| **Champions / Loyal VIPs** | 2,801 | 1,977 | 70.6% | R$ 308.59 | R$ 29,655 | **R$ 122,865** | **414.3%** |
| **Lost / Hibernating** | 42,260 | 32,327 | 76.5% | R$ 68.15 | R$ 484,905 | **R$ 65,841** | **13.6%** |

### Strategic P&L Recommendations:
1. **Focus 85%+ of Retention Budget on At-Risk High Spenders & VIPs:** These two groups generate **R$ 1.60M out of the R$ 1.72M in net profit** with ROIs exceeding **390% to 410%**.
2. **Cease Paid Couponing for Lost / Hibernating Shoppers:** Targeting the dormant segment only yields a marginal **13.6% ROI** while consuming over R$ 484k in capital. Shift this group to zero-cost organic email re-engagement.
3. **Protect at-risk Active Buyers:** The model flags 17.2% of recent buyers who experienced delivery delays. Intervening early saves R$ 50k in profit before they slip into hibernation.

---

## 9. Methodological Limitations & Production Recommendations

To maintain complete scientific rigor, the following real-world limitations and architectural enhancements should be noted:

1. **Static Snapshot vs. Rolling Observation Window:**
   * *Limitation:* Features and target were computed over the full historical period. In a live enterprise environment, features should be computed over a historical window (e.g., $T_0 \rightarrow T_{\text{cutoff}}$) and churn evaluated over a future observation window (e.g., $T_{\text{cutoff}} \rightarrow T_{\text{cutoff}} + 90\text{d}$).
2. **Historical Spend as LTV Proxy:**
   * *Limitation:* The simulation uses historical cumulative spend as a proxy for customer value. A full production implementation would integrate a predictive Gamma-Gamma / BG-NBD model to estimate *future* residual cash flows.
3. **Promotional Assumptions:**
   * *Limitation:* The R$ 15 coupon cost and 25% win-back rate are industry-standard retail benchmarks. In production, these parameters should be calibrated via randomized **A/B testing**.

---

## 10. Repository Structure & Quickstart Guide

```text
ecommerce-customer-churn-ltv/
├── data/
│   ├── raw/                 # Original 8 Olist CSV tables
│   └── processed/           # Parquet datasets (customer_features, orders_clean)
├── models/                  # Serialized Scikit-Learn / XGBoost pipeline (.joblib)
├── reports/
│   └── figures/             # 15 publication-grade figures (300 DPI)
├── src/
│   ├── data_processing.py   # Relational merges, aggregations & feature engineering
│   ├── eda_temporal.py      # GMV growth & Cohort Retention matrix
│   ├── eda_logistics.py     # Logistics Cliff & Regional disparity charts
│   ├── eda_financials.py    # Unit economics, payment shares & correlation matrix
│   ├── segmentation.py      # RFM transformations & K-Means clustering (k=4)
│   ├── model_pipeline.py    # ColumnTransformer & anti-leakage stratified splitting
│   ├── baseline_models.py   # Dummy, Logistic Regression, Random Forest benchmarks
│   ├── train_xgboost.py     # RandomizedSearchCV cross-validation & model persistence
│   ├── model_evaluation.py  # Confusion matrix, ROC/PR curves & 0.40 threshold tuning
│   └── shap_explainability.py # Global Beeswarm, Bar & Individual Waterfall plots
├── pyproject.toml           # Project dependencies managed via uv
└── README.md
```

### Quickstart Execution:
```bash
# 1. Clone repository & install dependencies
git clone https://github.com/bruno-reichert/ecommerce-customer-churn-ltv.git
cd ecommerce-customer-churn-ltv
uv sync

# 2. Re-run relational data engineering pipeline
uv run python -m src.data_processing

# 3. Generate exploratory data analysis charts
uv run python -m src.eda_temporal
uv run python -m src.eda_logistics
uv run python -m src.eda_financials

# 4. Run RFM segmentation & K-Means clustering
uv run python -m src.segmentation

# 5. Execute baseline models & train champion XGBoost
uv run python -m src.baseline_models
uv run python -m src.train_xgboost

# 6. Run model evaluation, SHAP explainability & financial simulation
uv run python -m src.model_evaluation
uv run python -m src.shap_explainability
uv run python -m src.financial_simulation
```

---

## 👨‍💻 Author

**Bruno Benno Reichert**  
*Bachelor in Business Administration (UFRGS) | Junior AI Engineer & Python Back-end Developer*  
* [LinkedIn](https://linkedin.com/in/brunobennoreichert) • [GitHub](https://github.com/bruno-reichert) • [Portfolio](https://bruno-reichert.github.io/portfolio-page)
