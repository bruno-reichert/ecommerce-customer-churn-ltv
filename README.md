# 📦 E-Commerce Customer Churn & LTV Predictive Analytics
### Strategic Customer Retention, RFM Segmentation & Interpretable Machine Learning on Brazilian E-Commerce Data (Olist)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-Fast%20Packaging-purple.svg)](https://github.com/astral-sh/uv)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Data%20Engineering-Pandas%20%26%20Parquet-150458.svg)](https://pandas.pydata.org/)

An end-to-end data science and commercial analytics project analyzing **~100,000 real Brazilian e-commerce orders** from the Olist marketplace. 

This project bridges **Business Administration principles (Customer Lifetime Value, Unit Economics, Churn Mitigation)** with **Applied Machine Learning (K-Means, XGBoost, and SHAP)** to diagnose customer attrition, segment customer cohorts, and predict churn (test ROC-AUC **0.9256** after removing a leaked feature; see the caveats in Sections 5 and 9), and to simulate a retention campaign under explicitly stated assumptions.
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
3. **Predict Churn:** Train a supervised classifier (**XGBoost**) to identify at-risk customers *before* they leave, testing for and removing target leakage (one indirect leak was found and removed; see Section 5).
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
* **Weak Linear Signals:** Linear Pearson correlations between operational features and churn were weak ($|r| < 0.15$), which suggests that churn is not explained by simple linear relationships and motivated testing gradient-boosted decision trees.

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
* At $k=4$, the model reaches a moderate silhouette score ($\approx 0.40$) while yielding **four interpretable commercial personas** (`09_rfm_customer_personas.png`):
> *Note:* "Champions / Loyal VIPs" are the 2,801 repeat buyers. Their average recency is 219.8 days, so the label describes purchase frequency, not current activity.

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
* **Leakage Guardrail:** `recency_days`, timestamps, and customer IDs were excluded from the feature matrix $X$. An indirect leak through `persona` was later found and removed (see the case study below). Residual time-proxy risk is discussed in Section 9.
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
Removing the leaked feature reduced test accuracy from 87.7% to **84.75%** and ROC-AUC from 0.9479 to **0.9256**.

**Caveat:** `recency_days` is excluded from $X$, but the churn label is a direct function of the purchase date, and 97% of customers bought only once. For these customers, "churned" largely means "their single purchase happened more than 180 days before the dataset ended". Operational features such as delivery times, delays and freight vary over calendar time, so they may partly act as proxies for *when* the purchase happened. The 0.9256 ROC-AUC should therefore be read as an **upper bound** on predictive performance until the model is validated with a time-based design (cutoff date, features before it, churn measured over a future window; see Section 9).

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
* **Cost of False Positive ($FP$):** Sending a retention coupon to someone who was going to stay anyway (assumed cost: R$ 15.00).
* **Cost of False Negative ($FN$):** Missing a customer who could have been won back (assumed expected win-back value: average historical spend of R$ 165.20 × an assumed 25% win-back rate = R$ 41.30).

Because a missed churner is assumed to cost **2.75× more** than a coupon, we simulated net preserved revenue across 81 probability thresholds ($p \in [0.10, 0.90]$):

$$\text{Net Preserved Revenue} = (\text{True Positives} \times \text{R\$ } 41.30) - ((\text{True Positives} + \text{False Positives}) \times \text{R\$ } 15.00)$$

* **Finding:** On the test set, the revenue-maximizing threshold was **0.40**, lower than the default 0.50.
* **Result:** At that threshold the simulated net preserved revenue on the test cohort is **R$ 239,234.20**.
* **Caveat:** The threshold was selected on the same test set used for reporting, so this figure is optimistic. In production it should be tuned on a separate validation set. The R$ 15 coupon cost and 25% win-back rate are **assumptions**, not measured values.

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
* **Possible Explanations (not tested):**
  * `Total Freight Paid` is cumulative, so repeat buyers accumulate more freight. However, only 3.0% of customers are repeat buyers, so this likely explains only part of the pattern.
  * Freight also varies with region, product mix and time period, so it may partly act as a proxy for purchase timing (see Section 9). It should be treated as a hypothesis for follow-up, not a conclusion.
* On `Avg Freight Ratio` ($\frac{\text{Freight}}{\text{Item Value}}$), high relative freight burdens do push predictions toward churn.
* Conversely, on `Avg Freight Ratio` ($\frac{\text{Freight}}{\text{Item Value}}$), high relative freight burdens do push customers toward churn.

### Individual Customer Diagnostic Waterfalls:
* **Case A: An At-Risk High Spender (`13_shap_waterfall_at_risk_customer.png`):**
  * Predicted churn risk: **86.1%** (ground truth: churned). Historical spend: R$ 335.18; delivery arrived 11.5 days early.
  * The waterfall shows that basket-related features (`Total Items Bought = 1.31`, contribution $+0.64$) and the absence of repeat purchases pushed the score up. This is a single illustrative case: SHAP explains the model's prediction, not the customer's actual reasons for leaving.
* **Case B: The Retained / Loyal Customer (`14_shap_waterfall_loyal_customer.png`):**
  * Predicted Churn Risk: **1.5%** (Ground Truth: Active).
  * The waterfall plot displays a continuous cascade of **blue downward arrows** led by accumulated freight ($-2.34$), low basket size, and fast transit ($-0.42$), pulling the score from baseline $+0.386$ down to $-4.165$.

---

## 8. Financial Simulation & P&L Impact Analysis

To conclude the project, we ran a simulation (`src/financial_simulation.py`) comparing three strategies on the **held-out test set**. All figures depend on assumed parameters (R$ 15 coupon cost, 25% win-back rate) and use **historical spend as a proxy for customer value**, so they illustrate relative efficiency rather than forecast real profit.

Both targeted strategies are credited by the same rule: *25% of the historical spend of targeted customers who actually churned, minus R$ 15 per targeted customer.*

1. **Strategy 1: Status Quo:** No spend, no customers recovered.
2. **Strategy 2: Blanket Coupons:** R$ 15 to every customer. Spend: **R$ [A]**. Net revenue preserved: **R$ [B]** (ROI **[C]%**).
3. **Strategy 3: Model-Targeted (threshold [t]):** Targets **[N]** customers. Spend: **R$ [D]**. Net revenue preserved: **R$ [E]** (ROI **[F]%**).

*[One sentence stating the honest result, e.g. "The targeted strategy achieves higher ROI with lower spend" or "Blanket preserves more total revenue but at lower ROI". Write whichever the numbers show.]*

### Breakdown by Customer Persona (`15_financial_simulation_roi.png`):

| Persona | Total Customers | Customers Targeted | % Targeted | Avg LTV | Campaign Spend | Net Revenue Preserved (assumed) | Program ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active / Recent Buyers** | 16,026 | 2,761 | 17.2% | R$ 133.66 | R$ 41,415 | **R$ 50,847** | **122.8%** |
| **At-Risk High Spenders** | 32,270 | 25,115 | 77.8% | R$ 295.51 | R$ 376,725 | **R$ 1,478,708** | **392.5%** |
| **Champions / Loyal VIPs** | 2,801 | 1,977 | 70.6% | R$ 308.59 | R$ 29,655 | **R$ 122,865** | **414.3%** |
| **Lost / Hibernating** | 42,260 | 32,327 | 76.5% | R$ 68.15 | R$ 484,905 | **R$ 65,841** | **13.6%** |

### Strategic Recommendations (illustrative, assumption-dependent):
1. **Focus 85%+ of Retention Budget on At-Risk High Spenders & VIPs:** These two groups generate **R$ 1.60M out of the R$ 1.72M in net profit** with ROIs exceeding **390% to 410%**.
2. **Reconsider paid couponing for Lost / Hibernating Shoppers:** Targeting the dormant segment only yields a marginal **13.6% ROI** while consuming over R$ 484k in capital. Shift this group to zero-cost organic email re-engagement.
3. **Protect at-risk Active Buyers:** The model flags 17.2% of recent buyers who experienced delivery delays. Intervening early saves R$ 50k in profit before they slip into hibernation.

---

## 9. Methodological Limitations & Production Recommendations

1. **Static snapshot and possible time proxies (most important):**
   * Features and the churn label were computed over the same full period, and the label is a function of purchase date. Since 97% of customers bought once, delivery, freight and category features may partly encode *when* the purchase happened, so the reported ROC-AUC (0.9256) is likely an upper bound. A production design would compute features up to a cutoff date and measure churn over a future window (e.g., 90 days).
2. **No temporal validation:** The split is stratified random, not time-based. A time-based split should be used to estimate real-world performance.
3. **Threshold selected on the test set:** The decision threshold was tuned on the same data used for reporting. It should be tuned on a validation set.
4. **Simulation scope:** Historical spend is used as a proxy for customer value (revenue, not margin), and the simulation is not an out-of-sample forecast. A full implementation would estimate future value (e.g., BG/NBD + Gamma-Gamma).
5. **Assumed parameters:** The R$ 15 coupon cost and 25% win-back rate are assumptions. They should be calibrated with randomized A/B tests.
6. **Interpretation caveats:** SHAP values describe the model's behavior, not causal reasons for churn. The freight finding is a hypothesis.

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
