# 📊 Power BI Dashboard: Customer Retention & Commercial Analytics

An interactive 3-page Power BI dashboard built on top of the processed Olist Brazilian e-commerce dataset (2016–2018). The dashboard translates the exploratory data analysis, customer segmentation (RFM + K-Means), and predictive churn model (XGBoost) into an accessible reporting interface for commercial and operational evaluation.

---

## 🏗️ Data Model & Relationships

The report uses a Star Schema centered on order transactions, filtered by customer and calendar dimensions:

| Relationship | From | To | Cardinality | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Orders by Customer** | `dim_customers [customer_unique_id]` | `fact_orders [customer_unique_id]` | 1 to Many (`1 : *`) | Single |
| **Orders by Date** | `dim_date [date]` | `fact_orders [order_date]` | 1 to Many (`1 : *`) | Single |

### Implementation Notes:
* **Calendar Scope:** `dim_date` was generated in DAX and bounded to `2017-01-01` through `2018-08-31` to match the mature operational window analyzed in the Python pipeline.
* **Separation of Measures:** All calculations are explicit DAX measures stored in a dedicated `_measures` table.

---

## 🖥️ Dashboard Structure & Key Findings

### Page 1: Commercial Health & The Leaky Bucket
* **Core Metrics:**
  * Total GMV: R$ 15.42M across 96,477 delivered orders
  * Unique Customers: 93,357
  * Repeat Customer Rate: **3.00%** (2,801 repeat buyers)
  * Baseline Churn Rate (>180 days inactive): **59.16%** (55,232 customers)
* **Visualizations:**
  * *Monthly GMV & Volume (Combo Chart):* Tracks growth from early 2017 to mid-2018, highlighting the November 2017 Black Friday volume surge.
  * *Revenue by Persona (Donut Chart):* Displays the revenue split among the 4 K-Means segments.
  * *Persona Commercial Diagnostic (Matrix):* Breaks down customer volume, total spend, AOV, and repeat rate per segment.
* **Key Insight:** Customer spend is heavily concentrated: **At-Risk High Spenders represent 34.6% of customers but account for 61.8% of total gross revenue (R$ 9.54M)**.

---

### Page 2: Operations & Logistics Friction
* **Core Metrics:**
  * Average Delivery Lead Time: **12.56 days**
  * Late Delivery Rate (vs. estimated date): **8.11%**
  * Average Review Score: **4.16 / 5.0**
  * Negative Review Rate (<= 2 stars): **12.68%**
* **Visualizations:**
  * *The Logistics Cliff (Combo Chart):* Plots delivery delay buckets against average rating and 1-star review share. On-time orders average ~4.2 stars. Once delivery is **> 3 days late**, average satisfaction falls to **2.39 stars**, and 1-star reviews exceed **50%** (reaching 66.7% for delays > 7 days).
  * *Regional Logistics Disparities (Dual-Axis Chart):* Compares transit time (days) and freight cost (R$) across Brazil's 5 macro-regions. The Southeast averages 10.8 days and R$ 19.84 freight, while the North/Northeast averages 20–22.6 days and R$ 36–41 freight.
  * *Macro-Region Performance Diagnostic (Matrix):* Table detailing orders, transit times, late rates, and freight by region.

---

### Page 3: Predictive Churn & Retention ROI Simulator
* **Simulation Assumptions:**
  * Model threshold: Churn probability >= 40% (flags 64,382 at-risk customers).
  * Intervention: R$ 15.00 discount voucher/outreach cost per targeted customer.
  * Win-back rate: 25% of targeted churners successfully retained.
  * Value preserved: 25% of the customer's historical lifetime spend.
* **Simulated Outcomes:**
  * Campaign Budget Required: R$ 965,730
  * Gross Preserved LTV: R$ 2,793,737
  * Net Preserved Profit: **R$ 1,828,007**
  * Simulated Campaign ROI: **189.29%**
* **Visualizations:**
  * *Head-to-Head Comparison (Bar Chart):* Compares Blanket Marketing (outreach to all 93k users: R$ 1.40M spend, R$ 880k net profit, 62.9% ROI) against ML-Targeted Precision (R$ 965k spend, R$ 1.83M net profit, 189.3% ROI).
  * *Persona Payoff (Combo Chart):* Shows that At-Risk High Spenders generate over 82% of the simulated net profit (R$ 1.5M+) at ~390% ROI, whereas Lost/Hibernating Shoppers barely cover campaign costs (~14% ROI).
  * *High-Value Target List (Table):* Filterable table of at-risk customers sorted by lifetime spend with predicted churn risk.

---

## 🎛️ Interactive Features

* **Custom Tooltip (`Tooltip_Persona`):** Hovering over any persona on Page 1 displays a dedicated mini-card showing that segment's specific Average Order Value, Repeat Rate, and Churn Rate.
* **Drill-Through (`Regional Deep-Dive`):** Right-clicking any region on Page 2 enables drill-through to a pre-filtered sub-page showing performance across individual Brazilian states (e.g., viewing transit times and late rates for all 9 Northeast states).
* **Cross-Filtering:** Selecting segments or regions across visuals dynamically filters connected metrics across the active canvas.

---

## 📐 Core DAX Measures (Reference)

```dax
-- Repeat Purchase Rate
repeat_customer_rate = 
DIVIDE(
    CALCULATE(DISTINCTCOUNT(dim_customers[customer_unique_id]), dim_customers[total_orders] > 1),
    DISTINCTCOUNT(dim_customers[customer_unique_id]),
    0
)

-- Simulated Campaign Cost (at R$ 15 / customer)
campaign_cost = [customers_at_risk] * 15.00

-- Simulated Gross LTV Preserved (25% win-back assumption)
gross_preserved_ltv = 
SUMX(
    FILTER(dim_customers, dim_customers[trigger_retention] = 1),
    dim_customers[total_spend] * 0.25
)

-- Simulated Net Preserved Profit
net_profit_preserved = [gross_preserved_ltv] - [campaign_cost]

-- Simulated Retention ROI
retention_roi = DIVIDE([net_profit_preserved], [campaign_cost], 0)