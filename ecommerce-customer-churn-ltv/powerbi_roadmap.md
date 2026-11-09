# 🗺️ Power BI Implementation Roadmap: Olist E-Commerce Retention & Churn Analytics

> **Project Goal:** Translate the Olist data engineering pipeline, customer segmentation, and XGBoost churn models into an executive-grade, interactive 3-page Power BI dashboard modeled on a robust Star Schema.

---

## 🏗️ Architecture Overview
* **Data Model:** Star Schema (`fact_orders` connected to `dim_customers` and `dim_date`).
* **Semantic Layer:** Dedicated `_Measures` table containing 100% explicit DAX measures (no implicit column aggregations).
* **Narrative Arc:** 
  1. *Executive Overview* (Macro Commercial Health & The Leaky Bucket)
  2. *Logistics & Operations* (The Logistics Cliff & Review Score Impact)
  3. *ML Retention & ROI Simulator* (Predictive Churn Targeting vs. Blanket Marketing)

---

## 📅 Action Plan & Milestones

### 🧱 Phase 1: Data Modeling & Star Schema Setup
- [X] **1.1 Data Ingestion & Typing (Power Query):**
  - Import `data/powerbi/fact_orders.csv` and `data/powerbi/dim_customers.csv`.
  - Validate and enforce explicit column types (IDs as Text, Timestamps as Date/Time, Currency as Decimal, Flags as Whole Number).
  - Remove redundant/duplicate columns if present.
- [X] **1.2 Create Dynamic Date Dimension (`dim_date`):**
  - Generate a standardized calendar table via DAX covering `2016-01-01` to `2018-12-31`.
  - Add standard time attributes: `Year`, `Month`, `Month-Year`, `YearMonth (Sort Column)`, `Quarter`, `Day of Week`.
  - Mark as official Date Table in Power BI.
- [X] **1.3 Relationship Mapping (Model View):**
  - Build `1 : *` relationship: `dim_customers [customer_unique_id]` $\rightarrow$ `fact_orders [customer_unique_id]` (Single direction).
  - Build `1 : *` relationship: `dim_date [Date]` $\rightarrow$ `fact_orders [order_date]` (Single direction).
  - Hide foreign keys in `fact_orders` from Report View to enforce dimensional integrity.

---

### 🧮 Phase 2: Core DAX Metric Engine
- [X] **2.1 Base Measures Table Setup:**
  - Create a disconnected table named `_Measures` to house all business calculations cleanly.
- [X] **2.2 Volume & Revenue Measures:**
  - `[Total GMV]` $= \sum(\text{Payment Value})$
  - `[Total Orders]` $= \text{DISTINCTCOUNT}(\text{order\_id})$
  - `[Total Customers]` $= \text{DISTINCTCOUNT}(\text{customer\_unique\_id})$
  - `[Avg Order Value (AOV)]` $= \text{[Total GMV]} / \text{[Total Orders]}$
- [X] **2.3 Retention & Friction Diagnostics:**
  - `[Repeat Buyer Count]` (Customers with $> 1$ order)
  - `[Repeat Customer Rate %]` (Repeat Customers / Total Customers)
  - `[Churn Rate %]` (Churned Customers / Total Customers)
  - `[Late Delivery Rate %]` (Orders where `is_late_delivery = 1` / Total Orders)
  - `[Avg Delivery Lead Time]` $= \text{AVERAGE}(\text{actual\_delivery\_days})$
  - `[Avg Review Score]` $= \text{AVERAGE}(\text{review\_score})$
- [X] **2.4 ML Churn & Financial Simulation Measures:**
  - `[Customers At-Risk]` (Count of customers flagged for campaign intervention)
  - `[Campaign Cost]` $= \text{[Customers At-Risk]} \times \text{R\$\ } 15.00$
  - `[Gross Preserved LTV]` $= \text{[Customers At-Risk]} \times 25\% \times \text{[Customer Avg Spend]}$
  - `[Net Profit Preserved]` $= \text{[Gross Preserved LTV]} - \text{[Campaign Cost]}$
  - `[Retention ROI %]` $= \text{[Net Profit Preserved]} / \text{[Campaign Cost]}$

---

### 🎨 Phase 3: Dashboard Architecture & Page Design

#### 📊 Page 1: Executive Overview & Commercial Health
- [ ] **Header KPI Cards:** Total GMV, Total Unique Customers, Repeat Purchase Rate (3.0%), Baseline Churn Rate (59.2%).
- [ ] **Visual 1.1:** Monthly Order Volume & GMV Growth (Clustered Column + Line Chart highlighting Black Friday 2017).
- [ ] **Visual 1.2:** Customer Persona Contribution (Donut or Tree Map: Customer Volume % vs. Revenue Contribution %).
- [ ] **Visual 1.3:** The Revenue Concentration Reality (Card/Callout showing *At-Risk High Spenders* control 61.8% of GMV).

#### 🚚 Page 2: Logistics Friction & Customer Backlash
- [ ] **Header KPI Cards:** Avg Delivery Days (12.5d), Late Delivery Rate (%), Avg Review Score (4.15★).
- [ ] **Visual 2.1 (The Logistics Cliff):** Clustered Column + Line combo plotting Delivery Delay Buckets vs. Avg Review Score & % of 1-Star Reviews (proving the $>3$ day breaking point).
- [ ] **Visual 2.2 (Regional Disparities):** Brazilian Map / Bar Chart comparing Southeast vs. North/Northeast delivery lead times and freight surcharges.
- [ ] **Visual 2.3 (Shipping Burden):** Scatter plot or grouped bar of Freight Ratio vs. Review Score.

#### 🤖 Page 3: Machine Learning Retention & ROI Simulator
- [ ] **Header KPI Cards:** Targeted Customers, ML Campaign Cost, Net Revenue Preserved, Campaign ROI (184.2%).
- [ ] **Visual 3.1 (Strategic Comparison):** Grouped Bar Chart comparing **Status Quo** (R$ 0) vs. **Blanket Marketing** (62.9% ROI) vs. **ML-Targeted Precision** (184.2% ROI).
- [ ] **Visual 3.2 (Persona ROI Breakdown):** Bar chart showing Net Profit and ROI % across the 4 RFM personas (demonstrating why *Lost / Hibernating* should be skipped).
- [ ] **Visual 3.3 (High-Value Action Table):** Interactive grid of top at-risk customers (ID, Persona, Historical Spend, Churn Probability, Recommended Action).

---

### 🎛️ Phase 4: UX, Interactivity & Visual Polish
- [ ] **4.1 Unified Theme & Corporate Palette:**
  - Primary Navy: `#2c3e50` | Accent Blue: `#2980b9`
  - Success Green: `#27ae60` | Warning Amber: `#f39c12` | Alert Red: `#e74c3c`
  - Clean card borders, standardized fonts (Segoe UI), consistent padding.
- [ ] **4.2 Global Slicer Navigation:**
  - Date Range Slicer (tied to `dim_date`).
  - Customer Persona Slicer.
  - State / Region Slicer.
- [ ] **4.3 Tooltips & Drill-throughs:**
  - Create a custom Tooltip page showing RFM distribution when hovering over any Persona.
  - Drill-through capability from State down to City level.

---

### 📦 Phase 5: Documentation & Portfolio Artifacts
- [ ] **5.1 File Archival:** Save clean `.pbix` file into `reports/powerbi/`.
- [ ] **5.2 Export High-Res Presentation Assets:**
  - Export multi-page executive summary PDF.
  - Capture high-resolution PNGs of each dashboard page.
- [ ] **5.3 Documentation:**
  - Add Power BI Architecture section and visuals to root `README.md`.