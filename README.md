# Customer Engagement & Product Utilization Analytics for Retention Strategy

**Unified Mentor Internship Project**  
*Audience & Supervisory Framing:* Bank Leadership and Regulatory Stakeholders (European Central Bank Context)  
*Dataset:* 10,000 European Bank Customers  
*Ground Truth Churn Rate:* 20.37% (7,963 Retained, 2,037 Churned)

---

## 📌 Project Overview & Research Thesis
Banks frequently assume that affluent customers (high balance or high estimated salary) are inherently loyal. This project challenges that assumption, testing the core thesis:

> **Customer retention is primarily governed by behavioral engagement and relationship depth, rather than account balance or demographics alone.**

### Key Empirical Findings
- **High Balance Alone Does Not Ensure Loyalty:** When comparing funded accounts (Balance > 0), active members with balances >= 119,839.69 churn at 18.06% vs. 16.03% for lower-balance funded members. For inactive members, high-balance customers churn at 31.95% vs. 32.55% for lower-balance members. Balance size provides no protective insulation; active engagement is the governing factor.
- **Inactivity Multiplies Churn by 1.7 to 2.4x Across Every Market:** In France (1.79x to 1.96x), Germany (1.73x to 1.74x), and Spain (1.99x to 2.38x), inactive customers churn at nearly double the rate of active members across both high and lower funded balance levels.
- **Product Utilization Sweet Spot vs. Friction Zone:** Churn collapses to **7.58%** at 2 products (the optimal anchor), but surges to **82.71%** at 3 products and **100.00%** at 4 products (85.89% combined). However, only **46 retained customers** in the bank hold 3-4 products (fee fatigue and operational complexity are possible explanations we cannot test with this data). Banks should avoid pushing a third product until tested.
- **Behavioral vs. Demographic Information Content:** A paired bootstrap benchmark (5,000 resamples) confirms that Behavioral-only and Demographic-only models achieve comparable predictive power (Delta ROC-AUC: -0.0112, 95% CI: [-0.0445, +0.0232]; Delta PR-AUC: +0.0405, 95% CI: [-0.0218, +0.0999]). Because both CIs cross zero, neither is statistically superior in isolation. They are complementary: demographics sets non-actionable baseline risk, while behavior is directly actionable. Combining both achieves **0.8690 ROC-AUC** and **0.7171 PR-AUC**.
- **Value-at-Risk Definition:** The dataset has no time period (no observation window or timestamps), so Value-at-Risk is a **risk-weighted balance for ranking and relative sizing, not a forecast of money lost in a year**. Sizing strictly evaluates customers still at the bank (Exited = 0). All retained customers average 13.13% calibrated churn probability, the Broad Inactive High-Balance group averages 19.95%, and the Core Hollow Premium segment averages 24.20%.

---

## 📁 Repository Structure
```text
customer-engagement-retention/
  app/
    Home.py                   # Main router entrypoint using st.navigation ("Dashboard")
    utils.py                  # Caching, Plotly corporate styling, and shared sidebar filters
    views/
      home.py                 # Executive overview and module directory
      overview.py             # Engagement vs Churn Overview
      product_utilization.py  # Product depth curve, sweet spot & friction alert
      disengaged_detector.py  # High-value disengaged detector (Value-at-Risk ranking)
      retention_scoring.py    # Relationship Strength Index (RSI) & tier stability
      what_if_simulator.py    # Calibrated counterfactual simulator & population counterfactual
  data/
    raw/                      # Untouched raw dataset (European_Bank.csv)
    processed/
      bank_churn_cleaned.csv  # Cleaned dataset
      bank_churn_profiled.csv # Profiled dataset with mutually exclusive cohorts
      bank_churn_tiered.csv   # Retention-tiered dataset
      bank_churn_scored.csv   # Consolidated scored dataset consumed by Streamlit app
  models/
    best_churn_model.joblib   # Persisted CalibratedClassifierCV XGBoost pipeline
  notebooks/
    01_engagement_retention_analysis.ipynb # End-to-end reproducible research notebook
  reports/
    figures/                  # Publication figures at 300 DPI (SHAP, tiers, heatmap)
    tables/                   # Exported CSV and Markdown empirical tables
  src/
    data_loader.py            # Data ingestion and regulatory validation
    features.py               # Profiling, balance loyalty tests, and retention tiers
    kpis.py                   # The 5 executive retention KPIs (ERR, PDI, HBDR, CCSS, RSI)
    modeling.py               # ML training, calibration, paired bootstrap, U1-U6
    plots.py                  # Publication-grade plotting suite
  requirements.txt            # Pinned environment dependencies
  README.md                   # Project documentation and run steps
```

---

## ⚙️ Setup & Execution Instructions (Windows PowerShell)

### 1. Environment Setup
Open Windows PowerShell in the project root directory (`d:\UM\customer-engagement-retention`):

```powershell
# Create Python virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional: If script execution is restricted on your system, enable it for this process)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Upgrade pip and install exact pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Model & Scored Dataset
To verify that the persisted model loads cleanly in the virtual environment:

```powershell
python -c "import joblib; model = joblib.load('models/best_churn_model.joblib'); print('Model loaded successfully:', type(model))"
```

### 3. Launch Interactive Streamlit Application
```powershell
streamlit run app/Home.py
```

The application will open automatically in your browser at `http://localhost:8501`.

### 4. Run Reproducible Research Notebook
To run or inspect the end-to-end research notebook:
```powershell
jupyter notebook notebooks/01_engagement_retention_analysis.ipynb
```
Or execute top-to-bottom via Python:
```powershell
python scratch/build_full_notebook.py
```
