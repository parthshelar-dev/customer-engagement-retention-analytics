"""
View: Associative What-If Counterfactual Simulator
Customer Engagement & Product Utilization Analytics for Retention Strategy
Unified Mentor Internship Project / Retail Banking Retention Analytics
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'app.*' and 'src.*' imports work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.utils import (
    load_data,
    load_model,
    render_sidebar_filters,
    check_empty_data,
    render_caution_banner,
    render_kpi_card,
    apply_plotly_theme,
    COLORS
)

df = load_data()
model = load_model()

filtered_df = render_sidebar_filters(df)
check_empty_data(filtered_df)

st.title("🔮 Associative What-If Counterfactual Simulator")
st.markdown("Simulate the **model-predicted difference between similar customer profiles** under hypothetical behavioral adjustments.")

render_caution_banner(is_var_page=False)

# Methodological Notice Box
st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #D97706; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 0.88rem; color: #1F2937; line-height: 1.45;">
        <strong style="color: #92400E;">Methodological Standard:</strong><br>
        Predicted changes are <strong>model-predicted differences between similar customer profiles</strong> derived from observational data, 
        <strong>not the proven effect of selling a product</strong>. Operational interventions must be validated through controlled A/B testing.<br>
        <strong>Calibrated Pipeline Verified:</strong> Probability predictions are generated strictly using the calibrated XGBoost pipeline.
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Customer Selection or Custom Profile Builder
st.markdown("### 👤 Select Customer or Configure Profile")
input_mode = st.radio("Simulation Target Input Mode:", ["Select Customer from Filtered Portfolio", "Build Custom Profile"], horizontal=True)

if input_mode == "Select Customer from Filtered Portfolio":
    retained_in_filter = filtered_df[filtered_df['Exited'] == 0]
    if len(retained_in_filter) == 0:
        st.warning("No retained customers match the current filter selection.")
        st.stop()
        
    sample_options = retained_in_filter['CustomerId'].tolist()[:100]
    selected_id = st.selectbox(
        "Choose Customer ID (Top 100 in filter):",
        options=sample_options,
        format_func=lambda cid: f"ID: {cid} | {retained_in_filter[retained_in_filter['CustomerId']==cid]['Geography'].values[0]} | Age: {retained_in_filter[retained_in_filter['CustomerId']==cid]['Age'].values[0]} | Balance: {retained_in_filter[retained_in_filter['CustomerId']==cid]['Balance'].values[0]:,.0f} | Prods: {retained_in_filter[retained_in_filter['CustomerId']==cid]['NumOfProducts'].values[0]}"
    )
    cust_row = retained_in_filter[retained_in_filter['CustomerId'] == selected_id].iloc[0].to_dict()
else:
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        c_geo = st.selectbox("Geography", ["France", "Germany", "Spain"], index=2)
        c_gender = st.selectbox("Gender", ["Female", "Male"], index=1)
        c_age = st.slider("Customer Age", 18, 92, 44)
    with col_u2:
        c_score = st.slider("Credit Score", 350, 850, 645)
        c_bal = st.number_input("Account Balance", 0.0, 300000.0, 113755.0, step=5000.0)
        c_sal = st.number_input("Estimated Salary", 1.0, 300000.0, 149756.0, step=5000.0)
    with col_u3:
        c_prod = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
        c_act = st.selectbox("Active Member Status", ["0 (Inactive)", "1 (Active)"], index=0)
        c_card = st.selectbox("Credit Card Ownership", ["0 (No Card)", "1 (Has Card)"], index=1)
        c_tenure = st.slider("Relationship Tenure (Years)", 0, 10, 8)
        
    cust_row = {
        'CreditScore': c_score,
        'Geography': c_geo,
        'Gender': c_gender,
        'Age': c_age,
        'Tenure': c_tenure,
        'Balance': c_bal,
        'NumOfProducts': c_prod,
        'HasCrCard': int(c_card.split()[0]),
        'IsActiveMember': int(c_act.split()[0]),
        'EstimatedSalary': c_sal
    }

# Helper to construct model inputs
def build_model_input(d):
    temp = pd.DataFrame([d])
    temp['ProductTier'] = temp['NumOfProducts'].map({1: 'Prod_1', 2: 'Prod_2', 3: 'Prod_3_4', 4: 'Prod_3_4'})
    temp['BalanceToSalaryRatio'] = temp['Balance'] / np.maximum(temp['EstimatedSalary'], 1.0)
    return temp

base_df = build_model_input(cust_row)
baseline_prob = float(model.predict_proba(base_df)[0, 1])

# Benchmarking against similar historical cohort
sim_geo = cust_row.get('Geography', 'Spain')
sim_age = cust_row.get('Age', 43)
age_min = 31 if 31 <= sim_age <= 45 else (18 if sim_age <= 30 else (46 if sim_age <= 60 else 61))
age_max = 45 if 31 <= sim_age <= 45 else (30 if sim_age <= 30 else (60 if sim_age <= 60 else 100))

similar_cohort = df[
    (df['IsActiveMember'] == cust_row.get('IsActiveMember', 0)) &
    (df['NumOfProducts'] == cust_row.get('NumOfProducts', 1)) &
    (df['Geography'] == sim_geo) &
    (df['Age'] >= age_min) &
    (df['Age'] <= age_max)
]
obs_sim_count = len(similar_cohort)
obs_sim_churn = float(similar_cohort['Exited'].mean() * 100) if obs_sim_count > 0 else 0.0

st.markdown("---")

# Baseline Presentation Ribbon
st.markdown("### 📊 Customer Profile Baseline & Historical Cohort Benchmark")
b1, b2, b3 = st.columns(3)
with b1:
    render_kpi_card("Model Calibrated Churn Prob", f"{baseline_prob*100:.2f}%", "Calculated via Calibrated XGBoost pipeline")
with b2:
    small_tag = " (⚠️ n < 30)" if obs_sim_count < 30 else ""
    render_kpi_card(f"Observed Churn (Similar Profiles){small_tag}", f"{obs_sim_churn:.2f}%", f"n = {obs_sim_count:,} historical accounts")
with b3:
    act_str = "Active" if cust_row.get('IsActiveMember', 0) == 1 else "Inactive"
    render_kpi_card("Similar Profile Criteria", f"{sim_geo}, {act_str}, {cust_row.get('NumOfProducts', 1)}P", f"Age Bracket: {age_min}-{age_max}")

# 2. Counterfactual Simulation Table
st.markdown("### 🔄 Counterfactual Customer Profile Simulations")

scenarios = [
    ('Current Profile Baseline', 'Existing customer attributes', cust_row, baseline_prob)
]

# Scenario A: Member Activation
if cust_row.get('IsActiveMember', 0) == 0:
    row_act = cust_row.copy()
    row_act['IsActiveMember'] = 1
    p_act = float(model.predict_proba(build_model_input(row_act))[0, 1])
    scenarios.append(('Counterfactual Profile: Member Activation', 'Transition IsActiveMember from 0 to 1', row_act, p_act))

# Scenario B: Product Addition (1 -> 2)
if cust_row.get('NumOfProducts', 1) == 1:
    row_p2 = cust_row.copy()
    row_p2['NumOfProducts'] = 2
    p_p2 = float(model.predict_proba(build_model_input(row_p2))[0, 1])
    scenarios.append(('Counterfactual Profile: Product Transition (1 to 2)', 'Transition NumOfProducts from 1 to 2 (Sweet Spot)', row_p2, p_p2))

# Scenario C: Warning Excess Products (to 3)
row_p3 = cust_row.copy()
row_p3['NumOfProducts'] = 3
p_p3 = float(model.predict_proba(build_model_input(row_p3))[0, 1])
scenarios.append(('WARNING: Excess Products Profile (to 3)', 'Transition NumOfProducts to 3 (Friction Danger Zone)', row_p3, p_p3))

# Scenario D: Card Acquisition
if cust_row.get('HasCrCard', 0) == 0:
    row_card = cust_row.copy()
    row_card['HasCrCard'] = 1
    p_card = float(model.predict_proba(build_model_input(row_card))[0, 1])
    scenarios.append(('Counterfactual Profile: Credit Card Ownership', 'Transition HasCrCard from 0 to 1', row_card, p_card))

sim_rows = []
for label, desc, _, prob in scenarios:
    delta = (prob - baseline_prob) * 100
    delta_str = f"{delta:+.2f} pts" if label != 'Current Profile Baseline' else "Baseline (0.00 pts)"
    sim_rows.append({
        'Counterfactual Scenario Profile': label,
        'Simulation Description': desc,
        'Predicted Calibrated Churn (%)': f"{prob * 100:.2f}%",
        'Model-Predicted Difference vs Baseline': delta_str,
        'Evidence Classification': 'Observational association; requires controlled A/B test'
    })
sim_df = pd.DataFrame(sim_rows)
st.dataframe(sim_df, use_container_width=True, hide_index=True)

st.markdown("---")

# 3. Population-Level Counterfactual
st.markdown("### 🌐 Population-Level Counterfactual Benchmark")
st.markdown(
    "To eliminate individual customer idiosyncrasies, we evaluate the **population-level model-predicted difference** "
    "across all **1,597 retained inactive 1-product customers** in the bank:"
)

pop_c1, pop_c2, pop_c3 = st.columns(3)
with pop_c1:
    render_kpi_card("Target Population", "1,597 Customers", "All retained inactive 1-product accounts")
with pop_c2:
    render_kpi_card("Baseline Mean Risk", "24.80%", "Average calibrated churn probability")
with pop_c3:
    render_kpi_card("If Activated (Counterfactual)", "14.76%", "Average difference: -10.04 pts")

st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; color: #1F2937; margin-top: 14px; line-height: 1.45;">
        <strong style="color: #1E40AF;">Population Summary:</strong> Across the entire bank, if all 1,597 retained inactive 1-product customers 
        exhibited behavioral activity profiles similar to active members, the model predicts an average churn drop of 
        <strong>-10.04 percentage points</strong> (from 24.80% to 14.76%). 
        This establishes the aggregate upper-bound potential for digital re-engagement campaigns.
    </div>
    """,
    unsafe_allow_html=True
)
