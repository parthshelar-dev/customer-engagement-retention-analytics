"""
View: High-Value Disengaged Customer Detector (Hollow Premium)
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
    render_sidebar_filters,
    check_empty_data,
    render_caution_banner,
    render_kpi_card,
    apply_plotly_theme,
    COLORS
)

df = load_data()
filtered_df = render_sidebar_filters(df)
check_empty_data(filtered_df)

st.title("🎯 High-Value Disengaged Customer Detector")
st.markdown(
    "Detect and prioritize **Hollow Premium** accounts: high-balance customers with low product depth "
    "and dormant activity who represent substantial silent capital flight risk."
)

render_caution_banner(is_var_page=True)

# Strict Hollow Premium Definition Filter: Retained ONLY (Exited == 0)
retained_df = filtered_df[filtered_df['Exited'] == 0].copy()

if len(retained_df) == 0:
    st.warning("⚠️ No retained customers exist under the current filter selection.")
    st.stop()

# Hollow Premium Definition: Inactive + Balance >= 119,839.69 + 1 Product + Retained
hollow_premium = retained_df[
    (retained_df['IsActiveMember'] == 0) &
    (retained_df['Balance'] >= 119839.69) &
    (retained_df['NumOfProducts'] == 1)
].copy()

# Sort by Value-at-Risk descending
hollow_premium = hollow_premium.sort_values(by='ValueAtRisk', ascending=False)

n_hollow = len(hollow_premium)
tot_bal_hollow = float(hollow_premium['Balance'].sum())
tot_var_hollow = float(hollow_premium['ValueAtRisk'].sum())
avg_prob_hollow = float(hollow_premium['OOF_Calibrated_Churn_Prob'].mean() * 100) if n_hollow > 0 else 0.0
avg_prob_all_retained = float(retained_df['OOF_Calibrated_Churn_Prob'].mean() * 100)

# Executive KPI Ribbon
st.markdown("### 💼 Hollow Premium Portfolio Exposure")
c1, c2, c3, c4 = st.columns(4)

with c1:
    render_kpi_card("Hollow Premium Base", f"{n_hollow:,}", f"{(n_hollow/len(retained_df)*100):.1f}% of retained accounts")
with c2:
    render_kpi_card("Nominal Capital at Risk", f"{tot_bal_hollow/1e6:.2f}M", f"Avg: {tot_bal_hollow/n_hollow:,.0f}" if n_hollow > 0 else "")
with c3:
    render_kpi_card("Expected Value-at-Risk", f"{tot_var_hollow/1e6:.2f}M", "Risk-weighted balance for ranking")
with c4:
    render_kpi_card("Avg Calibrated Churn Risk", f"{avg_prob_hollow:.2f}%", f"vs. {avg_prob_all_retained:.2f}% across all retained")

# Context Callout on Probabilities
st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; color: #1F2937; margin: 16px 0; line-height: 1.45;">
        <strong style="color: #1E40AF;">Probability Distribution Context:</strong> 
        The average calibrated churn probability across all <strong>retained customers</strong> (n = {len(retained_df):,}) is 
        <strong>{avg_prob_all_retained:.2f}%</strong> (13.13% across the entire bank). 
        In contrast, the <strong>Core Hollow Premium segment</strong> faces an average calibrated churn risk of 
        <strong>{avg_prob_hollow:.2f}%</strong> (24.20% bank-wide benchmark), reflecting elevated exposure driven by single-product inactivity.
        Hollow Premium is defined by observable business criteria; an explicit RSI threshold was omitted to avoid circular model coupling, though Hollow Premium accounts naturally exhibit depressed RSI scores.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Geographic Breakdown
st.markdown("### 🌍 Hollow Premium Capital Exposure by Geography")
geo_hollow = hollow_premium.groupby('Geography', observed=True).agg(
    CustomerCount=('CustomerId', 'count'),
    NominalBalance=('Balance', 'sum'),
    ValueAtRisk=('ValueAtRisk', 'sum'),
    AvgChurnProb=('OOF_Calibrated_Churn_Prob', 'mean'),
    AvgBalance=('Balance', 'mean')
).reset_index()

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_geo_var = px.bar(
        geo_hollow,
        x='Geography',
        y='ValueAtRisk',
        color='Geography',
        text=geo_hollow.apply(lambda r: f"VaR: {r['ValueAtRisk']/1e6:.2f}M (n={r['CustomerCount']:,})", axis=1),
        color_discrete_sequence=['#1E3A8A', '#2563EB', '#60A5FA']
    )
    fig_geo_var.update_traces(textposition='outside')
    fig_geo_var = apply_plotly_theme(fig_geo_var, "Value-at-Risk by Geography (Risk-Weighted Balance)")
    fig_geo_var.update_layout(showlegend=False, yaxis_title="Value-at-Risk", xaxis_title="")
    fig_geo_var.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_geo_var, use_container_width=True, theme=None)
    st.caption("Bar labels show total risk-weighted Value-at-Risk and account count (n).")

with col_g2:
    fig_geo_bal = px.bar(
        geo_hollow,
        x='Geography',
        y='NominalBalance',
        color='Geography',
        text=geo_hollow.apply(lambda r: f"Bal: {r['NominalBalance']/1e6:.2f}M", axis=1),
        color_discrete_sequence=['#1E3A8A', '#2563EB', '#60A5FA']
    )
    fig_geo_bal.update_traces(textposition='outside')
    fig_geo_bal = apply_plotly_theme(fig_geo_bal, "Total Nominal Balance by Geography")
    fig_geo_bal.update_layout(showlegend=False, yaxis_title="Nominal Balance", xaxis_title="")
    fig_geo_bal.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_geo_bal, use_container_width=True, theme=None)
    st.caption("Bar labels show total nominal account balance still to protect.")

st.markdown("---")

# Customer Ranked Table & Download
st.markdown("### 📋 Ranked Customer Target List (Prioritized by Value-at-Risk)")
st.markdown(
    "Customers currently still at the bank ranked by Value-at-Risk. "
    "**Privacy Notice:** Customer surnames are strictly redacted to comply with data privacy standards."
)

if n_hollow > 0:
    # Display table (NO Surname)
    cols_display = [
        'CustomerId', 'Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts',
        'HasCrCard', 'Balance', 'OOF_Calibrated_Churn_Prob', 'ValueAtRisk', 'RSI'
    ]
    
    table_view = hollow_premium[cols_display].copy()
    table_view['Calibrated Churn Prob (%)'] = (table_view['OOF_Calibrated_Churn_Prob'] * 100).map(lambda x: f"{x:.2f}%")
    table_view['Account Balance'] = table_view['Balance'].map(lambda x: f"{x:,.2f}")
    table_view['Value-at-Risk'] = table_view['ValueAtRisk'].map(lambda x: f"{x:,.2f}")
    table_view['Credit Card'] = table_view['HasCrCard'].map({1: 'Yes', 0: 'No'})
    
    show_cols = [
        'CustomerId', 'Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts',
        'Credit Card', 'Account Balance', 'Calibrated Churn Prob (%)', 'Value-at-Risk', 'RSI'
    ]
    
    st.dataframe(table_view[show_cols], use_container_width=True, hide_index=True)
    
    # Download Button (Strictly EXCLUDING Surname)
    download_cols = [
        'CustomerId', 'Geography', 'Gender', 'Age', 'Tenure', 'NumOfProducts',
        'HasCrCard', 'Balance', 'OOF_Calibrated_Churn_Prob', 'ValueAtRisk', 'RSI', 'Profile', 'RetentionTier'
    ]
    csv_bytes = hollow_premium[download_cols].to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Prioritized Target List (CSV - Privacy Protected)",
        data=csv_bytes,
        file_name="hollow_premium_target_list.csv",
        mime="text/csv",
        help="Download the filtered, ranked Hollow Premium target list without customer surnames."
    )
else:
    st.info("No customers meet the Hollow Premium criteria under the current sidebar filters.")
