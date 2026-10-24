"""
View: Overview (Engagement vs Churn)
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
    add_churn_baseline,
    COLORS,
    PROFILE_COLORS
)

df = load_data()
filtered_df = render_sidebar_filters(df)
check_empty_data(filtered_df)

st.title("📈 Engagement vs. Churn Overview")
st.markdown("Examine how customer behavioral engagement and activity indicators relate to retention across European markets.")

render_caution_banner(is_var_page=False)

# KPI Cards
n_total = len(filtered_df)
active_sub = filtered_df[filtered_df['IsActiveMember'] == 1]
inactive_sub = filtered_df[filtered_df['IsActiveMember'] == 0]

n_active = len(active_sub)
n_inactive = len(inactive_sub)

rate_active = (active_sub['Exited'].mean() * 100) if n_active > 0 else 0.0
rate_inactive = (inactive_sub['Exited'].mean() * 100) if n_inactive > 0 else 0.0
err_churn = (rate_inactive / rate_active) if rate_active > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Active Members", f"{n_active:,}", f"{(n_active/n_total*100):.1f}% of cohort")
with k2:
    render_kpi_card("Active Churn Rate", f"{rate_active:.2f}%", f"{int(active_sub['Exited'].sum()):,} churned")
with k3:
    render_kpi_card("Inactive Churn Rate", f"{rate_inactive:.2f}%", f"{int(inactive_sub['Exited'].sum()):,} churned")
with k4:
    render_kpi_card("ERR (Churn Ratio)", f"{err_churn:.2f}x", "Inactive churn / Active churn")

st.markdown("---")

# Chart 1: Active vs Inactive across Geography
st.markdown("### 🌍 Regional Engagement Disparity (Active vs. Inactive Churn)")
geo_act = filtered_df.groupby(['Geography', 'IsActiveMember'], observed=True).agg(
    Count=('CustomerId', 'count'),
    ChurnRate=('Exited', 'mean')
).reset_index()
geo_act['Activity'] = geo_act['IsActiveMember'].map({1: 'Active Member', 0: 'Inactive Member'})
geo_act['ChurnRatePct'] = geo_act['ChurnRate'] * 100

fig_geo = px.bar(
    geo_act,
    x='Geography',
    y='ChurnRatePct',
    color='Activity',
    barmode='group',
    text=geo_act.apply(lambda r: f"{r['ChurnRatePct']:.1f}% (n={r['Count']:,})", axis=1),
    color_discrete_map={'Active Member': '#2563EB', 'Inactive Member': '#DC2626'}
)
fig_geo.update_traces(textposition='outside')
fig_geo = apply_plotly_theme(fig_geo, "Churn Rate by Country and Member Activity Status")
fig_geo = add_churn_baseline(fig_geo, 20.37)
fig_geo.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="")
st.plotly_chart(fig_geo, use_container_width=True, theme=None)
st.caption("Bar labels show observed churn rate and customer count (n). Dashed line denotes portfolio baseline (20.37%).")

# Strategic Insight Note
st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 6px; font-size: 0.9rem; color: #1F2937; margin-bottom: 24px;">
        <strong>Empirical Finding:</strong> Across every European market, inactive members churn at 
        <strong>1.7 to 2.4 times</strong> the rate of active members. In Germany, dormant members experience 
        over 40% attrition, showing that inactivity is universally associated with heightened attrition regardless of geography.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Chart 2: Mutually Exclusive Engagement Profiles
st.markdown("### 👥 Mutually Exclusive Behavioral Profiles")
col_p1, col_p2 = st.columns(2)

with col_p1:
    prof_stats = filtered_df.groupby('Profile', observed=True).agg(
        Count=('CustomerId', 'count'),
        ChurnRate=('Exited', 'mean'),
        AvgBal=('Balance', 'mean')
    ).reset_index()
    prof_stats['ChurnRatePct'] = prof_stats['ChurnRate'] * 100
    
    fig_p1 = px.bar(
        prof_stats,
        x='Profile',
        y='ChurnRatePct',
        color='Profile',
        text=prof_stats.apply(lambda r: f"{r['ChurnRatePct']:.1f}% (n={r['Count']:,})", axis=1),
        color_discrete_map=PROFILE_COLORS
    )
    fig_p1.update_traces(textposition='outside')
    fig_p1 = apply_plotly_theme(fig_p1, "Churn Rate by Engagement Profile")
    fig_p1 = add_churn_baseline(fig_p1, 20.37)
    fig_p1.update_layout(showlegend=False, yaxis_title="Churn Rate (%)", xaxis_title="")
    st.plotly_chart(fig_p1, use_container_width=True, theme=None)
    st.caption("Bar labels show churn rate and customer sample size (n).")

with col_p2:
    fig_p2 = px.box(
        filtered_df,
        x='Profile',
        y='Balance',
        color='Profile',
        color_discrete_map=PROFILE_COLORS
    )
    fig_p2 = apply_plotly_theme(fig_p2, "Balance Distribution Across Engagement Profiles")
    fig_p2.update_layout(showlegend=False, yaxis_title="Account Balance", xaxis_title="")
    st.plotly_chart(fig_p2, use_container_width=True, theme=None)
    st.caption("Box plots display median, interquartile range (IQR), and balance spread per profile.")

# Profile Table
st.markdown("#### 📋 Profile Metric Summary Table")
prof_table = prof_stats.rename(columns={
    'Profile': 'Engagement Profile',
    'Count': 'Customer Count',
    'ChurnRatePct': 'Churn Rate (%)',
    'AvgBal': 'Average Balance'
})
prof_table['Churn Rate (%)'] = prof_table['Churn Rate (%)'].map(lambda x: f"{x:.2f}%")
prof_table['Average Balance'] = prof_table['Average Balance'].map(lambda x: f"{x:,.2f}")
prof_table['Share of Filtered Cohort'] = (prof_table['Customer Count'] / n_total * 100).map(lambda x: f"{x:.2f}%")

st.dataframe(prof_table[['Engagement Profile', 'Customer Count', 'Share of Filtered Cohort', 'Churn Rate (%)', 'Average Balance']], use_container_width=True, hide_index=True)
