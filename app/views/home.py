"""
View: Home Executive Overview
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
    PROFILE_COLORS,
    TIER_COLORS
)

# Load pre-scored data
df = load_data()

# Render Standardized Sidebar Filters (appears below navigation menu)
filtered_df = render_sidebar_filters(df)
check_empty_data(filtered_df)

# Page Header
st.title("🏦 Customer Engagement & Product Utilization Analytics")
st.markdown(
    "**Unified Mentor Internship Project** | *Retail Banking Retention Analytics*  \n"
    "**Core Research Thesis:** Customer retention is associated with behavioral engagement and relationship depth, rather than account balance or demographics alone."
)

render_caution_banner(is_var_page=True)

# Executive KPI Ribbon
st.markdown("### 📊 Executive Portfolio Overview")
col1, col2, col3, col4, col5 = st.columns(5)

n_customers = len(filtered_df)
churned_cnt = int(filtered_df['Exited'].sum())
churn_rate = (churned_cnt / n_customers * 100) if n_customers > 0 else 0.0
tot_bal = float(filtered_df['Balance'].sum())
tot_var = float(filtered_df[filtered_df['Exited'] == 0]['ValueAtRisk'].sum())
avg_rsi = float(filtered_df['RSI'].mean()) if n_customers > 0 else 0.0

with col1:
    render_kpi_card("Filtered Cohort", f"{n_customers:,}", f"{(n_customers/len(df)*100):.1f}% of total portfolio")

with col2:
    render_kpi_card("Observed Churn", f"{churn_rate:.2f}%", f"{churned_cnt:,} churned accounts")

with col3:
    render_kpi_card("Total Balance", f"{tot_bal/1e6:.1f}M", f"Avg: {tot_bal/n_customers:,.0f}" if n_customers > 0 else "")

with col4:
    render_kpi_card("Value-at-Risk", f"{tot_var/1e6:.1f}M", "Risk-weighted balance (Exited = 0)")

with col5:
    render_kpi_card("Mean RSI Score", f"{avg_rsi:.1f} / 100", "Relationship Strength Index")

st.markdown("---")

# Portfolio Breakdown: Profiles & Geographies
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 👤 Churn by Behavioral Engagement Profile")
    prof_agg = filtered_df.groupby('Profile', observed=True).agg(
        Count=('CustomerId', 'count'),
        ChurnRate=('Exited', 'mean'),
        TotalBalance=('Balance', 'sum')
    ).reset_index()
    prof_agg['ChurnRatePct'] = prof_agg['ChurnRate'] * 100
    
    fig_prof = px.bar(
        prof_agg,
        x='Profile',
        y='ChurnRatePct',
        color='Profile',
        text=prof_agg.apply(lambda r: f"{r['ChurnRatePct']:.1f}% (n={r['Count']:,})", axis=1),
        color_discrete_map=PROFILE_COLORS
    )
    fig_prof.update_traces(textposition='outside')
    fig_prof = apply_plotly_theme(fig_prof, "Churn Rate by Engagement Profile")
    fig_prof = add_churn_baseline(fig_prof, 20.37)
    fig_prof.update_layout(showlegend=False, yaxis_title="Churn Rate (%)", xaxis_title="")
    st.plotly_chart(fig_prof, use_container_width=True, theme=None)
    st.caption("Bar labels show observed churn rate and customer count (n). Dashed line indicates portfolio baseline (20.37%).")

with col_right:
    st.markdown("#### 🌍 Market Exposure by Geography")
    geo_agg = filtered_df.groupby('Geography', observed=True).agg(
        Count=('CustomerId', 'count'),
        ChurnRate=('Exited', 'mean'),
        TotalBalance=('Balance', 'sum'),
        VaR=('ValueAtRisk', 'sum')
    ).reset_index()
    geo_agg['ChurnRatePct'] = geo_agg['ChurnRate'] * 100
    
    fig_geo = px.bar(
        geo_agg,
        x='Geography',
        y='Count',
        color='Geography',
        text=geo_agg.apply(lambda r: f"n={r['Count']:,} ({r['ChurnRatePct']:.1f}% churn)", axis=1),
        color_discrete_sequence=['#1E3A8A', '#2563EB', '#60A5FA']
    )
    fig_geo.update_traces(textposition='outside')
    fig_geo = apply_plotly_theme(fig_geo, "Customer Distribution Across European Markets")
    fig_geo.update_layout(showlegend=False, yaxis_title="Customer Count", xaxis_title="")
    fig_geo.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_geo, use_container_width=True, theme=None)
    st.caption("Bar labels show total customer volume (n) and observed churn rate by country.")

st.markdown("---")

# Strategic Module Navigation Directory
st.markdown("### 🧭 Strategic Module Directory")
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    st.info("**1. Overview**  \nDeep-dive into activity indicators, baseline distributions, and country churn differentials.")

with nav_col2:
    st.info("**2. Product Utilization**  \nAnalyze product depth curves, the 2-product sweet spot, and the 3-4 product friction alert.")

with nav_col3:
    st.info("**3. Disengaged Detector**  \nIdentify and rank high-balance dormant customers facing silent flight by Value-at-Risk.")

with nav_col4:
    st.info("**4. Retention Scoring**  \nExplore the Relationship Strength Index (RSI), naive benchmarks, and tier stability.")

with nav_col5:
    st.info("**5. What-If Simulator**  \nInteractive calibrated simulation of behavioral interventions and population counterfactuals.")
