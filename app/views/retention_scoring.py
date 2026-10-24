"""
View: Retention Strength Scoring & Relationship Strength Index (RSI)
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
    COLORS,
    TIER_COLORS
)

df = load_data()
filtered_df = render_sidebar_filters(df)
check_empty_data(filtered_df)

st.title("🛡️ Relationship Strength Index (RSI)")
st.markdown(
    "Evaluate customer retention probability via the **Relationship Strength Index (RSI)**: an interpretable "
    "0–100 scale representing 100 x predicted P(staying) derived from behavioral relationship indicators."
)

render_caution_banner(is_var_page=False)

# RSI Mathematical Formulation Note
st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #2563EB; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; color: #1F2937; margin-bottom: 16px; line-height: 1.45;">
        <strong style="color: #1E40AF;">RSI Definition & Baseline Equivalence:</strong> 
        RSI equals <strong>100 x predicted probability of staying</strong>, computed from an interpretable logistic regression model fit on training data. 
        Because logistic regression with an intercept mathematically constrains the sample mean of predicted probabilities to match the training empirical retention rate, 
        the portfolio-wide average RSI is exactly equal to the empirical retention rate (<strong>79.63 / 100</strong>).
    </div>
    """,
    unsafe_allow_html=True
)

# Portfolio RSI KPIs
avg_rsi = float(filtered_df['RSI'].mean())
median_rsi = float(filtered_df['RSI'].median())
sticky_pct = (filtered_df['RSI'] >= 80).mean() * 100
fragile_pct = (filtered_df['RSI'] < 50).mean() * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi_card("Mean RSI Score", f"{avg_rsi:.1f} / 100", f"Median: {median_rsi:.1f}")
with c2:
    render_kpi_card("Sticky Base (RSI >= 80)", f"{sticky_pct:.1f}%", f"{int((filtered_df['RSI'] >= 80).sum()):,} accounts")
with c3:
    render_kpi_card("Fragile Base (RSI < 50)", f"{fragile_pct:.1f}%", f"{int((filtered_df['RSI'] < 50).sum()):,} accounts")
with c4:
    render_kpi_card("RSI Model Test AUC", "0.7514", "Significantly outperforms naive heuristics")

st.markdown("---")

# Chart 1: RSI Distribution by Churn Status
col_d1, col_d2 = st.columns(2)

with col_d1:
    fig_hist = px.histogram(
        filtered_df,
        x='RSI',
        color='Exited',
        barmode='overlay',
        nbins=35,
        color_discrete_map={0: '#2563EB', 1: '#DC2626'},
        labels={'Exited': 'Outcome'}
    )
    fig_hist = apply_plotly_theme(fig_hist, "RSI Score Distribution by Churn Outcome")
    fig_hist.update_layout(
        xaxis_title="Relationship Strength Index (0 to 100)",
        yaxis_title="Customer Count"
    )
    fig_hist.update_yaxes(rangemode="tozero")
    # Relabel legend traces
    for trace, new_name in zip(fig_hist.data, ['Retained (0)', 'Churned (1)']):
        trace.name = new_name
    st.plotly_chart(fig_hist, use_container_width=True, theme=None)
    st.caption("Distribution shows distinct separation between retained accounts (blue, concentrated > 80) and churned accounts (red, concentrated < 50).")

with col_d2:
    # RSI by Retention Tier Box Plot
    fig_tier_box = px.box(
        filtered_df,
        x='RetentionTier',
        y='RSI',
        color='RetentionTier',
        color_discrete_map=TIER_COLORS
    )
    fig_tier_box = apply_plotly_theme(fig_tier_box, "RSI Score Spread Across Retention Tiers")
    fig_tier_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="RSI Score (0-100)")
    fig_tier_box.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_tier_box, use_container_width=True, theme=None)
    st.caption("Box plot illustrates clear rank ordering and monotonic decrease across retention tiers.")

st.markdown("---")

# Benchmark Section: Data-Driven RSI vs Naive Baselines
st.markdown("### 🔬 Empirical Formulation Benchmark (Paired Bootstrap Validation)")
st.markdown("Evaluating the discrimination power of the Data-Driven RSI against heuristic naive baselines on the unseen test set:")

bench_records = [
    {
        'RSI Formulation': 'U1 Data-Driven RSI (Trained Behavioral Weights)',
        'Formulation Logic': '100 x predicted P(staying) from logistic regression on behavioral features',
        'Test ROC-AUC': '0.7514 [0.7243 - 0.7775]',
        'Paired Delta vs. Data-Driven': 'Reference Model (0.0000)',
        'Paired 95% CI of Difference': 'Reference',
        'Statistical Verdict': 'Superior discrimination; captures non-linear product penalty'
    },
    {
        'RSI Formulation': 'Naive RSI 1: 4-Feature Heuristic',
        'Formulation Logic': 'Unweighted sum of Active, 2 Products, Card, and Tenure / 10',
        'Test ROC-AUC': '0.6874 [0.6597 - 0.7140]',
        'Paired Delta vs. Data-Driven': '+0.0640',
        'Paired 95% CI of Difference': '[0.0471 - 0.0814]',
        'Statistical Verdict': 'Significantly Inferior (CI excludes 0; misses 3-4 product friction)'
    },
    {
        'RSI Formulation': 'Naive RSI 2: Active + 2-Product Only',
        'Formulation Logic': 'Equal 50/50 weights on IsActiveMember and 2-Product flag only',
        'Test ROC-AUC': '0.7054 [0.6812 - 0.7295]',
        'Paired Delta vs. Data-Driven': '+0.0460',
        'Paired 95% CI of Difference': '[0.0322 - 0.0607]',
        'Statistical Verdict': 'Significantly Inferior (CI excludes 0; omits ratio and tenure)'
    }
]
bench_df = pd.DataFrame(bench_records)
st.dataframe(bench_df, use_container_width=True, hide_index=True)

# Strategic Insight Note
st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; color: #1F2937; margin-top: 14px; line-height: 1.45;">
        <strong style="color: #1E40AF;">Methodological Conclusion:</strong> 
        Because both paired bootstrap confidence intervals strictly exclude zero, the Data-Driven RSI provides statistically significant 
        superior discrimination over both naive unweighted formulas. Its predictive edge stems from properly assigning a negative weight 
        to multi-product friction (3-4 products) and calibrating the relative importance of digital activity over simple tenure.
    </div>
    """,
    unsafe_allow_html=True
)
