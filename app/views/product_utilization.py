"""
View: Product Utilization Impact
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
    COLORS
)

df = load_data()
filtered_df = render_sidebar_filters(df)
check_empty_data(filtered_df)

st.title("📦 Product Utilization & Relationship Depth")
st.markdown("Analyze how the number of products held relates to customer churn, identifying the 2-product sweet spot and potential product friction risks.")

render_caution_banner(is_var_page=False)

# Prominent Sample Size Warning Banner
st.markdown(
    """
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #D97706; padding: 12px 18px; border-radius: 6px; font-size: 0.88rem; color: #1F2937; margin-bottom: 20px; line-height: 1.45;">
        <strong style="color: #92400E;">High-Product Sample Caution:</strong> 
        Only <strong>46 retained customers</strong> across the entire bank hold 3 or 4 products (36 hold 3 products, 10 hold 4 products). 
        The observed attrition rate above 80% reflects historical patterns and must be interpreted with caution. Cross-sell initiatives to a third product should be approached via controlled A/B pilot testing.
    </div>
    """,
    unsafe_allow_html=True
)

# KPI Metrics Across Product Tiers
p_stats = filtered_df.groupby('NumOfProducts', observed=True).agg(
    Count=('CustomerId', 'count'),
    Churned=('Exited', 'sum'),
    ChurnRate=('Exited', 'mean'),
    TotalBal=('Balance', 'sum'),
    AvgBal=('Balance', 'mean')
).reset_index()
p_stats['ChurnRatePct'] = p_stats['ChurnRate'] * 100

cols = st.columns(4)
for i, prod_num in enumerate([1, 2, 3, 4]):
    row = p_stats[p_stats['NumOfProducts'] == prod_num]
    with cols[i]:
        if len(row) > 0:
            cnt = int(row['Count'].values[0])
            churn_pct = float(row['ChurnRatePct'].values[0])
            small_note = " ⚠️ (n < 30)" if cnt < 30 else ""
            desc = "Sweet Spot" if prod_num == 2 else ("High Friction Zone" if prod_num >= 3 else "Shallow Adoption")
            render_kpi_card(f"{prod_num} Product(s){small_note}", f"{churn_pct:.1f}%", f"{cnt:,} accounts ({desc})")
        else:
            render_kpi_card(f"{prod_num} Product(s)", "N/A", "0 accounts in filter")

st.markdown("---")

# Chart 1: Product Depth Curve
col_c1, col_c2 = st.columns(2)

with col_c1:
    fig_depth = go.Figure()
    
    prod_palette = {1: '#3B82F6', 2: '#2563EB', 3: '#D97706', 4: '#DC2626'}
    
    fig_depth.add_trace(go.Bar(
        x=[f"{p} Product{'s' if p > 1 else ''}" for p in p_stats['NumOfProducts']],
        y=p_stats['ChurnRatePct'],
        text=p_stats.apply(lambda r: f"{r['ChurnRatePct']:.1f}% (n={r['Count']:,})" + (" ⚠️ n<30" if r['Count'] < 30 else ""), axis=1),
        textposition='outside',
        marker_color=[prod_palette.get(p, '#6B7280') for p in p_stats['NumOfProducts']]
    ))
    fig_depth = apply_plotly_theme(fig_depth, "Observed Churn Rate by Number of Products Held")
    fig_depth = add_churn_baseline(fig_depth, 20.37)
    fig_depth.update_layout(yaxis_title="Churn Rate (%)", yaxis_range=[0, 118], xaxis_title="")
    st.plotly_chart(fig_depth, use_container_width=True, theme=None)
    st.caption("2 products represents the lowest churn rate (sweet spot). Small sample badge ⚠️ marks groups with n < 30.")

with col_c2:
    # Cross Product Count with Active vs Inactive
    p_act = filtered_df.groupby(['NumOfProducts', 'IsActiveMember'], observed=True).agg(
        Count=('CustomerId', 'count'),
        ChurnRate=('Exited', 'mean')
    ).reset_index()
    p_act['Activity'] = p_act['IsActiveMember'].map({1: 'Active Member', 0: 'Inactive Member'})
    p_act['ChurnRatePct'] = p_act['ChurnRate'] * 100
    p_act['ProdLabel'] = p_act['NumOfProducts'].map(lambda x: f"{x} Prod")
    
    fig_p_act = px.bar(
        p_act,
        x='ProdLabel',
        y='ChurnRatePct',
        color='Activity',
        barmode='group',
        text=p_act.apply(lambda r: f"{r['ChurnRatePct']:.1f}% (n={r['Count']:,})", axis=1),
        color_discrete_map={'Active Member': '#2563EB', 'Inactive Member': '#DC2626'}
    )
    fig_p_act.update_traces(textposition='outside')
    fig_p_act = apply_plotly_theme(fig_p_act, "Product Depth Crossed with Member Activity Status")
    fig_p_act = add_churn_baseline(fig_p_act, 20.37)
    fig_p_act.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="")
    st.plotly_chart(fig_p_act, use_container_width=True, theme=None)
    st.caption("Active membership is associated with lower churn across all product tiers.")

st.markdown("---")

# Tabular Breakdown
st.markdown("### 📋 Product Utilization Breakdown Table")
table_display = p_stats.rename(columns={
    'NumOfProducts': 'Products Held',
    'Count': 'Customer Count',
    'Churned': 'Churned Customers',
    'ChurnRatePct': 'Churn Rate (%)',
    'AvgBal': 'Average Balance'
})
table_display['Share of Filtered Cohort'] = (table_display['Customer Count'] / len(filtered_df) * 100).map(lambda x: f"{x:.2f}%")
table_display['Churn Rate (%)'] = table_display['Churn Rate (%)'].map(lambda x: f"{x:.2f}%")
table_display['Average Balance'] = table_display['Average Balance'].map(lambda x: f"{x:,.2f}")
table_display['Sample Size Flag'] = table_display['Customer Count'].map(lambda x: "⚠️ Small Group (n < 30)" if x < 30 else "Robust Sample (n >= 30)")

st.dataframe(
    table_display[['Products Held', 'Customer Count', 'Share of Filtered Cohort', 'Churn Rate (%)', 'Average Balance', 'Sample Size Flag']],
    use_container_width=True,
    hide_index=True
)
