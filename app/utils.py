"""
Shared Utility Module for Streamlit Analytics Application
Customer Engagement & Product Utilization Analytics for Retention Strategy
Unified Mentor Internship Project / Retail Banking Retention Analytics
"""

import sys
from pathlib import Path
import os
from typing import Tuple, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib


# Universal Calm & Colorblind-Friendly Palette (Light Theme Optimized)
COLORS = {
    'primary': '#1E3A8A',       # Deep Navy / Blue Accent
    'secondary': '#2563EB',     # Royal Blue
    'accent_safe': '#2563EB',   # Retained / Low-Risk Base (Calm Blue)
    'accent_warn': '#DC2626',   # Churn / High-Risk Base (Crimson Red)
    'accent_amber': '#D97706',  # Latent / Vulnerable (Warm Amber)
    'neutral_dark': '#1F2937',  # Charcoal Text
    'neutral_light': '#FFFFFF', # Pure White
    'surface': '#F8FAFC',       # Very Light Slate
    'border': '#E5E7EB',        # Subtle Grey Border
    'grid': '#E5E7EB',          # Chart Gridlines
    'muted': '#6B7280'          # Subdued Slate Grey
}

PROFILE_COLORS = {
    'Active Engaged': '#2563EB',      # Core Active (Royal Blue)
    'Active Low-Product': '#3B82F6',  # Secondary Active (Slate Blue)
    'Inactive Latent': '#D97706',     # Latent Amber
    'Inactive Disengaged': '#DC2626'  # Disengaged Red
}

TIER_COLORS = {
    'Tier 1: Core Sticky (Active, 2 Products)': '#2563EB',
    'Tier 2: Latent Sticky (Inactive, 2 Products)': '#3B82F6',
    'Tier 3: Vulnerable (Active, 1 Product)': '#D97706',
    'Tier 4a: Inactive Shallow (1 Product)': '#DC2626',
    'Tier 4b: Friction Zone (3-4 Products)': '#991B1B',
    'Tier 4 Combined': '#B91C1C'
}


def inject_custom_css():
    """Inject universal light theme CSS ensuring pure white backgrounds, crisp text, and clean borders."""
    css = """
    <style>
        /* Force light theme container backgrounds */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #1F2937 !important;
        }
        
        /* Force sidebar to pure white with a crisp light-grey border */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E5E7EB !important;
        }
        section[data-testid="stSidebar"] > div {
            background-color: #FFFFFF !important;
        }
        
        /* Typography overrides to guarantee dark readable text everywhere */
        h1, h2, h3, h4, h5, h6 {
            color: #111827 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif !important;
            letter-spacing: -0.01em;
        }
        p, span, label, div {
            color: #1F2937;
        }
        
        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            color: #111827 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #4B5563 !important;
        }
        
        /* Table/Dataframe styling */
        [data-testid="stDataFrame"] {
            border: 1px solid #E5E7EB !important;
            border-radius: 6px !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load pre-scored dataset containing all computed features, tiers, and calibrated probabilities."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'bank_churn_scored.csv')
    df = pd.read_csv(data_path)
    return df


@st.cache_resource
def load_model():
    """Load calibrated best churn model pipeline."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_churn_model.joblib')
    return joblib.load(model_path)


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render standard interactive sidebar filters across all pages:
    - Appears below the navigation menu
    - Single-select dropdown for Engagement Profile (with 'All' default)
    - Single-select dropdown for Geography (with 'All' default)
    - Product Count Range Slider
    - Minimum Balance Threshold Slider
    - Minimum Salary Threshold Slider
    """
    inject_custom_css()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Customer Filters")
    st.sidebar.markdown(
        "<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid #2563EB; padding: 8px 12px; border-radius: 4px; font-size: 0.82rem; color: #1F2937; margin-bottom: 12px; line-height: 1.4;'>"
        "<strong style='color: #1E40AF;'>Notice:</strong> Associations from observational data, not proven causes."
        "</div>",
        unsafe_allow_html=True
    )
    
    # 1. Profile filter (Single-select dropdown with "All" default)
    raw_profiles = sorted(df['Profile'].dropna().unique().tolist())
    profile_options = ["All"] + raw_profiles
    selected_profile = st.sidebar.selectbox(
        "Engagement Profile",
        options=profile_options,
        index=0,
        help="Filter by customer behavioral engagement profile (or 'All' for entire portfolio)."
    )
    
    # 2. Geography filter (Single-select dropdown with "All" default)
    raw_geos = sorted(df['Geography'].dropna().unique().tolist())
    geo_options = ["All"] + raw_geos
    selected_geo = st.sidebar.selectbox(
        "Geography",
        options=geo_options,
        index=0,
        help="Filter by country market: France, Germany, Spain (or 'All' for all markets)."
    )
    
    # 3. Product Count Range Slider
    min_prod, max_prod = int(df['NumOfProducts'].min()), int(df['NumOfProducts'].max())
    selected_products = st.sidebar.slider(
        "Number of Products",
        min_value=min_prod,
        max_value=max_prod,
        value=(min_prod, max_prod),
        step=1,
        help="Filter customers by number of bank products held (1 to 4)."
    )
    
    # 4. Minimum Balance Slider
    max_bal = float(df['Balance'].max())
    min_balance = st.sidebar.slider(
        "Minimum Account Balance",
        min_value=0.0,
        max_value=float(np.ceil(max_bal / 10000.0) * 10000.0),
        value=0.0,
        step=5000.0,
        format="%.0f",
        help="Filter customers by minimum nominal account balance."
    )
    
    # 5. Minimum Salary Slider
    max_sal = float(df['EstimatedSalary'].max())
    min_salary = st.sidebar.slider(
        "Minimum Estimated Salary",
        min_value=0.0,
        max_value=float(np.ceil(max_sal / 10000.0) * 10000.0),
        value=0.0,
        step=5000.0,
        format="%.0f",
        help="Filter customers by minimum annual estimated salary."
    )
    
    # Apply filters safely ("All" means no filter)
    filtered = df.copy()
    if selected_profile != "All":
        filtered = filtered[filtered['Profile'] == selected_profile]
    if selected_geo != "All":
        filtered = filtered[filtered['Geography'] == selected_geo]
        
    filtered = filtered[
        (filtered['NumOfProducts'] >= selected_products[0]) &
        (filtered['NumOfProducts'] <= selected_products[1]) &
        (filtered['Balance'] >= min_balance) &
        (filtered['EstimatedSalary'] >= min_salary)
    ].copy()
    
    # Quick filter summary counter in sidebar
    st.sidebar.markdown(f"**Matched Customers:** {len(filtered):,} of {len(df):,} ({len(filtered)/len(df)*100:.1f}%)")
    
    return filtered


def check_empty_data(df: pd.DataFrame):
    """Graceful empty-state handler when filters yield zero rows."""
    if len(df) == 0:
        st.warning("⚠️ **No customers match the selected filter criteria.** Please broaden your filter parameters in the sidebar to view data.")
        st.stop()


def render_caution_banner(is_var_page: bool = False):
    """Render standardized observational and Value-at-Risk caution banners with soft grey/light-blue background."""
    if is_var_page:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #D97706; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 0.88rem; color: #1F2937; line-height: 1.45;">
                <strong style="color: #92400E;">Methodological Notice:</strong> 
                Associations derived from observational historical data, not proven causes. 
                <strong>Value-at-Risk (VaR)</strong> is a risk-weighted balance used strictly for ranking and relative sizing, not a 1-year financial loss forecast (the dataset contains no temporal time-period).
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #2563EB; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 0.88rem; color: #1F2937; line-height: 1.45;">
                <strong style="color: #1E40AF;">Analytical Notice:</strong> 
                Associations derived from observational historical data, not proven causes. 
                Any policy recommendations or cross-selling initiatives should be validated through controlled A/B pilot tests.
            </div>
            """,
            unsafe_allow_html=True
        )


def apply_plotly_theme(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply consistent modern corporate styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#111827", family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
            x=0.0,
            y=0.96
        ),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=45, r=40, t=60, b=45),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#1F2937", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=12, color="#1F2937"),
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            font_color="#111827"
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E5E7EB",
        zeroline=False,
        title_font=dict(size=13, color="#1F2937"),
        tickfont=dict(size=12, color="#1F2937")
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E5E7EB",
        zeroline=False,
        title_font=dict(size=13, color="#1F2937"),
        tickfont=dict(size=12, color="#1F2937")
    )
    return fig


def add_churn_baseline(fig: go.Figure, baseline: float = 20.37) -> go.Figure:
    """Add a thin dashed reference line for overall portfolio churn rate (20.37%) and set zero y-axis."""
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="#9CA3AF",
        line_width=1.5,
        annotation_text=f"Portfolio Churn Baseline ({baseline:.2f}%)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#4B5563")
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def render_kpi_card(label: str, value: str, subtext: str = ""):
    """Render a crisp executive KPI summary card with light theme styling."""
    card_html = f"""
    <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
        <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #4B5563; margin-bottom: 6px;">
            {label}
        </div>
        <div style="font-size: 1.65rem; font-weight: 700; color: #111827; line-height: 1.2;">
            {value}
        </div>
        <div style="font-size: 0.8rem; color: #6B7280; margin-top: 6px;">
            {subtext}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
