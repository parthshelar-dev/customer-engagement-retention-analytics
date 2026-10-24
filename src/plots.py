"""
Module: plots.py
Project: Customer Engagement & Product Utilization Analytics for Retention Strategy
Context: Unified Mentor Internship Project / European Central Bank Regulatory Framing

Provides standardized, publication-grade plotting functions (matplotlib & seaborn)
configured for 300 DPI output, consistent corporate styling, and rigorous presentation.
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Cohesive corporate color palette
COLORS = {
    'primary': '#1f4e79',       # Deep Navy / ECB Blue
    'secondary': '#2e75b6',     # Slate Blue
    'accent_warn': '#d9534f',   # Churn / Alert Red
    'accent_safe': '#28a745',   # Retained Green
    'neutral_dark': '#2c3e50',  # Charcoal
    'neutral_light': '#ecf0f1', # Light Gray
    'neutral_mid': '#95a5a6',   # Mid Gray
    'gold': '#f39c12'           # Gold / High Value
}


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save matplotlib figure to disk at 300 DPI with tight bounding box.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_target_and_data_health(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Phase 1 data health and churn distribution overview.
    
    Panel 1: Ground-truth target balance (Retained vs. Churned).
    Panel 2: Behavioral binary distribution (Active Status & Card Ownership).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Panel 1: Churn Target
    churn_counts = df['Exited'].value_counts()
    n_total = len(df)
    labels = ['Retained (0)', 'Churned (1)']
    vals = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
    pcts = [vals[0] / n_total * 100, vals[1] / n_total * 100]
    bar_colors = [COLORS['primary'], COLORS['accent_warn']]
    
    bars1 = axes[0].bar(labels, vals, color=bar_colors, width=0.52, edgecolor='#ffffff', linewidth=1.5)
    axes[0].set_title('Ground-Truth Target Class Distribution', fontsize=13, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[0].set_ylabel('Number of Customers', fontsize=11, fontweight='semibold')
    axes[0].set_ylim(0, max(vals) * 1.18)
    
    for bar, val, pct in zip(bars1, vals, pcts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 140,
            f'{val:,}\n({pct:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=COLORS['neutral_dark']
        )
    
    # Panel 2: Engagement Indicators
    active_cnt = df['IsActiveMember'].value_counts()
    card_cnt = df['HasCrCard'].value_counts()
    
    categories = ['Active Member', 'Inactive Member', 'Credit Card Holder', 'No Credit Card']
    counts = [
        active_cnt.get(1, 0),
        active_cnt.get(0, 0),
        card_cnt.get(1, 0),
        card_cnt.get(0, 0)
    ]
    pct_list = [c / n_total * 100 for c in counts]
    bar_colors_2 = [COLORS['secondary'], COLORS['neutral_mid'], COLORS['secondary'], COLORS['neutral_mid']]
    
    bars2 = axes[1].barh(categories, counts, color=bar_colors_2, height=0.55, edgecolor='#ffffff', linewidth=1.2)
    axes[1].set_title('Customer Engagement & Banking Product Baseline', fontsize=13, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[1].set_xlabel('Number of Customers', fontsize=11, fontweight='semibold')
    axes[1].set_xlim(0, max(counts) * 1.22)
    axes[1].invert_yaxis()
    
    for bar, val, pct in zip(bars2, counts, pct_list):
        axes[1].text(
            bar.get_width() + 120,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,} ({pct:.1f}%)',
            ha='left', va='center', fontsize=10.5, fontweight='semibold', color=COLORS['neutral_dark']
        )
        
        plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_engagement_profiles(summary_df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Phase 2 Engagement Profiles:
    Panel 1: Churn Rate (%) and Customer Count across the 4 mutually exclusive profiles.
    Panel 2: Capital Exposure (Total Balance and Average Balance) by profile.
    
    Adheres strictly to no currency symbol rule for balances.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    profiles = summary_df['EngagementProfile'].tolist()
    churn_rates = summary_df['ChurnRatePct'].tolist()
    counts = summary_df['CustomerCount'].tolist()
    avg_balances = summary_df['AverageBalance'].tolist()
    total_balances_m = (summary_df['TotalBalance'] / 1e6).tolist()
    
    # Custom colors: Low risk = Green, Medium = Blue, High Disengaged = Orange, Silent Churn = Red
    color_map = [COLORS['accent_warn'], COLORS['gold'], COLORS['secondary'], COLORS['accent_safe']]
    
    # Panel 1: Churn Rates across Profiles
    bars1 = axes[0].bar(profiles, churn_rates, color=color_map, width=0.55, edgecolor='#ffffff', linewidth=1.5)
    axes[0].set_title('Churn Rate by Engagement Profile (Ordered by Risk Priority)', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[0].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='semibold')
    axes[0].set_ylim(0, max(churn_rates) * 1.25)
    axes[0].axhline(20.37, color='#7f8c8d', linestyle='--', linewidth=1.2, label='Cohort Baseline Churn (20.37%)')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].tick_params(axis='x', rotation=15)
    
    for bar, churn, count in zip(bars1, churn_rates, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{churn:.1f}%\n(n={count:,})',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['neutral_dark']
        )
        
    # Panel 2: Capital Exposure (Total Aggregate Balance in Millions)
    bars2 = axes[1].bar(profiles, total_balances_m, color=color_map, width=0.55, edgecolor='#ffffff', linewidth=1.5)
    axes[1].set_title('Aggregate Capital Exposure by Profile (Millions)', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[1].set_ylabel('Total Aggregate Balance (Millions)', fontsize=11, fontweight='semibold')
    axes[1].set_ylim(0, max(total_balances_m) * 1.25)
    axes[1].tick_params(axis='x', rotation=15)
    
    for bar, tot_m, avg_b in zip(bars2, total_balances_m, avg_balances):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4.0,
            f'{tot_m:.1f}M\n(Avg: {avg_b:,.0f})',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLORS['neutral_dark']
        )
        
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_product_utilization(depth_df: pd.DataFrame, anomaly_geo_df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Phase 3 Product Utilization:
    Panel 1: Product Depth vs Churn Curve with 95% bootstrap confidence interval error bars.
    Panel 2: Product Group Churn across Geography (France, Germany, Spain).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Panel 1: Product depth curve (products 1 to 4)
    # Filter only 1-4 products
    single_multi = depth_df[depth_df['Product Category'].str.contains(r'Product\(s\)')].copy()
    x_vals = [1, 2, 3, 4]
    y_vals = [float(v.replace('%', '')) for v in single_multi['Churn Rate']]
    ci_low = [float(v.replace('%', '')) for v in single_multi['CI 95 Pct Lower']]
    ci_high = [float(v.replace('%', '')) for v in single_multi['CI 95 Pct Upper']]
    counts = single_multi['Customer Count'].tolist()
    
    y_err_lower = [y - l for y, l in zip(y_vals, ci_low)]
    y_err_upper = [u - y for y, u in zip(y_vals, ci_high)]
    y_err = [y_err_lower, y_err_upper]
    
    axes[0].errorbar(
        x_vals, y_vals, yerr=y_err, fmt='-o', color=COLORS['primary'],
        ecolor=COLORS['accent_warn'], elinewidth=2.2, capsize=6, capthick=2,
        markersize=9, markerfacecolor=COLORS['secondary'], markeredgecolor='white',
        markeredgewidth=1.8, label='Churn Rate with 95% Bootstrap CI'
    )
    axes[0].axhline(20.37, color='#7f8c8d', linestyle='--', linewidth=1.2, label='Baseline Churn (20.37%)')
    axes[0].set_title('Product Depth vs. Churn Rate (with 95% Bootstrap CIs)', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[0].set_xlabel('Number of Bank Products Held', fontsize=11, fontweight='semibold')
    axes[0].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='semibold')
    axes[0].set_xticks(x_vals)
    axes[0].set_xticklabels(['1 Product', '2 Products\n(Optimal Anchor)', '3 Products\n(Anomaly)', '4 Products\n(100% Churn)'])
    axes[0].set_ylim(-2, 112)
    axes[0].legend(loc='upper left', frameon=True)
    
    for x, y, cnt, l, u in zip(x_vals, y_vals, counts, ci_low, ci_high):
        offset = 5 if y < 90 else -9
        axes[0].annotate(
            f'{y:.1f}%\n(n={cnt:,})\nCI:[{l:.1f}%, {u:.1f}%]',
            xy=(x, y), xytext=(x, y + offset),
            ha='center', va='bottom' if offset > 0 else 'top',
            fontsize=9, fontweight='bold', color=COLORS['neutral_dark']
        )
        
    # Panel 2: Product Group Churn across Geography
    # Prepare pivot table for bar chart
    geo_pivot = anomaly_geo_df.pivot(index='Geography', columns='ProductGroup', values='ChurnRate')
    # Convert string percentages to floats
    for col in geo_pivot.columns:
        geo_pivot[col] = geo_pivot[col].str.replace('%', '').astype(float)
        
    geo_pivot = geo_pivot[['1 Product', '2 Products', '3-4 Products']]
    x_indices = np.arange(len(geo_pivot.index))
    width = 0.25
    
    b1 = axes[1].bar(x_indices - width, geo_pivot['1 Product'], width=width, label='1 Product', color=COLORS['secondary'])
    b2 = axes[1].bar(x_indices, geo_pivot['2 Products'], width=width, label='2 Products (Sweet Spot)', color=COLORS['accent_safe'])
    b3 = axes[1].bar(x_indices + width, geo_pivot['3-4 Products'], width=width, label='3-4 Products (Alert)', color=COLORS['accent_warn'])
    
    axes[1].set_title('Churn Rate by Product Group Across European Markets', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[1].set_xlabel('Country / Market', fontsize=11, fontweight='semibold')
    axes[1].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='semibold')
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(geo_pivot.index, fontsize=11, fontweight='semibold')
    axes[1].set_ylim(0, 115)
    axes[1].legend(loc='upper left', frameon=True)
    
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2, height + 1.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold'
                )
                
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_financial_commitment_vs_engagement(
    pivot_churn: pd.DataFrame,
    pivot_counts: pd.DataFrame,
    loyalty_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Phase 4 Financial Commitment vs. Engagement:
    Panel 1: Churn Heatmap across Balance Tiers (D00 Zero Balance + 10 Funded Deciles) x Activity Status.
             Annotates churn rate and sample size, masking cells with n < 30.
    Panel 2: Does High Balance Ensure Loyalty? Bar chart comparing Active and Inactive members across
             lower vs. high balance with 95% bootstrap confidence intervals.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Panel 1: Balance Decile Heatmap
    # Reorder columns to Active Member, Inactive Member
    cols = ['Active Member', 'Inactive Member']
    churn_matrix = pivot_churn[cols]
    count_matrix = pivot_counts[cols]
    
    annot_matrix = np.empty(churn_matrix.shape, dtype=object)
    mask = np.zeros(churn_matrix.shape, dtype=bool)
    
    for r in range(churn_matrix.shape[0]):
        for c in range(churn_matrix.shape[1]):
            val = churn_matrix.iloc[r, c]
            cnt = count_matrix.iloc[r, c]
            if cnt < 30:
                mask[r, c] = True
                annot_matrix[r, c] = f"n={cnt}\n[Masked]"
            else:
                annot_matrix[r, c] = f"{val:.1f}%\n(n={cnt:,})"
                
    sns.heatmap(
        churn_matrix,
        annot=annot_matrix,
        fmt="",
        cmap='Blues',
        cbar_kws={'label': 'Churn Rate (%)'},
        linewidths=1.2,
        linecolor='white',
        ax=axes[0],
        mask=mask,
        annot_kws={'fontsize': 8.5, 'fontweight': 'bold'}
    )
    axes[0].set_title('Churn Rate Heatmap: Balance Tiers x Engagement Status\n(Note: cells contain ~300 customers; single-cell differences reflect sampling variance)', fontsize=11.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[0].set_ylabel('Balance Tier (Zero Balance + Funded Deciles)', fontsize=11, fontweight='semibold')
    axes[0].set_xlabel('Customer Engagement Status', fontsize=11, fontweight='semibold')
    axes[0].tick_params(axis='y', rotation=0)
    
    # Panel 2: Does High Balance Ensure Loyalty?
    cohort_labels = [
        'Active:\nLower/Zero Bal',
        'Active:\nHigh Balance',
        'Inactive:\nLower/Zero Bal',
        'Inactive:\nHigh Balance\n(Flight Risk)'
    ]
    churn_vals = [float(v.replace('%', '')) for v in loyalty_df['Churn Rate']]
    ci_low = [float(v.replace('%', '')) for v in loyalty_df['CI 95 Pct Lower']]
    ci_high = [float(v.replace('%', '')) for v in loyalty_df['CI 95 Pct Upper']]
    counts = loyalty_df['Customer Count'].tolist()
    
    y_err_lower = [y - l for y, l in zip(churn_vals, ci_low)]
    y_err_upper = [u - y for y, u in zip(churn_vals, ci_high)]
    y_err = [y_err_lower, y_err_upper]
    
    bar_colors = [COLORS['secondary'], COLORS['primary'], COLORS['gold'], COLORS['accent_warn']]
    bars = axes[1].bar(cohort_labels, churn_vals, color=bar_colors, width=0.55, edgecolor='#ffffff', linewidth=1.5, yerr=y_err, capsize=5)
    
    axes[1].axhline(20.37, color='#7f8c8d', linestyle='--', linewidth=1.2, label='Cohort Baseline Churn (20.37%)')
    axes[1].set_title('Does High Balance Ensure Loyalty? (With 95% Bootstrap CIs)', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[1].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='semibold')
    axes[1].set_ylim(0, 42)
    axes[1].legend(loc='upper left', frameon=True)
    
    for bar, churn, cnt, l, u in zip(bars, churn_vals, counts, ci_low, ci_high):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3.0,
            f'{churn:.1f}%\n(n={cnt:,})\n[{l:.1f}%-{u:.1f}%]',
            ha='center', va='bottom', fontsize=8.8, fontweight='bold', color=COLORS['neutral_dark']
        )
        
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_retention_strength_tiers(
    tier_summary: pd.DataFrame,
    age_cross_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Phase 5 Retention Strength Tiers:
    Panel 1: Churn Rate across Behavioral Retention Tiers (Tier 1 Core Sticky to Tier 4 Fragile)
             with 95% bootstrap confidence intervals.
    Panel 2: Tier Churn Rate across Demographic Age Bands, showing that behavioral tiers
             successfully stratify retention even within high-risk age segments (46-60).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    
    # Panel 1: Tiers Churn Curve & Confidence Intervals
    # If 6 rows (including 4a, 4b, and Combined) are provided, filter to the 4 primary tiers
    if 'Tier 4 Combined (Mixed)' in tier_summary['Retention Tier'].values:
        plot_df = tier_summary[tier_summary['Retention Tier'].isin([
            'Tier 1: Core Sticky', 'Tier 2: Latent Sticky', 'Tier 3: Vulnerable', 'Tier 4 Combined (Mixed)'
        ])].copy()
    else:
        plot_df = tier_summary.iloc[:4].copy()
        
    tier_labels = ['Tier 1:\nCore Sticky', 'Tier 2:\nLatent Sticky', 'Tier 3:\nVulnerable', 'Tier 4:\nFragile (Mixed)']
    churn_rates = [float(v.replace('%', '')) for v in plot_df['Churn Rate']]
    ci_low = [float(v.replace('%', '')) for v in plot_df['CI 95 Pct Lower']]
    ci_high = [float(v.replace('%', '')) for v in plot_df['CI 95 Pct Upper']]
    counts = plot_df['Customer Count'].tolist()
    
    y_err_lower = [y - l for y, l in zip(churn_rates, ci_low)]
    y_err_upper = [u - y for y, u in zip(churn_rates, ci_high)]
    y_err = [y_err_lower, y_err_upper]
    
    tier_colors = [COLORS['accent_safe'], COLORS['secondary'], COLORS['gold'], COLORS['accent_warn']]
    bars1 = axes[0].bar(tier_labels, churn_rates, color=tier_colors, width=0.55, edgecolor='#ffffff', linewidth=1.5, yerr=y_err, capsize=6)
    
    axes[0].axhline(20.37, color='#7f8c8d', linestyle='--', linewidth=1.2, label='Cohort Baseline Churn (20.37%)')
    axes[0].set_title('Retention Strength: Behavioral Engagement Tiers (with 95% Bootstrap CIs)', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[0].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='semibold')
    axes[0].set_ylim(0, 52)
    axes[0].legend(loc='upper left', frameon=True)
    
    for bar, churn, cnt, l, u in zip(bars1, churn_rates, counts, ci_low, ci_high):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3.2,
            f'{churn:.1f}%\n(n={cnt:,})\n[{l:.1f}%-{u:.1f}%]',
            ha='center', va='bottom', fontsize=9.2, fontweight='bold', color=COLORS['neutral_dark']
        )
        
    # Panel 2: Tiers Across Age Bands
    age_pivot = age_cross_df.pivot(index='AgeBand', columns='RetentionTier', values='Churn Rate')
    for col in age_pivot.columns:
        age_pivot[col] = age_pivot[col].str.replace('%', '').astype(float)
        
    ordered_cols = ['Tier 1: Core Sticky', 'Tier 2: Latent Sticky', 'Tier 3: Vulnerable', 'Tier 4: Fragile']
    age_pivot = age_pivot[ordered_cols]
    
    x_indices = np.arange(len(age_pivot.index))
    width = 0.20
    
    for i, (col, color) in enumerate(zip(ordered_cols, tier_colors)):
        bars_age = axes[1].bar(x_indices + (i - 1.5) * width, age_pivot[col], width=width, label=col, color=color, edgecolor='#ffffff', linewidth=1.2)
        for bar in bars_age:
            h = bar.get_height()
            if h > 0:
                axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.2, f'{h:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
    axes[1].set_title('Behavioral Tier Separation Across Demographic Age Bands', fontsize=12.5, fontweight='bold', pad=12, color=COLORS['neutral_dark'])
    axes[1].set_xlabel('Age Band', fontsize=11, fontweight='semibold')
    axes[1].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='semibold')
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(age_pivot.index, fontsize=10, fontweight='semibold')
    axes[1].set_ylim(0, 95)
    axes[1].legend(loc='upper left', frameon=True, fontsize=9)
    
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_balance_activity_matrix(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Convenience wrapper for plot_financial_commitment_vs_engagement."""
    from src.features import analyze_balance_activity_heatmap, evaluate_balance_loyalty
    _, pivot_churn, pivot_counts = analyze_balance_activity_heatmap(df)
    loyalty_all, _ = evaluate_balance_loyalty(df)
    return plot_financial_commitment_vs_engagement(pivot_churn, pivot_counts, loyalty_all, output_path=output_path)




