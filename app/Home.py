"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Unified Mentor Internship Project / Retail Banking Retention Analytics
Main Router Entrypoint with st.navigation
"""

import sys
from pathlib import Path
import os

# Ensure project root is in sys.path so 'app.*' and 'src.*' imports work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Customer Engagement & Retention Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Views directory
views_dir = os.path.join(os.path.dirname(__file__), "views")

# Define pages with icons under Dashboard heading
home_page = st.Page(os.path.join(views_dir, "home.py"), title="Home", icon="🏦", default=True)
overview_page = st.Page(os.path.join(views_dir, "overview.py"), title="Overview", icon="📈")
product_page = st.Page(os.path.join(views_dir, "product_utilization.py"), title="Product Utilization", icon="📦")
detector_page = st.Page(os.path.join(views_dir, "disengaged_detector.py"), title="Disengaged Detector", icon="🎯")
retention_page = st.Page(os.path.join(views_dir, "retention_scoring.py"), title="Retention Scoring", icon="🛡️")
simulator_page = st.Page(os.path.join(views_dir, "what_if_simulator.py"), title="What-If Simulator", icon="🔮")

# Configure Navigation with "Dashboard" section header
pg = st.navigation({
    "Dashboard": [
        home_page,
        overview_page,
        product_page,
        detector_page,
        retention_page,
        simulator_page
    ]
})

# Run the selected page
pg.run()
