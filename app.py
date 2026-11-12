"""
Student Performance Prediction and Analysis System
Main Application Entry Point
"""

import streamlit as st
import pandas as pd

import auth
import data_manager
import ml_pipeline
import ml_engine
import styles

from views import (
    render_dataset_management_view,
    render_prediction_view,
    render_tree_explorer_view,
    render_model_evaluation_view,
)

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------
auth.init_db()
styles.apply_theme()

if "username" not in st.session_state:
    st.session_state.username = "Student"

if "current_nav" not in st.session_state:
    st.session_state.current_nav = "Dataset Management"

data_manager.init_dataset_state(st.session_state)

if "trained_dt_model" not in st.session_state or st.session_state.trained_dt_model is None:
    st.session_state.trained_dt_model = ml_engine.load_saved_decision_tree()
if "trained_bundle" not in st.session_state or st.session_state.trained_bundle is None:
    st.session_state.trained_bundle = st.session_state.trained_dt_model or ml_pipeline.load_model_bundle()


# -----------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">Student Performance</div>
        <div class="sidebar-brand-subtitle">Prediction & Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    bundle = st.session_state.get("trained_bundle") or st.session_state.get("trained_dt_model")
    if bundle is not None:
        acc_val = bundle.get("accuracy", 0.0) * 100
        depth_val = bundle.get("tree_depth", bundle.get("tree_info", {}).get("max_depth_actual", "N/A"))
        st.markdown(f"""
        <div style="background:#F0FDF4; border:1px solid rgba(22,163,74,0.3); border-radius:8px; padding:8px 12px; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:8px; height:8px; border-radius:50%; background:#16A34A; display:inline-block;"></span>
                <span style="color:#16A34A; font-weight:700; font-size:0.78rem;">Model Active: Decision Tree</span>
            </div>
            <div style="color:#64748B; font-size:0.75rem; margin-top:2px;">Accuracy: {acc_val:.1f}% &bull; Depth: {depth_val}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FFFBEB; border:1px solid rgba(217,119,6,0.3); border-radius:8px; padding:8px 12px; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="width:8px; height:8px; border-radius:50%; background:#D97706; display:inline-block;"></span>
                <span style="color:#D97706; font-weight:700; font-size:0.78rem;">Model Untrained</span>
            </div>
            <div style="color:#64748B; font-size:0.75rem; margin-top:2px;">Upload dataset & train first</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    nav_options = [
        "📂 Dataset Management",
        "🎯 Performance Prediction",
        "🌳 Decision Tree Explorer",
        "📊 Model Evaluation",
    ]

    current_idx = 0
    for idx, opt in enumerate(nav_options):
        if st.session_state.current_nav in opt:
            current_idx = idx
            break

    selected_nav_full = st.radio(
        label="Navigation",
        options=nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )

    # Extract nav name (after emoji + space)
    selected_nav = selected_nav_full.split(" ", 1)[1]
    st.session_state.current_nav = selected_nav


# -----------------------------------------------------------------
# PAGE ROUTER
# -----------------------------------------------------------------
if "Dataset Management" in selected_nav:
    render_dataset_management_view()
elif "Performance Prediction" in selected_nav:
    render_prediction_view()
elif "Decision Tree Explorer" in selected_nav:
    render_tree_explorer_view()
elif "Model Evaluation" in selected_nav:
    render_model_evaluation_view()
else:
    render_dataset_management_view()


# -----------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <div>
        <strong>Student Performance Prediction System</strong> &bull;
        Decision Tree Classifier
    </div>
    <div>
        Python &bull; Streamlit &bull; Scikit-Learn &bull; Pandas &bull; SQLite
    </div>
</div>
""", unsafe_allow_html=True)
