"""
What-If Simulator View
Student Performance Prediction and Analysis System
Study pattern simulator, multi-attribute sensitivity curves, and category transition modeling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml_engine
from styles import render_page_header, render_active_dataset_badge, render_metric_card

CLASS_COLORS = {
    "Excellent": "#10B981",
    "Good": "#3B82F6",
    "Average": "#F59E0B",
    "Needs Improvement": "#EF4444",
}


def render_what_if_simulator_view():
    """Render What-If Study Pattern Simulator interface."""
    render_page_header(
        title="What-If Study Pattern Simulator",
        subtitle="Simulate the impact of modifying academic habits, attendance, and preparation on predicted performance categories.",
        badge="Scenario Modeling"
    )

    render_active_dataset_badge(st.session_state)

    trained_model = st.session_state.get("trained_dt_model")
    if trained_model is None:
        saved = ml_engine.load_saved_decision_tree()
        if saved is not None:
            st.session_state.trained_dt_model = saved
            trained_model = saved

    if trained_model is None:
        st.warning("⚠️ No trained Decision Tree model found. Please train a model in **Model Evaluation** to enable scenario simulations.")
        return

    active_df = getattr(st.session_state, "active_df", None)
    pipeline = trained_model["pipeline"]
    feature_columns = trained_model["feature_columns"]
    num_features = trained_model["numerical_columns"]
    classes = trained_model["classes"]

    if not num_features:
        st.error("No numerical features available for continuous parameter simulation.")
        return

    # -------------------------------------------------------------
    # 1. PARAMETER SELECTION & BASELINE SETUP
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Simulation Controls")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
        "Select which continuous academic parameter to alter, and configure the student baseline values."
        "</p>",
        unsafe_allow_html=True
    )

    col_target_param, col_baseline_summary = st.columns([1, 2])

    with col_target_param:
        default_idx = 0
        if "study_hours" in num_features:
            default_idx = num_features.index("study_hours")
        elif "attendance" in num_features:
            default_idx = num_features.index("attendance")

        sim_feature = st.selectbox(
            "Parameter to Simulate",
            options=num_features,
            index=default_idx,
            help="Choose the numerical variable to sweep across a range."
        )

    # Determine range for chosen feature
    if active_df is not None and sim_feature in active_df.columns:
        s_min = float(active_df[sim_feature].min())
        s_max = float(active_df[sim_feature].max())
    else:
        s_min, s_max = 0.0, 10.0

    if s_min == s_max:
        s_max = s_min + 10.0

    # Build baseline dictionary
    latest_inputs = st.session_state.get("latest_pred_inputs", {})
    baseline_inputs = {}

    for f in feature_columns:
        if f in latest_inputs:
            baseline_inputs[f] = latest_inputs[f]
        elif active_df is not None and f in active_df.columns:
            if pd.api.types.is_numeric_dtype(active_df[f]):
                baseline_inputs[f] = float(active_df[f].median())
            else:
                baseline_inputs[f] = str(active_df[f].dropna().iloc[0]) if not active_df[f].dropna().empty else "Standard"
        else:
            baseline_inputs[f] = 0.0

    with col_baseline_summary:
        st.markdown(
            f"<div style='background:#0E1626; border:1px solid rgba(148,163,184,0.1); border-radius:8px; padding:10px 14px; margin-top:4px; font-size:0.83rem; color:#94A3B8;'>"
            f"<b style='color:#FFFFFF;'>Simulating:</b> <span style='color:#93C5FD;'>{sim_feature.replace('_', ' ').title()}</span> "
            f"from <b>{s_min:.1f}</b> to <b>{s_max:.1f}</b> while holding other {len(feature_columns)-1} variables fixed."
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. RUN SIMULATION & PLOT PROBABILITY TRAJECTORY
    # -------------------------------------------------------------
    param_steps = np.linspace(s_min, s_max, 25).tolist()
    sim_results_df = ml_engine.simulate_what_if_trajectory(
        pipeline=pipeline,
        feature_columns=feature_columns,
        base_inputs=baseline_inputs,
        param_name=sim_feature,
        param_range=param_steps
    )

    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Category Probability Trajectory Chart")
    st.markdown(
        f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
        f"Dynamic shifts in predicted category likelihood as <b>{sim_feature.replace('_', ' ').title()}</b> varies:"
        "</p>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#0E1626")
    ax.set_facecolor("#0E1626")

    for cls in classes:
        prob_col = f"Prob_{cls}"
        if prob_col in sim_results_df.columns:
            line_color = CLASS_COLORS.get(cls, "#93C5FD")
            ax.plot(
                sim_results_df[sim_feature],
                sim_results_df[prob_col],
                label=cls,
                color=line_color,
                linewidth=2.5,
                marker="o",
                markersize=4
            )

    ax.set_xlabel(sim_feature.replace("_", " ").title(), color="#94A3B8", fontsize=10, labelpad=8)
    ax.set_ylabel("Predicted Probability (%)", color="#94A3B8", fontsize=10, labelpad=8)
    ax.set_title(f"Sensitivity Analysis: Impact of {sim_feature.replace('_', ' ').title()}", color="#FFFFFF", fontsize=11, pad=10)
    ax.tick_params(colors="#CBD5E1", labelsize=9)
    ax.set_ylim(-5, 105)
    ax.grid(color="#1E293B", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1E293B")
    ax.spines["bottom"].set_color("#1E293B")

    ax.legend(facecolor="#131E33", edgecolor="#1E293B", labelcolor="#F8FAFC", fontsize=8.5)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. TRANSITION TABLE & ACADEMIC NOTICE
    # -------------------------------------------------------------
    col_tbl, col_notice = st.columns([1.2, 0.8])

    with col_tbl:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Category Transition Table")
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.82rem; margin-bottom:12px;'>"
            "Predicted categories at discrete checkpoints along the parameter span:"
            "</p>",
            unsafe_allow_html=True
        )

        # Sample 5 evenly spaced checkpoints
        checkpoints = sim_results_df.iloc[::5].copy()
        display_cols = [sim_feature, "Predicted_Category", "Risk_Tier"]
        checkpoints[sim_feature] = checkpoints[sim_feature].round(2)
        st.dataframe(checkpoints[display_cols], use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_notice:
        st.markdown("""
        <div class="standard-card" style="border-left: 3px solid #F59E0B;">
            <h4 style="margin-top:0; color:#F59E0B; font-size:0.95rem;">Academic Simulation Notice</h4>
            <p style="color:#CBD5E1; font-size:0.83rem; line-height:1.6; margin-bottom:10px;">
                These curves represent <b>statistical model estimates</b> derived from historical cohort decision rules.
            </p>
            <p style="color:#94A3B8; font-size:0.8rem; line-height:1.5; margin:0;">
                They are intended for advising scenario analysis, identifying break-even study thresholds,
                and setting incremental student goals. They do not constitute guaranteed academic outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True)
