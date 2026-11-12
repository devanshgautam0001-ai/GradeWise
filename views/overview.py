"""
Overview & Academic Analytics View
Student Performance Prediction and Analysis System
Executive dashboard with dual-scope analytics:
Cohort-level dataset analytics vs. Individual student prediction history analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from styles import render_page_header, render_active_dataset_badge, render_metric_card
import auth
import ml_engine

CATEGORY_COLORS = {
    "Excellent": "#10B981",
    "Good": "#3B82F6",
    "Average": "#F59E0B",
    "Needs Improvement": "#EF4444",
}


def render_overview_view():
    """Render Advanced Academic Analytics & Executive Overview dashboard."""
    render_page_header(
        title="Executive Overview & Academic Analytics",
        subtitle="Institutional analytics hub combining cohort-level dataset metrics with private student evaluation trajectories.",
        badge="Analytics Hub"
    )

    render_active_dataset_badge(st.session_state)

    active_df = getattr(st.session_state, "active_df", None)
    username = st.session_state.username
    user_stats = auth.get_user_stats(username)
    trained_model = st.session_state.get("trained_dt_model")
    if trained_model is None:
        trained_model = ml_engine.load_saved_decision_tree()

    # -------------------------------------------------------------
    # 1. TOP EXECUTIVE METRIC CARDS
    # -------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        total_cohort = f"{len(active_df):,}" if active_df is not None else "0"
        col_count = len(active_df.columns) if active_df is not None else 0
        render_metric_card(
            title="Cohort Dataset Size",
            value=total_cohort,
            subtext=f"{col_count} Feature Columns",
            top_accent=True
        )

    with c2:
        model_acc = f"{trained_model['accuracy']*100:.1f}%" if trained_model else "Untrained"
        sub_info = f"Tree Depth: {trained_model['tree_depth']}" if trained_model else "Go to Model Evaluation"
        render_metric_card(
            title="Decision Tree Accuracy",
            value=model_acc,
            subtext=sub_info,
            top_accent=True
        )

    with c3:
        user_preds = user_stats.get("total_predictions", 0)
        render_metric_card(
            title="Total Student Evaluations",
            value=str(user_preds),
            subtext=f"User: {username}",
            top_accent=True
        )

    with c4:
        latest_standing = user_stats.get("latest_category", "None")
        latest_risk = user_stats.get("latest_risk", "None")
        render_metric_card(
            title="Latest Assessment Standing",
            value=str(latest_standing),
            subtext=f"Risk: {latest_risk}",
            top_accent=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. DUAL-SCOPE ANALYTICS: COHORT DATASET VS. PERSONAL HISTORY
    # -------------------------------------------------------------
    tab_cohort, tab_personal = st.tabs([
        "🏛️ Cohort-Level Dataset Analytics (Active Dataset)",
        "👤 Personal Student Evaluation Analytics (User History)"
    ])

    # TAB A: COHORT DATASET ANALYTICS
    with tab_cohort:
        if active_df is not None:
            st.markdown(
                f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
                f"Statistical distributions computed across all records in active dataset <b>{st.session_state.active_dataset_name}</b> (N = {len(active_df):,}):"
                f"</p>",
                unsafe_allow_html=True
            )

            col_dist, col_imp = st.columns([1.1, 0.9])

            with col_dist:
                st.markdown('<div class="standard-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Cohort Performance / Score Distribution")

                if "final_score" in active_df.columns:
                    # Categorize and plot
                    cat_series = ml_engine.categorize_numeric_scores(active_df["final_score"])
                    cat_counts = cat_series.value_counts()

                    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#0E1626")
                    ax.set_facecolor("#0E1626")

                    bar_colors = [CATEGORY_COLORS.get(c, "#3B82F6") for c in cat_counts.index]
                    bars = ax.bar(cat_counts.index, cat_counts.values, color=bar_colors, edgecolor="none", width=0.55)

                    ax.set_ylabel("Student Count", color="#94A3B8", fontsize=9)
                    ax.tick_params(colors="#CBD5E1", labelsize=8.5)
                    ax.grid(axis="y", color="#1E293B", linestyle="--", alpha=0.6)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color("#1E293B")
                    ax.spines["bottom"].set_color("#1E293B")

                    for b in bars:
                        h = b.get_height()
                        ax.text(b.get_x() + b.get_width()/2, h + 8, f"{int(h)} ({h/len(active_df)*100:.1f}%)", ha="center", color="#FFFFFF", fontsize=8, fontweight="bold")

                    st.pyplot(fig)
                    plt.close(fig)

                elif trained_model and trained_model["target_column"] in active_df.columns:
                    t_col = trained_model["target_column"]
                    t_counts = active_df[t_col].value_counts()

                    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#0E1626")
                    ax.set_facecolor("#0E1626")
                    ax.bar(t_counts.index.astype(str), t_counts.values, color="#3B82F6", width=0.55)
                    ax.set_ylabel("Count", color="#94A3B8", fontsize=9)
                    ax.tick_params(colors="#CBD5E1", labelsize=8.5)
                    ax.grid(axis="y", color="#1E293B", linestyle="--", alpha=0.6)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("ℹ Final score or target column not present in current active dataset.")

                st.markdown('</div>', unsafe_allow_html=True)

            with col_imp:
                st.markdown('<div class="standard-card">', unsafe_allow_html=True)
                st.markdown("### 🏆 Feature Importance Summary")
                if trained_model and not trained_model["feature_importance_df"].empty:
                    top_imp = trained_model["feature_importance_df"].head(5)

                    fig_imp, ax_imp = plt.subplots(figsize=(5.5, 3.4), facecolor="#0E1626")
                    ax_imp.set_facecolor("#0E1626")

                    y_pos = np.arange(len(top_imp))
                    ax_imp.barh(y_pos, top_imp["Percentage"], color="#8B5CF6", height=0.55)
                    ax_imp.set_yticks(y_pos)
                    ax_imp.set_yticklabels(top_imp["Feature"], color="#CBD5E1", fontsize=8.5)
                    ax_imp.invert_yaxis()
                    ax_imp.set_xlabel("Contribution (%)", color="#94A3B8", fontsize=8.5)
                    ax_imp.tick_params(colors="#CBD5E1")
                    ax_imp.grid(axis="x", color="#1E293B", linestyle="--", alpha=0.6)
                    ax_imp.spines["top"].set_visible(False)
                    ax_imp.spines["right"].set_visible(False)
                    ax_imp.spines["left"].set_color("#1E293B")
                    ax_imp.spines["bottom"].set_color("#1E293B")

                    for idx, p in enumerate(top_imp["Percentage"]):
                        ax_imp.text(p + 0.8, idx, f"{p:.1f}%", va="center", color="#FFFFFF", fontsize=8, fontweight="bold")

                    st.pyplot(fig_imp)
                    plt.close(fig_imp)
                else:
                    st.info("Train a Decision Tree model in Model Evaluation to unlock feature importance rankings.")

                st.markdown('</div>', unsafe_allow_html=True)

            # Study Hours & Attendance Distributions
            if "study_hours" in active_df.columns and "attendance" in active_df.columns:
                st.markdown('<div class="standard-card">', unsafe_allow_html=True)
                st.markdown("### ⏱️ Cohort Study Hours & Attendance Distributions")

                col_sh, col_att = st.columns(2)
                with col_sh:
                    fig_s, ax_s = plt.subplots(figsize=(5.5, 2.8), facecolor="#0E1626")
                    ax_s.set_facecolor("#0E1626")
                    ax_s.hist(active_df["study_hours"].dropna(), bins=15, color="#3B82F6", edgecolor="#0E1626", alpha=0.85)
                    ax_s.set_title(f"Study Hours (Mean: {active_df['study_hours'].mean():.1f} hrs/day)", color="#FFFFFF", fontsize=9.5)
                    ax_s.tick_params(colors="#CBD5E1", labelsize=8)
                    ax_s.grid(color="#1E293B", linestyle="--", alpha=0.6)
                    ax_s.spines["top"].set_visible(False)
                    ax_s.spines["right"].set_visible(False)
                    ax_s.spines["left"].set_color("#1E293B")
                    ax_s.spines["bottom"].set_color("#1E293B")
                    st.pyplot(fig_s)
                    plt.close(fig_s)

                with col_att:
                    fig_a, ax_a = plt.subplots(figsize=(5.5, 2.8), facecolor="#0E1626")
                    ax_a.set_facecolor("#0E1626")
                    ax_a.hist(active_df["attendance"].dropna(), bins=15, color="#10B981", edgecolor="#0E1626", alpha=0.85)
                    ax_a.set_title(f"Attendance (Mean: {active_df['attendance'].mean():.1f}%)", color="#FFFFFF", fontsize=9.5)
                    ax_a.tick_params(colors="#CBD5E1", labelsize=8)
                    ax_a.grid(color="#1E293B", linestyle="--", alpha=0.6)
                    ax_a.spines["top"].set_visible(False)
                    ax_a.spines["right"].set_visible(False)
                    ax_a.spines["left"].set_color("#1E293B")
                    ax_a.spines["bottom"].set_color("#1E293B")
                    st.pyplot(fig_a)
                    plt.close(fig_a)

                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("No active dataset loaded. Please navigate to Dataset Management.")

    # TAB B: PERSONAL STUDENT EVALUATION ANALYTICS
    with tab_personal:
        st.markdown(
            f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
            f"Personal audit metrics compiled exclusively for <b>{username}</b> from local SQLite:"
            f"</p>",
            unsafe_allow_html=True
        )

        user_records = auth.get_user_history(username, limit=100)
        if user_records:
            cat_list = [r["performance_category"] for r in user_records if r["performance_category"]]
            risk_list = [r["risk"] for r in user_records if r["risk"]]

            u_c1, u_c2 = st.columns(2)

            with u_c1:
                st.markdown('<div class="standard-card">', unsafe_allow_html=True)
                st.markdown("### 👤 Personal Performance Breakdown")
                if cat_list:
                    s_cats = pd.Series(cat_list).value_counts()
                    fig_u, ax_u = plt.subplots(figsize=(5, 3), facecolor="#0E1626")
                    ax_u.set_facecolor("#0E1626")
                    colors_u = [CATEGORY_COLORS.get(c, "#3B82F6") for c in s_cats.index]
                    ax_u.pie(s_cats.values, labels=s_cats.index, autopct="%1.0f%%", colors=colors_u, textprops={"color": "white", "fontsize": 8.5})
                    st.pyplot(fig_u)
                    plt.close(fig_u)
                else:
                    st.write("No categories logged yet.")
                st.markdown('</div>', unsafe_allow_html=True)

            with u_c2:
                st.markdown('<div class="standard-card">', unsafe_allow_html=True)
                st.markdown("### 🚦 Personal Academic Risk Breakdown")
                if risk_list:
                    s_risks = pd.Series(risk_list).value_counts()
                    fig_r, ax_r = plt.subplots(figsize=(5, 3), facecolor="#0E1626")
                    ax_r.set_facecolor("#0E1626")
                    ax_r.bar(s_risks.index.astype(str), s_risks.values, color="#F59E0B", width=0.5)
                    ax_r.set_ylabel("Assessments", color="#94A3B8", fontsize=8.5)
                    ax_r.tick_params(colors="#CBD5E1", labelsize=8)
                    ax_r.grid(axis="y", color="#1E293B", linestyle="--", alpha=0.6)
                    st.pyplot(fig_r)
                    plt.close(fig_r)
                else:
                    st.write("No risk tiers logged yet.")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("ℹ No personal prediction records logged yet. Make an assessment in Performance Prediction to build personal analytics.")
