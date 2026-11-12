"""
Prediction History View
Student Performance Prediction and Analysis System
User-specific SQLite prediction history, search filtering, side-by-side assessment comparison,
individual record deletion, performance trend charts, and CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from styles import render_page_header, render_active_dataset_badge, render_metric_card
import auth


def render_history_view():
    """Render User Prediction History interface with trend charts, search, and comparison."""
    render_page_header(
        title="Personal Prediction History & Audit Trail",
        subtitle="Private audit trail of past student evaluations, multi-assessment comparisons, trend trajectory charts, and record management.",
        badge="Audit Log"
    )

    render_active_dataset_badge(st.session_state)

    username = st.session_state.username
    stats = auth.get_user_stats(username)

    # -------------------------------------------------------------
    # 1. SUMMARY METRIC CARDS
    # -------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Assessments", str(stats["total_predictions"]), f"User: {username}", True)
    with m2:
        render_metric_card("Average Study Allocation", f"{stats['avg_study_hours']} hrs/day", "Historical mean", True)
    with m3:
        render_metric_card("Average Attendance", f"{stats['avg_attendance']}%", "Historical mean", True)
    with m4:
        render_metric_card("Latest Assessment Standing", str(stats["latest_category"]), f"Risk: {stats['latest_risk']}", True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. SEARCH & HISTORY TABLE
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Assessment History Search & Filter")

    search_col, count_col = st.columns([2.5, 1])
    with search_col:
        search_query = st.text_input(
            "Search by category, risk tier, or date:",
            placeholder="e.g. Excellent, High Risk, 2026-11...",
            key="history_search_input"
        )

    records = auth.get_user_history(username, search_query=search_query, limit=100)

    with count_col:
        st.markdown(
            f"<div style='font-size:0.85rem; color:#94A3B8; padding-top:28px;'>"
            f"Showing <b style='color:#FFFFFF;'>{len(records)}</b> matching records"
            f"</div>",
            unsafe_allow_html=True
        )

    if records:
        rows_display = []
        for r in records:
            rows_display.append({
                "Record ID": r["id"],
                "Timestamp": r["created_at"],
                "Category": r["performance_category"],
                "Risk Tier": r["risk"],
                "Study Hours": f"{r['study_hours']:.1f}",
                "Attendance (%)": f"{r['attendance']:.1f}%",
            })

        df_display = pd.DataFrame(rows_display)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Action Buttons
        col_dl, col_clr = st.columns([1.5, 1.5])
        with col_dl:
            csv_data = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Assessment History (CSV)",
                data=csv_data,
                file_name=f"{username}_academic_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_clr:
            if st.button("🗑️ Clear Entire History", use_container_width=True):
                ok = auth.clear_user_history(username)
                if ok:
                    st.success("All personal assessment records cleared.")
                    st.rerun()
                else:
                    st.error("Failed to clear records.")

    else:
        st.info("ℹ No assessment records found. Complete a student evaluation in **Performance Prediction** to record history.")

    st.markdown('</div>', unsafe_allow_html=True)

    if not records:
        return

    # -------------------------------------------------------------
    # 3. PERFORMANCE TREND TRAJECTORY CHART
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Historical Performance Trend Trajectory")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:12px;'>"
        "Chronological trend of study hours and attendance across your evaluation sessions:"
        "</p>",
        unsafe_allow_html=True
    )

    trend_df = auth.get_user_trend_data(username)
    if not trend_df.empty and len(trend_df) >= 2:
        fig, ax1 = plt.subplots(figsize=(9, 3.8), facecolor="#0E1626")
        ax1.set_facecolor("#0E1626")

        x_indices = np.arange(len(trend_df))
        ax1.plot(x_indices, trend_df["Study Hours"], color="#3B82F6", marker="o", linewidth=2.5, label="Daily Study Hours")
        ax1.set_ylabel("Study Hours (hrs/day)", color="#3B82F6", fontsize=9.5)
        ax1.tick_params(axis="y", labelcolor="#3B82F6", labelsize=8.5)
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(trend_df["Assessment ID"], color="#CBD5E1", fontsize=8, rotation=30)
        ax1.grid(color="#1E293B", linestyle="--", alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(x_indices, trend_df["Attendance"], color="#10B981", marker="s", linewidth=2.5, linestyle="--", label="Attendance (%)")
        ax2.set_ylabel("Attendance Rate (%)", color="#10B981", fontsize=9.5)
        ax2.tick_params(axis="y", labelcolor="#10B981", labelsize=8.5)

        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax1.spines["left"].set_color("#1E293B")
        ax1.spines["bottom"].set_color("#1E293B")
        ax2.spines["right"].set_color("#1E293B")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("ℹ At least 2 assessment evaluations are required to render trend trajectory charts.")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. SIDE-BY-SIDE ASSESSMENT COMPARISON & INDIVIDUAL DELETION
    # -------------------------------------------------------------
    col_comp, col_del = st.columns([1.3, 0.7])

    with col_comp:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Side-by-Side Assessment Comparison")
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.82rem; margin-bottom:10px;'>"
            "Select any two historical assessments to inspect changes in student indicators and outcomes:"
            "</p>",
            unsafe_allow_html=True
        )

        record_ids = [r["id"] for r in records]
        if len(record_ids) >= 2:
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                id_a = st.selectbox("Baseline Assessment", options=record_ids, index=0, key="comp_id_a")
            with c_sel2:
                id_b = st.selectbox("Comparison Assessment", options=record_ids, index=min(1, len(record_ids)-1), key="comp_id_b")

            rec_a = next((r for r in records if r["id"] == id_a), None)
            rec_b = next((r for r in records if r["id"] == id_b), None)

            if rec_a and rec_b:
                comp_rows = [
                    {"Metric": "Assessment Date", "Baseline (A)": rec_a["created_at"], "Comparison (B)": rec_b["created_at"]},
                    {"Metric": "Performance Category", "Baseline (A)": rec_a["performance_category"], "Comparison (B)": rec_b["performance_category"]},
                    {"Metric": "Academic Risk", "Baseline (A)": rec_a["risk"], "Comparison (B)": rec_b["risk"]},
                    {"Metric": "Daily Study Hours", "Baseline (A)": f"{rec_a['study_hours']:.1f}", "Comparison (B)": f"{rec_b['study_hours']:.1f}"},
                    {"Metric": "Attendance", "Baseline (A)": f"{rec_a['attendance']:.1f}%", "Comparison (B)": f"{rec_b['attendance']:.1f}%"},
                ]
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Log at least 2 assessments to enable side-by-side comparison.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_del:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 🗑️ Delete Specific Record")
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.82rem; margin-bottom:12px;'>"
            "Select an assessment ID to remove it permanently:"
            "</p>",
            unsafe_allow_html=True
        )

        del_id = st.selectbox("Select Record ID to Delete", options=record_ids, key="del_record_id")
        if st.button("Delete Selected Record", use_container_width=True):
            deleted = auth.delete_user_prediction(del_id, username)
            if deleted:
                st.success(f"Record #{del_id} deleted successfully.")
                st.rerun()
            else:
                st.error("Failed to delete record.")

        st.markdown('</div>', unsafe_allow_html=True)
