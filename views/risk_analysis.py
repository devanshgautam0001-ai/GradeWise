"""
Academic Risk Analysis View
Student Performance Prediction and Analysis System
Rule-based academic risk profiling, cohort vulnerability analysis,
and clear separation between Decision Tree classifications and institutional risk policies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml_engine
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_risk_analysis_view():
    """Render Academic Risk Analysis & Institutional Policy Stratification dashboard."""
    render_page_header(
        title="Academic Risk Analysis",
        subtitle="Institutional risk stratification, early-warning policy rules, and cohort vulnerability diagnostics.",
        badge="Risk Intelligence"
    )

    render_active_dataset_badge(st.session_state)

    # -------------------------------------------------------------
    # 1. INSTITUTIONAL RISK TIERS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🚦 Institutional Risk Stratification Standards")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.88rem; margin-bottom:14px;'>"
        "The system evaluates academic risk using a deterministic two-stage architecture: "
        "first, the <b>Decision Tree Classifier</b> predicts student performance category; "
        "second, <b>documented institutional policy rules</b> assign a risk tier based on performance standing and behavioral gates."
        "</p>",
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:16px;">
            <div style="color:#10B981; font-weight:800; font-size:1.05rem;">Low Academic Risk</div>
            <div style="font-size:1.25rem; font-weight:900; color:#FFFFFF; margin:6px 0;">Optimal Trajectory</div>
            <p style="color:#CBD5E1; font-size:0.82rem; line-height:1.6; margin:0;">
                Assigned when model predicts <b>Excellent</b> or <b>Good</b> with attendance &ge; 65%. Minimal probability of course failure.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3); border-radius:10px; padding:16px;">
            <div style="color:#F59E0B; font-weight:800; font-size:1.05rem;">Medium Academic Risk</div>
            <div style="font-size:1.25rem; font-weight:900; color:#FFFFFF; margin:6px 0;">Targeted Monitoring</div>
            <p style="color:#CBD5E1; font-size:0.82rem; line-height:1.6; margin:0;">
                Assigned when model predicts <b>Average</b>, or <b>Good</b> with marginal attendance (&lt;65%). Student is vulnerable to grade drops.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:16px;">
            <div style="color:#EF4444; font-weight:800; font-size:1.05rem;">High Academic Risk</div>
            <div style="font-size:1.25rem; font-weight:900; color:#FFFFFF; margin:6px 0;">Urgent Intervention</div>
            <p style="color:#CBD5E1; font-size:0.82rem; line-height:1.6; margin:0;">
                Assigned when model predicts <b>Needs Improvement</b>, or <b>Average</b> with severe attendance lag (&lt;70%) or study deficit (&lt;3 hrs/day).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. SEPARATION OF ML MODEL VS. RISK RULES
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Architectural Principle: Machine Learning vs. Institutional Policy")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.85rem; line-height:1.6;'>"
        "To ensure accountability, transparency, and ethical compliance in educational institutions, "
        "<b>risk tiers are not generated as random or opaque numbers</b>. The architecture strictly distinguishes between:"
        "</p>",
        unsafe_allow_html=True
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("""
        <div style="background:#0E1626; border-left:3px solid #3B82F6; padding:14px; border-radius:0 8px 8px 0;">
            <div style="color:#93C5FD; font-weight:700; font-size:0.9rem;">1. Machine Learning Model (Decision Tree)</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:6px; line-height:1.5;">
                Learns statistical decision boundaries from historical student cohorts.
                Outputs a performance category (e.g. <i>Excellent, Good, Average, Needs Improvement</i>)
                and class probabilities based on Gini/Entropy minimization.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown("""
        <div style="background:#0E1626; border-left:3px solid #10B981; padding:14px; border-radius:0 8px 8px 0;">
            <div style="color:#6EE7B7; font-weight:700; font-size:0.9rem;">2. Institutional Risk Decision Matrix</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:6px; line-height:1.5;">
                Applies auditable institutional guidelines onto the model output.
                If a student has a predicted average grade but has low attendance (&lt;70%),
                institutional policy elevates the status to <b>High Risk</b> to guarantee faculty notification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. LATEST STUDENT DIAGNOSTIC (IF EVALUATED)
    # -------------------------------------------------------------
    if st.session_state.get("latest_pred_category") is not None:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 👤 Current Student Risk Assessment")

        pred_cat = st.session_state.latest_pred_category
        risk_tier = st.session_state.latest_risk_tier
        risk_color = st.session_state.latest_risk_color
        risk_rule = st.session_state.latest_risk_rule

        c1, c2 = st.columns([1, 2])
        with c1:
            render_metric_card(
                title="Assigned Risk Status",
                value=risk_tier,
                subtext=f"Model Category: {pred_cat}",
                top_accent=True
            )
        with c2:
            st.markdown(f"""
            <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.1); border-radius:10px; padding:18px;">
                <div style="color:{risk_color}; font-weight:700; font-size:0.92rem; margin-bottom:6px;">Deterministic Policy Trigger:</div>
                <div style="color:#F8FAFC; font-size:0.9rem; line-height:1.6;">
                    {risk_rule}
                </div>
                <div style="color:#94A3B8; font-size:0.8rem; margin-top:8px;">
                    This classification was derived deterministically from the predicted performance category and academic threshold rules.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. KEY VULNERABILITY DRIVERS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Key Risk Vulnerability Drivers")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.1); border-radius:8px; padding:14px;">
            <div style="color:#EF4444; font-weight:700; font-size:0.9rem;">1. Severe Study Hour Deficit</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:6px; line-height:1.6;">
                Students logging fewer than 3.0 daily study hours exhibit an acute likelihood of falling into the <i>Needs Improvement</i> tier.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.1); border-radius:8px; padding:14px;">
            <div style="color:#F59E0B; font-weight:700; font-size:0.9rem;">2. Chronic Attendance Lag</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:6px; line-height:1.6;">
                Classroom attendance under 70% fractures continuous learning, increasing failure probability regardless of exam cramming.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.1); border-radius:8px; padding:14px;">
            <div style="color:#8B5CF6; font-weight:700; font-size:0.9rem;">3. Sleep & Stress Friction</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:6px; line-height:1.6;">
                Nightly sleep under 6 hours paired with high stress (scale > 7) compromises cognitive focus and exam retention.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
