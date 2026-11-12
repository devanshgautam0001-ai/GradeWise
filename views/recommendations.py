 """
Personalized Recommendations View
Student Performance Prediction and Analysis System
Rule-based academic advisory engine based on evaluated student attributes,
risk tiers, and active cohort benchmarks.
"""

import streamlit as st
import pandas as pd
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_recommendations_view():
    """Render Actionable Academic Recommendations interface."""
    render_page_header(
        title="Personalized Academic Recommendations",
        subtitle="Rule-based academic intervention guidance tailored to student performance categories, behavioral indicators, and cohort benchmarks.",
        badge="Academic Advisory"
    )

    render_active_dataset_badge(st.session_state)

    # Check for prediction
    pred_category = st.session_state.get("latest_pred_category")
    inputs = st.session_state.get("latest_pred_inputs")
    risk_tier = st.session_state.get("latest_risk_tier")
    risk_color = st.session_state.get("latest_risk_color", "#3B82F6")

    # Fallback to alternate session keys if needed
    if not pred_category and st.session_state.get("last_dt_prediction"):
        pred_category = st.session_state.last_dt_prediction.get("predicted_class")
        risk_tier = st.session_state.last_dt_prediction.get("risk_tier")
        risk_color = st.session_state.last_dt_prediction.get("risk_color", "#3B82F6")
        inputs = st.session_state.get("last_dt_inputs")

    if not pred_category or not inputs:
        st.markdown("""
        <div class="standard-card" style="border-color: rgba(59, 130, 246, 0.4);">
            <h3 style="color:#60A5FA; margin-top:0;">ℹ Prediction Required for Targeted Advisory</h3>
            <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
                The academic recommendation engine personalizes strategies based on a student's specific
                behavioral and academic inputs. Please navigate to <b>Performance Prediction</b>, enter the student
                attributes, and calculate a prediction first.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # -------------------------------------------------------------
    # 1. EVALUATION PROFILE SUMMARY
    # -------------------------------------------------------------
    st.markdown(f"""
    <div class="standard-card" style="border-left: 4px solid {risk_color};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8; font-weight:700;">Student Evaluation Profile</span>
                <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF; margin-top:4px;">
                    Predicted Standing: <span style="color:{risk_color};">{pred_category}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <span style="background:{risk_color}22; color:{risk_color}; border:1px solid {risk_color}55; padding:6px 14px; border-radius:6px; font-weight:700; font-size:0.85rem;">
                    ● {risk_tier}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. RULE-BASED INTERVENTION SYNTHESIS
    # -------------------------------------------------------------
    study_hours = float(inputs.get("study_hours", 5.0))
    attendance = float(inputs.get("attendance", 75.0))
    sleep_hours = float(inputs.get("sleep_hours", 7.0))
    previous_score = float(inputs.get("previous_score", 65.0))
    social_media = float(inputs.get("social_media_usage", 2.0))
    stress_level = int(inputs.get("stress_level", 5))
    prep_level = int(inputs.get("preparation_level", 3))

    critical_actions = []
    habit_actions = []
    wellness_actions = []

    # Attendance Rules
    if attendance < 75.0:
        gap = round(75.0 - attendance, 1)
        critical_actions.append({
            "title": f"Restore Classroom Attendance (Deficit: {gap}%)",
            "desc": f"Current attendance is {attendance:.1f}%. Institutional statistics identify attendance below 75% as the primary risk factor for course failure. Attend all upcoming lectures and consult faculty regarding attendance recovery assignments."
        })
    elif attendance < 85.0:
        habit_actions.append({
            "title": "Consolidate Attendance to 85%+",
            "desc": f"Attendance at {attendance:.1f}% is passing but has room for improvement. Consistently attending discussions reinforces complex problem-solving patterns."
        })

    # Study Hours Rules
    if study_hours < 3.5:
        critical_actions.append({
            "title": f"Increase Daily Study Allocation (Currently {study_hours:.1f} hrs/day)",
            "desc": "A study allocation under 3.5 hours leaves insufficient time for conceptual synthesis. Allocate an additional 1.5 to 2.0 hours daily of focused, non-lecture study time."
        })
    elif study_hours < 5.5:
        habit_actions.append({
            "title": "Adopt Structured Daily Study Routine",
            "desc": f"Currently logging {study_hours:.1f} hrs/day. Implement a fixed daily calendar slot (e.g. 4 PM - 7 PM) using the Pomodoro technique to build study momentum."
        })

    # Stress & Sleep Wellness Rules
    if stress_level >= 7:
        wellness_actions.append({
            "title": f"Mitigate Acute Academic Stress ({stress_level}/10)",
            "desc": "Elevated psychological stress inhibits memory retrieval during examinations. Break large projects into weekly bite-sized goals and seek faculty office hours early."
        })

    if sleep_hours < 6.5:
        wellness_actions.append({
            "title": f"Expand Nightly Sleep Window ({sleep_hours:.1f} hrs/day)",
            "desc": "Sleep duration below 6.5 hours disrupts rapid-eye-movement (REM) memory consolidation. Prioritize a regular 7.0 - 8.0 hour sleep window, especially before exams."
        })

    # Distraction Rules
    if social_media > 3.5:
        habit_actions.append({
            "title": f"Constrain Entertainment Screen Time ({social_media:.1f} hrs/day)",
            "desc": "Daily social media usage exceeding 3.5 hours causes cognitive fragmentation. Implement digital app limits during active revision blocks."
        })

    # Preparation & Previous Score
    if prep_level <= 2:
        critical_actions.append({
            "title": "Transition to Active Exam Recall Practice",
            "desc": "Preparation level is low. Move away from passive textbook reading toward active question solving, timed mock exams, and self-testing."
        })

    if previous_score < 55.0:
        critical_actions.append({
            "title": f"Remediate Core Subject Prerequisite Foundations (Prior: {previous_score:.1f})",
            "desc": "A low previous exam score signals underlying conceptual gaps. Review prerequisite foundational modules and arrange peer tutoring sessions."
        })

    # High Performer / Maintenance Rule
    if not critical_actions and not habit_actions and not wellness_actions:
        habit_actions.append({
            "title": "Exemplary Academic Trajectory Maintained",
            "desc": "Current behavioral, attendance, and study parameters reflect an optimal learning routine. Maintain consistency and consider academic mentoring or research projects."
        })

    # -------------------------------------------------------------
    # 3. DISPLAY CATEGORIZED ACTION CARDS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🚨 High-Priority Critical Interventions")
    if critical_actions:
        for a in critical_actions:
            st.markdown(f"""
            <div style="background:#0E1626; border-left:3px solid #EF4444; border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;">
                <div style="color:#EF4444; font-weight:700; font-size:0.92rem;">{a['title']}</div>
                <div style="color:#CBD5E1; font-size:0.84rem; margin-top:4px; line-height:1.5;">{a['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#10B981; font-size:0.85rem;'>✓ No critical academic risk gates triggered.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 💡 Habit & Study Optimization Recommendations")
    if habit_actions:
        for a in habit_actions:
            st.markdown(f"""
            <div style="background:#0E1626; border-left:3px solid #3B82F6; border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;">
                <div style="color:#93C5FD; font-weight:700; font-size:0.92rem;">{a['title']}</div>
                <div style="color:#CBD5E1; font-size:0.84rem; margin-top:4px; line-height:1.5;">{a['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if wellness_actions:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 🌿 Cognitive Wellness & Rest Balance")
        for a in wellness_actions:
            st.markdown(f"""
            <div style="background:#0E1626; border-left:3px solid #8B5CF6; border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;">
                <div style="color:#C4B5FD; font-weight:700; font-size:0.92rem;">{a['title']}</div>
                <div style="color:#CBD5E1; font-size:0.84rem; margin-top:4px; line-height:1.5;">{a['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. ACTION PLAN BLUEPRINT
    # -------------------------------------------------------------
    st.markdown("""
    <div class="standard-card">
        <h4 style="margin-top:0; font-size:1rem;">📅 Suggested Weekly Study Execution Blueprint</h4>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; margin-top:10px;">
            <div style="background:#0E1626; padding:12px; border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
                <b style="color:#93C5FD; font-size:0.85rem;">Phase 1: Week 1-2</b>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:4px; margin-bottom:0;">Attend 100% of lectures. Audit incomplete assignments and organize course notes.</p>
            </div>
            <div style="background:#0E1626; padding:12px; border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
                <b style="color:#C4B5FD; font-size:0.85rem;">Phase 2: Week 3-4</b>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:4px; margin-bottom:0;">Establish fixed daily study routine of 5+ hours. Solve past exam papers under timed conditions.</p>
            </div>
            <div style="background:#0E1626; padding:12px; border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
                <b style="color:#6EE7B7; font-size:0.85rem;">Phase 3: Pre-Exam</b>
                <p style="color:#94A3B8; font-size:0.8rem; margin-top:4px; margin-bottom:0;">Target 7.5 hours of sleep. Conduct active recall review and avoid last-minute cramming.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
