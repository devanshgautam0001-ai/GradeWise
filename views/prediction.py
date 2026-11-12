"""
Performance Prediction View
Student Performance Prediction and Analysis System
Dynamic prediction form, multiclass probabilities, rule-based academic risk,
Student vs. Class comparative analytics, and downloadable PDF performance report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import ml_engine
import auth
from styles import render_page_header, render_active_dataset_badge, render_metric_card
import pdf_report_generator

CATEGORY_COLORS = {
    "Excellent": "#10B981",
    "Good": "#3B82F6",
    "Average": "#F59E0B",
    "Needs Improvement": "#EF4444",
}


def render_prediction_view():
    """Render dynamic student performance prediction interface with PDF export and cohort benchmarking."""
    render_page_header(
        title="Student Performance Prediction",
        subtitle="Dynamic inference form generated from the active Decision Tree model. Evaluates performance categories, rule-based academic risk, cohort benchmarks, and official PDF reporting.",
        badge="Inference Engine"
    )

    render_active_dataset_badge(st.session_state)

    trained_model = st.session_state.get("trained_dt_model")
    if trained_model is None:
        saved = ml_engine.load_saved_decision_tree()
        if saved is not None:
            st.session_state.trained_dt_model = saved
            trained_model = saved

    if trained_model is None:
        st.warning("⚠️ No trained Decision Tree model available. Please train a model in **Model Evaluation** to enable predictions.")
        return

    active_df = getattr(st.session_state, "active_df", None)
    pipeline = trained_model["pipeline"]
    feature_columns = trained_model["feature_columns"]
    classes = trained_model["classes"]

    # -------------------------------------------------------------
    # 1. DYNAMIC INPUT FORM
    # -------------------------------------------------------------
    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 📥 Student Attribute Inputs")
        st.markdown(
            f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
            f"Enter values for the <b>{len(feature_columns)} features</b> configured during model training."
            f"</p>",
            unsafe_allow_html=True
        )

        input_values = {}
        sub1, sub2 = st.columns(2)

        for idx, feat in enumerate(feature_columns):
            curr_col = sub1 if (idx % 2 == 0) else sub2

            with curr_col:
                is_num = (
                    active_df is not None
                    and feat in active_df.columns
                    and pd.api.types.is_numeric_dtype(active_df[feat])
                )

                if is_num and active_df is not None:
                    series = active_df[feat].dropna()
                    min_val = float(series.min()) if not series.empty else 0.0
                    max_val = float(series.max()) if not series.empty else 100.0
                    med_val = float(series.median()) if not series.empty else 50.0

                    if min_val == max_val:
                        max_val = min_val + 10.0

                    label = feat.replace("_", " ").title()
                    step = 0.5 if (max_val - min_val) < 20 else 1.0

                    saved_val = st.session_state.get(f"pred_input_{feat}", med_val)
                    val = st.slider(
                        label=label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(saved_val),
                        step=step,
                        key=f"input_slider_{feat}"
                    )
                    input_values[feat] = val
                    st.session_state[f"pred_input_{feat}"] = val

                else:
                    label = feat.replace("_", " ").title()
                    opts = []
                    if active_df is not None and feat in active_df.columns:
                        opts = list(active_df[feat].dropna().unique())

                    if not opts:
                        opts = ["Standard", "Other"]

                    saved_cat = st.session_state.get(f"pred_input_{feat}", opts[0])
                    idx_opt = opts.index(saved_cat) if saved_cat in opts else 0
                    cat_val = st.selectbox(
                        label=label,
                        options=opts,
                        index=idx_opt,
                        key=f"input_select_{feat}"
                    )
                    input_values[feat] = cat_val
                    st.session_state[f"pred_input_{feat}"] = cat_val

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Student Performance Category", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. PREDICTION RESULTS & RISK ANALYSIS
    # -------------------------------------------------------------
    with right_col:
        has_prediction = predict_btn or st.session_state.get("latest_pred_category") is not None

        if has_prediction:
            if predict_btn:
                pred_category, prob_dict = ml_engine.predict_single_student(
                    pipeline=pipeline,
                    feature_columns=feature_columns,
                    input_values=input_values
                )
                risk_tier, risk_badge, risk_color, risk_rule = ml_engine.evaluate_academic_risk_tier(
                    predicted_category=pred_category,
                    input_values=input_values
                )

                st.session_state.latest_pred_category = pred_category
                st.session_state.latest_pred_probs = prob_dict
                st.session_state.latest_risk_tier = risk_tier
                st.session_state.latest_risk_badge = risk_badge
                st.session_state.latest_risk_color = risk_color
                st.session_state.latest_risk_rule = risk_rule
                st.session_state.latest_pred_inputs = input_values

                # Synchronize for recommendations view
                conf_val = max(prob_dict.values()) / 100.0 if prob_dict else 1.0
                st.session_state.last_dt_prediction = {
                    "predicted_class": pred_category,
                    "risk_tier": risk_tier,
                    "risk_color": risk_color,
                    "confidence": conf_val,
                    "probabilities": prob_dict
                }
                st.session_state.last_dt_inputs = input_values

                # Save record to user SQLite database
                study_val = float(input_values.get("study_hours", 0.0))
                att_val = float(input_values.get("attendance", 0.0))
                auth.save_prediction(
                    username=st.session_state.username,
                    study_hours=study_val,
                    attendance=att_val,
                    score=0.0,
                    risk=risk_tier,
                    performance_category=pred_category,
                    inputs_dict=input_values,
                    model_config=trained_model.get("hyperparameters", {})
                )

            pred_cat = st.session_state.latest_pred_category
            prob_dict = st.session_state.latest_pred_probs
            risk_tier = st.session_state.latest_risk_tier
            risk_color = st.session_state.latest_risk_color
            risk_rule = st.session_state.latest_risk_rule
            inputs_current = st.session_state.latest_pred_inputs

            cat_color = CATEGORY_COLORS.get(pred_cat, "#3B82F6")

            # Result Card
            st.markdown(f"""
            <div class="standard-card" style="border-color: {cat_color}66; background: linear-gradient(135deg, {cat_color}14 0%, rgba(15,23,42,0.8) 100%);">
                <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8;">Predicted Performance Category</div>
                <div style="font-size:2.4rem; font-weight:900; color:{cat_color}; margin:6px 0; line-height:1.1;">
                    {pred_cat}
                </div>
                <div style="margin-top:6px; margin-bottom:12px;">
                    <span style="background:{risk_color}22; color:{risk_color}; border:1px solid {risk_color}55; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.82rem;">
                        ● {risk_tier}
                    </span>
                </div>
                <p style="color:#CBD5E1; font-size:0.82rem; line-height:1.5; margin:0;">
                    {risk_rule}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Multiclass Probability Distribution
            st.markdown('<div class="standard-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Class Probability Distribution")
            for cls_name in classes:
                prob = prob_dict.get(cls_name, 0.0)
                cls_color = CATEGORY_COLORS.get(cls_name, "#93C5FD")
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;'>"
                    f"<span style='color:#FFFFFF; font-weight:600;'>{cls_name}</span>"
                    f"<span style='color:{cls_color}; font-weight:700;'>{prob:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.progress(int(max(0, min(100, prob))))
            st.markdown('</div>', unsafe_allow_html=True)

            # PDF Report Download Action
            st.markdown('<div class="standard-card">', unsafe_allow_html=True)
            st.markdown("### 📄 Official PDF Performance Report")
            st.markdown(
                "<p style='color:#94A3B8; font-size:0.82rem; margin-bottom:12px;'>"
                "Download a publication-grade academic assessment diagnostic report in PDF format."
                "</p>",
                unsafe_allow_html=True
            )

            # Prepare cohort benchmarks
            benchmarks_dict = {}
            if active_df is not None:
                for f in feature_columns:
                    if f in active_df.columns and pd.api.types.is_numeric_dtype(active_df[f]):
                        benchmarks_dict[f] = {"mean": float(active_df[f].mean())}

            # Top features
            top_feats = trained_model.get("feature_importance_df", pd.DataFrame()).head(4).to_dict(orient="records")

            # Quick tailored recommendations
            quick_recs = []
            att = float(inputs_current.get("attendance", 100))
            std = float(inputs_current.get("study_hours", 5))
            slp = float(inputs_current.get("sleep_hours", 7))
            if att < 75:
                quick_recs.append(f"Target immediate attendance recovery (current: {att:.1f}% vs 75% threshold).")
            if std < 4:
                quick_recs.append(f"Expand daily study allocation by 1.5 to 2.0 hours (current: {std:.1f} hrs/day).")
            if slp < 6.5:
                quick_recs.append(f"Maintain 7-8 hours of sleep for memory consolidation (current: {slp:.1f} hrs/day).")
            if not quick_recs:
                quick_recs.append("Maintain consistent study schedule and proactive lecture engagement.")

            try:
                pdf_bytes = pdf_report_generator.generate_student_performance_pdf(
                    student_name=st.session_state.username,
                    predicted_category=pred_cat,
                    risk_tier=risk_tier,
                    risk_color_hex=risk_color,
                    input_values=inputs_current,
                    probabilities=prob_dict,
                    top_features=top_feats,
                    recommendations=quick_recs,
                    cohort_benchmarks=benchmarks_dict
                )

                st.download_button(
                    label="📥 Download Assessment PDF Report",
                    data=pdf_bytes,
                    file_name=f"Student_Report_{st.session_state.username}_{pred_cat}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as pdf_err:
                st.error(f"Could not generate PDF: {str(pdf_err)}")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="standard-card">
                <h3 style="margin-top:0; font-size:1.1rem;">Inference Engine Ready</h3>
                <p style="color:#94A3B8; font-size:0.88rem; line-height:1.6;">
                    Adjust student parameters in the form and click
                    <b>Predict Student Performance Category</b> to execute inference through the Decision Tree pipeline.
                </p>
                <div style="background:#0E1626; padding:12px; border-radius:8px; border:1px solid rgba(148,163,184,0.1); font-size:0.82rem; color:#94A3B8;">
                    ℹ Preprocessing and imputation are executed using the exact Scikit-learn Pipeline fitted during model training.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. STUDENT VS. CLASS / COHORT COMPARATIVE ANALYTICS
    # -------------------------------------------------------------
    if has_prediction and active_df is not None:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 👥 Student vs. Class / Cohort Analytics")
        st.markdown(
            f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
            f"Comparing evaluated student indicators against active cohort benchmarks (<b>{st.session_state.active_dataset_name}</b>, N = {len(active_df):,}):"
            f"</p>",
            unsafe_allow_html=True
        )

        comp_cols = [c for c in ["study_hours", "attendance", "previous_score", "sleep_hours"] if c in active_df.columns and c in input_values]

        if comp_cols:
            c_cards = st.columns(len(comp_cols))
            for i, c_name in enumerate(comp_cols):
                with c_cards[i]:
                    student_val = float(input_values[c_name])
                    cohort_series = active_df[c_name].dropna()
                    c_mean = float(cohort_series.mean())
                    percentile = float((cohort_series < student_val).mean() * 100)

                    delta = student_val - c_mean
                    delta_str = f"{'+' if delta >= 0 else ''}{delta:.1f}"

                    render_metric_card(
                        title=c_name.replace("_", " ").title(),
                        value=f"{student_val:.1f}",
                        subtext=f"Cohort Mean: {c_mean:.1f} ({delta_str}) &bull; {percentile:.0f}th %ile",
                        top_accent=True
                    )
        else:
            st.info("ℹ Active dataset does not contain standard numeric comparison columns (study_hours, attendance, previous_score).")

        st.markdown('</div>', unsafe_allow_html=True)
