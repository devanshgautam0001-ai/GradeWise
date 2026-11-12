"""
Dataset Management View
Student Performance Prediction and Analysis System
Full dataset ingestion, file validation, quality checks, analytical preview,
and target/feature selection for Decision Tree classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import data_manager
import ml_engine
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_dataset_management_view():
    """Render Dataset Management, Ingestion, Quality Inspection, and Target Configuration."""
    render_page_header(
        title="Dataset Management & Data Ingestion",
        subtitle="Upload institutional student cohorts, inspect dataset schemas, validate data completeness, and configure target performance categories.",
        badge="Data Hub"
    )

    render_active_dataset_badge(st.session_state)

    # -------------------------------------------------------------
    # 1. DATASET SOURCE SELECTION: UPLOAD VS. SAMPLE
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Dataset Source Configuration")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.9rem; margin-bottom:14px;'>"
        "Choose whether to upload an institutional dataset (CSV / Excel) or load the verified sample benchmark dataset."
        "</p>",
        unsafe_allow_html=True
    )

    source_tab1, source_tab2 = st.tabs(["📤 Upload Your Dataset", "📚 Use Sample Dataset"])

    # TAB 1: UPLOAD DATASET
    with source_tab1:
        st.markdown(
            "<p style='color:#CBD5E1; font-size:0.88rem; margin-top:8px;'>"
            "Select or drag & drop a <b>CSV</b> (.csv) or <b>Excel</b> (.xlsx, .xls) file from your local computer. "
            "The file will be validated for integrity, headers, and record consistency."
            "</p>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            label="Upload Student Dataset (CSV or Excel XLSX)",
            type=["csv", "xlsx", "xls"],
            key="cohort_file_uploader",
            help="Supported formats: CSV, Excel XLSX. Browse your computer or drag and drop files here."
        )

        if uploaded_file is not None:
            is_new_upload = (
                st.session_state.get("last_uploaded_name") != uploaded_file.name
                or st.session_state.active_dataset_name != uploaded_file.name
            )

            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                process_btn = st.button("Apply & Activate Dataset", use_container_width=True)

            if process_btn or is_new_upload:
                with st.spinner("Validating and parsing dataset..."):
                    valid, message, parsed_df, metadata = data_manager.validate_and_load_file(uploaded_file)

                if valid and parsed_df is not None:
                    data_manager.set_active_dataset(
                        st.session_state,
                        parsed_df,
                        uploaded_file.name,
                        "Uploaded"
                    )
                    st.session_state.last_uploaded_name = uploaded_file.name
                    # Reset model configuration for new dataset
                    st.session_state.ml_target_col = None
                    st.session_state.ml_feature_cols = None
                    st.session_state.trained_dt_model = None
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ Data Validation Error: {message}")
                    if metadata.get("error_type") == "unsupported_format":
                        st.info("💡 Please convert your file into comma-separated values (.csv) or Microsoft Excel (.xlsx) format.")
                    elif metadata.get("error_type") == "empty_file":
                        st.info("💡 The uploaded file contains 0 bytes. Ensure you have saved data in the file before uploading.")
                    elif metadata.get("error_type") in ["parser_error", "corrupted_file"]:
                        st.info("💡 Verify that your file is not password-protected and that column delimiters are standard.")

    # TAB 2: USE SAMPLE DATASET
    with source_tab2:
        st.markdown(
            "<p style='color:#CBD5E1; font-size:0.88rem; margin-top:8px;'>"
            "The benchmark cohort dataset contains 1,000 verified student records with standardized behavioral, "
            "academic, and sleep habit attributes."
            "</p>",
            unsafe_allow_html=True
        )

        sample_choice = st.selectbox(
            "Select Benchmark Sample Cohort",
            options=["advanced_student_data.csv (1,000 records, 8 behavioral & academic features)",
                     "student_data.csv (10 records, introductory sample)"]
        )

        col_sample_btn, col_sample_space = st.columns([1.2, 2.8])
        with col_sample_btn:
            if st.button("Load Selected Sample Dataset", use_container_width=True):
                chosen_path = "advanced_student_data.csv" if "advanced" in sample_choice else "student_data.csv"
                ok, msg = data_manager.reset_to_sample(st.session_state, chosen_path)
                if ok:
                    st.session_state.ml_target_col = None
                    st.session_state.ml_feature_cols = None
                    st.session_state.trained_dt_model = None
                    st.success(f"✅ Active dataset updated to benchmark sample: {chosen_path}")
                    st.rerun()
                else:
                    st.error(f"❌ Could not load sample: {msg}")

    # ACTIVE DATASET ACTIONS ROW
    st.markdown("<hr style='border:none; border-top:1px solid rgba(148,163,184,0.12); margin:16px 0;'>", unsafe_allow_html=True)
    c_act1, c_act2, c_act3 = st.columns([1.5, 1.5, 2.5])

    with c_act1:
        if st.button("🔄 Reset to Default Sample", use_container_width=True):
            data_manager.reset_to_sample(st.session_state, "advanced_student_data.csv")
            st.session_state.ml_target_col = None
            st.session_state.ml_feature_cols = None
            st.session_state.trained_dt_model = None
            st.toast("Active dataset reset to verified default benchmark.")
            st.rerun()

    with c_act2:
        if st.button("🗑️ Remove Active Dataset", use_container_width=True):
            st.session_state.active_df = None
            st.session_state.active_dataset_name = "None"
            st.session_state.active_dataset_type = "None"
            st.session_state.trained_dt_model = None
            st.warning("Active dataset removed from session.")
            st.rerun()

    with c_act3:
        current_name = getattr(st.session_state, "active_dataset_name", "None")
        current_type = getattr(st.session_state, "active_dataset_type", "None")
        st.markdown(
            f"<div style='text-align:right; font-size:0.85rem; color:#94A3B8; padding-top:8px;'>"
            f"Active: <strong style='color:#FFFFFF;'>{current_name}</strong> "
            f"({current_type})</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. ACTIVE DATASET METRICS & VALIDATION REPORT
    # -------------------------------------------------------------
    active_df = getattr(st.session_state, "active_df", None)
    if active_df is None:
        st.warning("⚠️ No active dataset is currently selected. Upload a file above or choose the verified sample dataset.")
        return

    dataset_name = st.session_state.active_dataset_name
    dataset_type = st.session_state.active_dataset_type
    metrics = data_manager.compute_dataset_metrics(active_df, dataset_name, dataset_type)

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            title="Total Records (Rows)",
            value=f"{metrics['row_count']:,}",
            subtext=f"Memory: ~{metrics['memory_mb']} MB",
            top_accent=True
        )
    with m2:
        render_metric_card(
            title="Total Columns",
            value=str(metrics["column_count"]),
            subtext=f"{len(metrics['numerical_columns'])} Numeric, {len(metrics['categorical_columns'])} Categorical",
            top_accent=True
        )
    with m3:
        missing_val = f"{metrics['total_missing_cells']:,}"
        missing_sub = f"{metrics['total_missing_pct']}% of total matrix cells"
        render_metric_card(
            title="Missing Values",
            value=missing_val,
            subtext=missing_sub,
            top_accent=True
        )
    with m4:
        dup_val = f"{metrics['duplicate_rows']:,}"
        dup_sub = f"{metrics['duplicate_pct']}% duplicate records"
        render_metric_card(
            title="Duplicate Records",
            value=dup_val,
            subtext=dup_sub,
            top_accent=True
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Validation Checks Alert Banner
    if metrics["total_missing_cells"] == 0 and metrics["duplicate_rows"] == 0:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:10px 16px; margin-bottom:16px;">
            <span style="color:#10B981; font-weight:700;">✓ Data Quality Passed:</span>
            <span style="color:#E2E8F0; font-size:0.88rem; margin-left:6px;">Zero missing cells and zero duplicate rows detected across all features. Ready for machine learning inference.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        warnings = []
        if metrics["total_missing_cells"] > 0:
            warnings.append(f"{metrics['total_missing_cells']} missing values detected")
        if metrics["duplicate_rows"] > 0:
            warnings.append(f"{metrics['duplicate_rows']} duplicate rows detected")
        warn_str = ", ".join(warnings)
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); border-radius:8px; padding:10px 16px; margin-bottom:16px;">
            <span style="color:#F59E0B; font-weight:700;">⚠ Data Hygiene Notice:</span>
            <span style="color:#E2E8F0; font-size:0.88rem; margin-left:6px;">{warn_str}. Imputers will handle missing values automatically during training.</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. TARGET COLUMN SELECTION & CATEGORIZATION (PHASE 2 FEATURE)
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Machine Learning Target & Feature Setup")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.88rem; margin-bottom:16px;'>"
        "Configure the prediction target column, convert continuous numerical scores into documented performance categories, "
        "and select predictive features for the Decision Tree Classifier."
        "</p>",
        unsafe_allow_html=True
    )

    column_info = ml_engine.inspect_candidate_columns(active_df)
    all_cols = column_info["all_columns"]

    # Target column selector
    default_target_idx = 0
    saved_target = st.session_state.get("ml_target_col")
    if saved_target in all_cols:
        default_target_idx = all_cols.index(saved_target)
    elif column_info["recommended_targets"] and column_info["recommended_targets"][0] in all_cols:
        default_target_idx = all_cols.index(column_info["recommended_targets"][0])

    col_target, col_thresh = st.columns([1.1, 0.9])

    with col_target:
        selected_target = st.selectbox(
            "Select Target Column to Predict",
            options=all_cols,
            index=default_target_idx,
            help="Choose the student performance category or final score column."
        )

        is_numeric_target = pd.api.types.is_numeric_dtype(active_df[selected_target])

        if is_numeric_target:
            st.markdown(
                f"<div style='background:#0E1626; border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:10px 12px; margin-top:8px;'>"
                f"<span style='color:#93C5FD; font-weight:600; font-size:0.82rem;'>Numeric Target Detected:</span> "
                f"<span style='color:#CBD5E1; font-size:0.82rem;'>Column <code>{selected_target}</code> contains continuous numbers. "
                f"It will be converted into documented performance categories using the threshold ranges configured on the right.</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            unique_classes = active_df[selected_target].dropna().unique()
            st.markdown(
                f"<div style='background:#0E1626; border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:10px 12px; margin-top:8px;'>"
                f"<span style='color:#10B981; font-weight:600; font-size:0.82rem;'>Categorical Target Detected:</span> "
                f"<span style='color:#CBD5E1; font-size:0.82rem;'>Found {len(unique_classes)} distinct categories: "
                f"<b>{', '.join([str(c) for c in unique_classes[:6]])}</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with col_thresh:
        if is_numeric_target:
            st.markdown("<strong style='font-size:0.85rem; color:#FFFFFF;'>Score Category Cutoff Thresholds</strong>", unsafe_allow_html=True)
            saved_thresh = st.session_state.get("ml_thresholds", ml_engine.DEFAULT_SCORE_THRESHOLDS)

            t_c1, t_c2, t_c3 = st.columns(3)
            with t_c1:
                t_exc = st.number_input("Excellent (≥)", min_value=0.0, max_value=100.0, value=float(saved_thresh.get("Excellent", 80.0)), step=1.0)
            with t_c2:
                t_good = st.number_input("Good (≥)", min_value=0.0, max_value=t_exc, value=float(saved_thresh.get("Good", 65.0)), step=1.0)
            with t_c3:
                t_avg = st.number_input("Average (≥)", min_value=0.0, max_value=t_good, value=float(saved_thresh.get("Average", 50.0)), step=1.0)

            score_thresholds = {
                "Excellent": t_exc,
                "Good": t_good,
                "Average": t_avg
            }

            # Preview class counts
            temp_categorized = ml_engine.categorize_numeric_scores(active_df[selected_target], score_thresholds)
            counts = temp_categorized.value_counts().to_dict()
            st.markdown(
                f"<div style='font-size:0.78rem; color:#94A3B8; margin-top:6px;'>"
                f"Class Breakdown: "
                f"<span style='color:#10B981;'>Exc: {counts.get('Excellent', 0)}</span> | "
                f"<span style='color:#3B82F6;'>Good: {counts.get('Good', 0)}</span> | "
                f"<span style='color:#F59E0B;'>Avg: {counts.get('Average', 0)}</span> | "
                f"<span style='color:#EF4444;'>Needs Imp: {counts.get('Needs Improvement', 0)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            score_thresholds = None
            st.markdown(
                "<div style='color:#94A3B8; font-size:0.85rem; padding-top:20px;'>"
                "Categorical target will be used directly as student performance labels."
                "</div>",
                unsafe_allow_html=True
            )

    # Feature selection
    st.markdown("<hr style='border:none; border-top:1px solid rgba(148,163,184,0.1); margin:16px 0 12px 0;'>", unsafe_allow_html=True)
    st.markdown("<strong style='font-size:0.9rem; color:#FFFFFF;'>Select Predictive Features (Exclude IDs / Irrelevant Columns)</strong>", unsafe_allow_html=True)

    candidate_features = [c for c in all_cols if c != selected_target]
    default_selected_features = [
        c for c in candidate_features
        if c not in column_info["id_like_columns"]
    ]

    saved_features = st.session_state.get("ml_feature_cols")
    if saved_features and all(f in candidate_features for f in saved_features):
        default_selected_features = saved_features

    selected_features = st.multiselect(
        "Features to include in Decision Tree training:",
        options=candidate_features,
        default=default_selected_features,
        help="Select features that influence student performance. Exclude IDs, names, or post-exam variables."
    )

    if not selected_features:
        st.error("⚠️ Please select at least one predictive feature to proceed.")

    col_save, col_info = st.columns([1.5, 3])
    with col_save:
        if st.button("💾 Save ML Configuration", use_container_width=True):
            if selected_features:
                st.session_state.ml_target_col = selected_target
                st.session_state.ml_feature_cols = selected_features
                st.session_state.ml_is_numeric_target = is_numeric_target
                st.session_state.ml_thresholds = score_thresholds
                st.success(f"✓ Configuration saved: Target '{selected_target}' with {len(selected_features)} features.")
                st.toast("Ready for model training! Go to Model Evaluation section.")
            else:
                st.error("Cannot save: No features selected.")

    with col_info:
        st.markdown(
            f"<div style='font-size:0.84rem; color:#94A3B8; padding-top:8px;'>"
            f"Active Config: Target = <b style='color:#FFFFFF;'>{st.session_state.get('ml_target_col', selected_target)}</b> | "
            f"Features = <b style='color:#FFFFFF;'>{len(st.session_state.get('ml_feature_cols', selected_features))} selected</b>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. DATASET PREVIEW (FIRST 10 RECORDS)
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Preview (First 10 Records)")
    st.markdown(
        f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:12px;'>"
        f"Showing first 10 rows of <b>{dataset_name}</b> ({metrics['row_count']:,} total records &times; {metrics['column_count']} features):"
        f"</p>",
        unsafe_allow_html=True
    )

    st.dataframe(
        metrics["first_10_records"],
        use_container_width=True,
        hide_index=False
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. SCHEMA & STATISTICAL SUMMARY
    # -------------------------------------------------------------
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Column Schema & Missing Values")
        st.dataframe(
            metrics["missing_breakdown_df"],
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Descriptive Statistics")
        if metrics["numerical_summary"] is not None:
            st.dataframe(
                metrics["numerical_summary"],
                use_container_width=True,
                hide_index=False
            )
        else:
            st.info("No numerical features available for descriptive statistics.")
        st.markdown('</div>', unsafe_allow_html=True)
