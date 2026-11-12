"""
Dataset Management View
Student Performance Prediction and Analysis System
Dataset upload, validation, preview, and ML target/feature configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import data_manager
import ml_engine
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_dataset_management_view():
    """Render Dataset Management and ML Configuration."""
    render_page_header(
        title="Dataset Management & Data Ingestion",
        subtitle="Upload student datasets, inspect data quality, and configure target and features for the Decision Tree Classifier.",
        badge="Data Hub"
    )

    render_active_dataset_badge(st.session_state)

    # -------------------------------------------------------------
    # 1. DATASET SOURCE SELECTION
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Dataset Source")

    source_tab1, source_tab2 = st.tabs(["📤 Upload Your Dataset", "📚 Use Sample Dataset"])

    with source_tab1:
        st.markdown(
            "<p style='color:#475569; font-size:0.88rem; margin-top:8px;'>"
            "Select or drag & drop a <b>CSV</b> or <b>Excel</b> (.xlsx) file."
            "</p>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            label="Upload Student Dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="cohort_file_uploader",
            help="Supported: CSV, Excel XLSX"
        )

        if uploaded_file is not None:
            is_new_upload = (
                st.session_state.get("last_uploaded_name") != uploaded_file.name
                or st.session_state.active_dataset_name != uploaded_file.name
            )

            if st.button("Apply & Activate Dataset") or is_new_upload:
                with st.spinner("Validating and parsing dataset..."):
                    valid, message, parsed_df, metadata = data_manager.validate_and_load_file(uploaded_file)

                if valid and parsed_df is not None:
                    data_manager.set_active_dataset(
                        st.session_state, parsed_df, uploaded_file.name, "Uploaded"
                    )
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.ml_target_col = None
                    st.session_state.ml_feature_cols = None
                    st.session_state.trained_dt_model = None
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

    with source_tab2:
        st.markdown(
            "<p style='color:#475569; font-size:0.88rem; margin-top:8px;'>"
            "Load the built-in benchmark dataset with 1,000 student records."
            "</p>",
            unsafe_allow_html=True
        )

        sample_choice = st.selectbox(
            "Select Sample Dataset",
            options=["advanced_student_data.csv (1,000 records)",
                     "student_data.csv (10 records)"]
        )

        if st.button("Load Sample Dataset"):
            chosen_path = "advanced_student_data.csv" if "advanced" in sample_choice else "student_data.csv"
            ok, msg = data_manager.reset_to_sample(st.session_state, chosen_path)
            if ok:
                st.session_state.ml_target_col = None
                st.session_state.ml_feature_cols = None
                st.session_state.trained_dt_model = None
                st.success(f"✅ Loaded: {chosen_path}")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    # Dataset action buttons
    st.markdown("<hr style='border:none; border-top:1px solid #E2E8F0; margin:16px 0;'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reset to Default Sample", use_container_width=True):
            data_manager.reset_to_sample(st.session_state, "advanced_student_data.csv")
            st.session_state.ml_target_col = None
            st.session_state.ml_feature_cols = None
            st.session_state.trained_dt_model = None
            st.rerun()
    with c2:
        if st.button("🗑️ Remove Active Dataset", use_container_width=True):
            st.session_state.active_df = None
            st.session_state.active_dataset_name = "None"
            st.session_state.active_dataset_type = "None"
            st.session_state.trained_dt_model = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. DATASET METRICS
    # -------------------------------------------------------------
    active_df = getattr(st.session_state, "active_df", None)
    if active_df is None:
        st.warning("⚠️ No dataset loaded. Upload a file or choose the sample dataset above.")
        return

    dataset_name = st.session_state.active_dataset_name
    dataset_type = st.session_state.active_dataset_type
    metrics = data_manager.compute_dataset_metrics(active_df, dataset_name, dataset_type)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Records", f"{metrics['row_count']:,}", f"~{metrics['memory_mb']} MB")
    with m2:
        render_metric_card("Columns", str(metrics["column_count"]),
                           f"{len(metrics['numerical_columns'])} Numeric, {len(metrics['categorical_columns'])} Categorical")
    with m3:
        render_metric_card("Missing Values", f"{metrics['total_missing_cells']:,}",
                           f"{metrics['total_missing_pct']}% of cells")
    with m4:
        render_metric_card("Duplicates", f"{metrics['duplicate_rows']:,}",
                           f"{metrics['duplicate_pct']}% duplicate")

    # Quality banner
    if metrics["total_missing_cells"] == 0 and metrics["duplicate_rows"] == 0:
        st.success("✓ Data quality passed — zero missing values and zero duplicates.")
    else:
        warnings = []
        if metrics["total_missing_cells"] > 0:
            warnings.append(f"{metrics['total_missing_cells']} missing values")
        if metrics["duplicate_rows"] > 0:
            warnings.append(f"{metrics['duplicate_rows']} duplicate rows")
        st.warning(f"⚠ {', '.join(warnings)} detected. Will be handled during training.")

    # -------------------------------------------------------------
    # 3. TARGET & FEATURE SELECTION
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Target & Feature Setup")

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
            index=default_target_idx
        )

        is_numeric_target = pd.api.types.is_numeric_dtype(active_df[selected_target])

        if is_numeric_target:
            st.info(f"Column `{selected_target}` is numeric — it will be converted into performance categories using the thresholds on the right.")
        else:
            unique_classes = active_df[selected_target].dropna().unique()
            st.success(f"Categorical target with {len(unique_classes)} classes: {', '.join([str(c) for c in unique_classes[:6]])}")

    with col_thresh:
        if is_numeric_target:
            st.markdown("**Score Category Thresholds**")
            saved_thresh = st.session_state.get("ml_thresholds", ml_engine.DEFAULT_SCORE_THRESHOLDS)

            t_c1, t_c2, t_c3 = st.columns(3)
            with t_c1:
                t_exc = st.number_input("Excellent (≥)", min_value=0.0, max_value=100.0, value=float(saved_thresh.get("Excellent", 80.0)), step=1.0)
            with t_c2:
                t_good = st.number_input("Good (≥)", min_value=0.0, max_value=t_exc, value=float(saved_thresh.get("Good", 65.0)), step=1.0)
            with t_c3:
                t_avg = st.number_input("Average (≥)", min_value=0.0, max_value=t_good, value=float(saved_thresh.get("Average", 50.0)), step=1.0)

            score_thresholds = {"Excellent": t_exc, "Good": t_good, "Average": t_avg}

            temp_categorized = ml_engine.categorize_numeric_scores(active_df[selected_target], score_thresholds)
            counts = temp_categorized.value_counts().to_dict()
            st.caption(
                f"Exc: {counts.get('Excellent', 0)} | Good: {counts.get('Good', 0)} | "
                f"Avg: {counts.get('Average', 0)} | Needs Imp: {counts.get('Needs Improvement', 0)}"
            )
        else:
            score_thresholds = None
            st.caption("Categorical target — used directly as performance labels.")

    # Feature selection
    st.markdown("---")
    st.markdown("**Select Predictive Features**")

    candidate_features = [c for c in all_cols if c != selected_target]
    default_selected_features = [
        c for c in candidate_features if c not in column_info["id_like_columns"]
    ]

    saved_features = st.session_state.get("ml_feature_cols")
    if saved_features and all(f in candidate_features for f in saved_features):
        default_selected_features = saved_features

    selected_features = st.multiselect(
        "Features to include in Decision Tree training:",
        options=candidate_features,
        default=default_selected_features
    )

    if not selected_features:
        st.error("⚠️ Select at least one feature to proceed.")

    if st.button("💾 Save ML Configuration", use_container_width=True):
        if selected_features:
            st.session_state.ml_target_col = selected_target
            st.session_state.ml_feature_cols = selected_features
            st.session_state.ml_is_numeric_target = is_numeric_target
            st.session_state.ml_thresholds = score_thresholds
            st.success(f"✓ Saved: Target '{selected_target}' with {len(selected_features)} features.")
        else:
            st.error("No features selected.")

    feat_count = len(st.session_state.get('ml_feature_cols') or selected_features or [])
    target_name = st.session_state.get('ml_target_col', selected_target)
    st.caption(f"Active Config: Target = **{target_name}** | Features = **{feat_count} selected**")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. DATASET PREVIEW
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Preview (First 10 Records)")
    st.dataframe(metrics["first_10_records"], use_container_width=True, hide_index=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. SCHEMA & STATS
    # -------------------------------------------------------------
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Column Schema & Missing Values")
        st.dataframe(metrics["missing_breakdown_df"], use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Descriptive Statistics")
        if metrics["numerical_summary"] is not None:
            st.dataframe(metrics["numerical_summary"], use_container_width=True, hide_index=False)
        else:
            st.info("No numerical features available.")
        st.markdown('</div>', unsafe_allow_html=True)
