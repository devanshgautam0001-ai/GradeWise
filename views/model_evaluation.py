"""
Model Evaluation and Training View
Student Performance Prediction and Analysis System
Decision Tree Classifier training controls, test-set metrics, confusion matrix,
classification report, and feature importance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml_engine
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_model_evaluation_view():
    """Render Decision Tree Classifier training and evaluation dashboard."""
    render_page_header(
        title="Model Training & Evaluation Dashboard",
        subtitle="Configure Decision Tree hyperparameters, train the pipeline on the active dataset, and evaluate test-set performance metrics.",
        badge="Machine Learning"
    )

    render_active_dataset_badge(st.session_state)

    active_df = getattr(st.session_state, "active_df", None)
    if active_df is None:
        st.warning("⚠️ No active dataset is currently loaded. Please go to Dataset Management to upload or load data.")
        return

    # Check target and feature configuration
    target_col = st.session_state.get("ml_target_col")
    feature_cols = st.session_state.get("ml_feature_cols")
    is_numeric_target = st.session_state.get("ml_is_numeric_target", False)
    thresholds = st.session_state.get("ml_thresholds")

    # If not explicitly configured, initialize sensible defaults
    if not target_col or target_col not in active_df.columns:
        col_info = ml_engine.inspect_candidate_columns(active_df)
        target_col = col_info["recommended_targets"][0] if col_info["recommended_targets"] else active_df.columns[-1]
        is_numeric_target = pd.api.types.is_numeric_dtype(active_df[target_col])
        st.session_state.ml_target_col = target_col
        st.session_state.ml_is_numeric_target = is_numeric_target

    if not feature_cols or not all(f in active_df.columns for f in feature_cols):
        feature_cols = [c for c in active_df.columns if c != target_col]
        st.session_state.ml_feature_cols = feature_cols

    # -------------------------------------------------------------
    # 1. HYPERPARAMETER CONFIGURATION & TRAINING CONTROLS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Decision Tree Classifier Configuration")
    st.markdown(
        f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
        f"Active Target: <b style='color:#93C5FD;'>{target_col}</b> "
        f"({'Continuous Score' if is_numeric_target else 'Categorical'}) | "
        f"Features: <b style='color:#FFFFFF;'>{len(feature_cols)} columns</b>"
        f"</p>",
        unsafe_allow_html=True
    )

    c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)

    with c_p1:
        criterion = st.selectbox(
            "Splitting Criterion",
            options=["gini", "entropy"],
            index=0,
            help="Function to measure split quality. Gini Impurity or Information Gain (Entropy)."
        )

    with c_p2:
        max_depth_val = st.slider(
            "Max Tree Depth",
            min_value=1,
            max_value=20,
            value=6,
            help="Maximum depth of the tree to prevent overfitting."
        )

    with c_p3:
        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=50,
            value=10,
            help="Minimum number of samples required to split an internal node."
        )

    with c_p4:
        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=30,
            value=5,
            help="Minimum number of samples required to be at a leaf node."
        )

    with c_p5:
        test_size = st.slider(
            "Test Set Ratio",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            help="Percentage of cohort reserved for evaluation."
        )

    train_btn = st.button("🚀 Train Decision Tree Classifier", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. MODEL TRAINING EXECUTION
    # -------------------------------------------------------------
    if train_btn:
        with st.spinner("Executing preprocessing pipeline and fitting Decision Tree Classifier..."):
            ok, msg, train_res = ml_engine.train_decision_tree_pipeline(
                df=active_df,
                target_column=target_col,
                feature_columns=feature_cols,
                is_numeric_target=is_numeric_target,
                score_thresholds=thresholds,
                criterion=criterion,
                max_depth=max_depth_val,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                test_size=test_size
            )

        if ok and train_res is not None:
            st.session_state.trained_dt_model = train_res
            st.success(f"✓ {msg}")
            st.toast("Model trained and cached for predictions!")
        else:
            st.error(f"❌ Training Failed: {msg}")

    # Load existing model from session state or disk if present
    trained_model = st.session_state.get("trained_dt_model")
    if trained_model is None:
        saved = ml_engine.load_saved_decision_tree()
        if saved is not None:
            st.session_state.trained_dt_model = saved
            trained_model = saved

    if trained_model is None:
        st.info("ℹ Click **Train Decision Tree Classifier** above to build the predictive model from the active dataset.")
        return

    # -------------------------------------------------------------
    # 3. TEST SET PERFORMANCE METRICS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Test Set Evaluation Metrics")
    st.markdown(
        f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
        f"Evaluated on unseen holdout test split (N = {trained_model['test_samples']} students). "
        f"Trained on N = {trained_model['train_samples']} students ({trained_model['training_time_sec']}s train time)."
        f"</p>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            title="Accuracy",
            value=f"{trained_model['accuracy'] * 100:.2f}%",
            subtext=f"Total Test Samples: {trained_model['test_samples']}",
            top_accent=True
        )
    with m2:
        render_metric_card(
            title="Weighted F1 Score",
            value=f"{trained_model['f1_weighted']:.3f}",
            subtext=f"Macro F1: {trained_model['f1_macro']:.3f}",
            top_accent=True
        )
    with m3:
        render_metric_card(
            title="Weighted Precision",
            value=f"{trained_model['precision_weighted']:.3f}",
            subtext=f"Macro Precision: {trained_model['precision_macro']:.3f}",
            top_accent=True
        )
    with m4:
        render_metric_card(
            title="Weighted Recall",
            value=f"{trained_model['recall_weighted']:.3f}",
            subtext=f"Macro Recall: {trained_model['recall_macro']:.3f}",
            top_accent=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. CONFUSION MATRIX & CLASSIFICATION REPORT
    # -------------------------------------------------------------
    col_cm, col_rep = st.columns([1, 1.2])

    with col_cm:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 🔲 Confusion Matrix Heatmap")
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.82rem; margin-bottom:12px;'>"
            "Predicted vs. ground truth categories on the test set."
            "</p>",
            unsafe_allow_html=True
        )

        cm = trained_model["confusion_matrix"]
        classes = trained_model["classes"]

        # Styled dark theme confusion matrix with matplotlib
        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="#0E1626")
        ax.set_facecolor("#0E1626")

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right", color="#CBD5E1", fontsize=8)
        ax.set_yticklabels(classes, color="#CBD5E1", fontsize=8)

        ax.set_xlabel("Predicted Category", color="#94A3B8", fontsize=9, labelpad=8)
        ax.set_ylabel("Actual Ground Truth", color="#94A3B8", fontsize=9, labelpad=8)
        ax.set_title("Test Set Confusion Matrix", color="#FFFFFF", fontsize=10, pad=10)

        # Annotate cell numbers
        thresh = cm.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm[i, j]
                color = "white" if val > thresh else "#94A3B8"
                ax.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#1E293B")
        ax.spines["bottom"].set_color("#1E293B")

        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rep:
        st.markdown('<div class="standard-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Classification Report")
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.82rem; margin-bottom:12px;'>"
            "Precision, recall, and F1-score across all student performance classes."
            "</p>",
            unsafe_allow_html=True
        )

        clf_dict = trained_model["classification_report"]
        # Format into clean table
        report_rows = []
        for key, vals in clf_dict.items():
            if isinstance(vals, dict):
                report_rows.append({
                    "Class / Average": key,
                    "Precision": f"{vals.get('precision', 0):.3f}",
                    "Recall": f"{vals.get('recall', 0):.3f}",
                    "F1-Score": f"{vals.get('f1-score', 0):.3f}",
                    "Support": int(vals.get("support", 0))
                })

        report_df = pd.DataFrame(report_rows)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.1); border-radius:8px; padding:10px 12px; margin-top:14px; font-size:0.8rem; color:#94A3B8; line-height:1.5;">
            <b style="color:#FFFFFF;">Metric Reference:</b><br>
            • <b>Precision:</b> Accuracy of positive category predictions.<br>
            • <b>Recall:</b> Sensitivity in detecting students belonging to that category.<br>
            • <b>F1-Score:</b> Harmonic mean of precision and recall.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. FEATURE IMPORTANCE RANKING
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Decision Tree Feature Importance")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
        "Actual variance / Gini reduction contribution of each feature in the trained Decision Tree:"
        "</p>",
        unsafe_allow_html=True
    )

    imp_df = trained_model["feature_importance_df"]

    col_chart, col_tbl = st.columns([1.3, 0.7])

    with col_chart:
        # Plotly/Matplotlib horizontal bar chart
        fig2, ax2 = plt.subplots(figsize=(7, max(3.0, len(imp_df) * 0.4)), facecolor="#0E1626")
        ax2.set_facecolor("#0E1626")

        y_positions = np.arange(len(imp_df))
        ax2.barh(y_positions, imp_df["Percentage"], color="#3B82F6", height=0.6, edgecolor="none")
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(imp_df["Feature"], color="#CBD5E1", fontsize=9)
        ax2.invert_yaxis()  # Top feature at top
        ax2.set_xlabel("Contribution Percentage (%)", color="#94A3B8", fontsize=9)
        ax2.tick_params(colors="#CBD5E1")
        ax2.grid(axis="x", color="#1E293B", linestyle="--", alpha=0.6)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#1E293B")
        ax2.spines["bottom"].set_color("#1E293B")

        for idx, (p, v) in enumerate(zip(imp_df["Percentage"], imp_df["Importance"])):
            ax2.text(p + 0.8, idx, f"{p:.1f}%", va="center", color="#FFFFFF", fontsize=8, fontweight="bold")

        st.pyplot(fig2)
        plt.close(fig2)

    with col_tbl:
        display_imp = imp_df.copy()
        display_imp["Importance"] = display_imp["Importance"].round(4)
        display_imp["Percentage"] = display_imp["Percentage"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_imp, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)
