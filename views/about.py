"""
About Project View
Student Performance Prediction and Analysis System
Comprehensive system specifications, problem statement, objectives, proposed solution,
Decision Tree ML methodology, technology stack, architecture, limitations, and future scope.
"""

import streamlit as st
import pandas as pd
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_about_view():
    """Render About Project documentation interface."""
    render_page_header(
        title="About the Platform & Methodology",
        subtitle="Institutional system specifications, machine learning pipeline architecture, academic documentation, and project governance.",
        badge="Documentation"
    )

    render_active_dataset_badge(st.session_state)

    # -------------------------------------------------------------
    # 1. PROJECT INTRODUCTION & EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🎓 Project Introduction")
    st.markdown("""
    <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.7;">
        The <b>Student Performance Prediction and Analysis System</b> is an institutional-grade academic analytics platform 
        engineered to forecast student outcomes, detect early academic distress, and provide data-driven curricular interventions.
        Operating on transparent <b>Decision Tree Machine Learning algorithms</b> and strict empirical evaluations, the platform replaces 
        speculative academic assessments with interpretable decision rules, quantifiable classification metrics, and actionable advising roadmaps.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. PROBLEM STATEMENT & OBJECTIVES
    # -------------------------------------------------------------
    col_prob, col_obj = st.columns(2)

    with col_prob:
        st.markdown('<div class="standard-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Problem Statement")
        st.markdown("""
        <p style="color:#94A3B8; font-size:0.86rem; line-height:1.6;">
            In traditional university structures, student academic distress is often identified only after midterm or end-semester 
            examinations have concluded—too late for meaningful remedial intervention.
        </p>
        <ul style="color:#CBD5E1; font-size:0.84rem; line-height:1.7; padding-left:18px;">
            <li>High student-to-faculty ratios prevent individualized tracking of behavioral indicators.</li>
            <li>Early warning signs such as attendance lag, study deficits, and acute stress remain unquantified.</li>
            <li>Traditional grading systems lack predictive early-warning capabilities.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_obj:
        st.markdown('<div class="standard-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown("### 🎯 Project Objectives")
        st.markdown("""
        <ul style="color:#CBD5E1; font-size:0.84rem; line-height:1.8; padding-left:18px;">
            <li><b>Automated Data Ingestion:</b> Support dynamic CSV and Excel (.xlsx) student cohort uploads with schema validation.</li>
            <li><b>Interpretable Machine Learning:</b> Implement a Scikit-Learn Decision Tree Classifier with zero data leakage.</li>
            <li><b>Transparent Risk Stratification:</b> Classify students into Low, Medium, and High Risk tiers based on documented rules.</li>
            <li><b>Scenario Simulation:</b> Provide what-if sensitivity modeling to identify break-even study thresholds.</li>
            <li><b>Auditability & Privacy:</b> Maintain private SQLite history protected with salted PBKDF2-HMAC-SHA256 password hashing.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. PROPOSED SOLUTION & ALGORITHM EXPLANATION
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Machine Learning Algorithm: Decision Tree Classifier")
    st.markdown("""
    <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.7;">
        The system utilizes a supervised <b>Decision Tree Classifier (CART algorithm)</b> for multiclass categorization:
    </p>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div style="background:#0E1626; padding:14px; border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
            <div style="color:#3B82F6; font-weight:700; font-size:0.92rem;">Splitting Criteria</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:4px; line-height:1.5;">
                Selects the optimal feature and threshold at each node using <b>Gini Impurity</b> (measuring statistical variance) or 
                <b>Shannon Entropy</b> (maximizing Information Gain).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div style="background:#0E1626; padding:14px; border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
            <div style="color:#8B5CF6; font-weight:700; font-size:0.92rem;">Zero Data Leakage</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:4px; line-height:1.5;">
                Preprocessing <code>ColumnTransformer</code> (median imputers, scalers, and one-hot encoders) fits strictly on <code>X_train</code>, 
                preventing training contamination.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown("""
        <div style="background:#0E1626; padding:14px; border-radius:8px; border:1px solid rgba(148,163,184,0.1);">
            <div style="color:#10B981; font-weight:700; font-size:0.92rem;">Interpretable Leaves</div>
            <p style="color:#94A3B8; font-size:0.82rem; margin-top:4px; line-height:1.5;">
                Leaf nodes output probability vectors derived directly from node sample frequencies (<i>p<sub>c</sub> = N<sub>c</sub> / N<sub>leaf</sub></i>), 
                eliminating arbitrary probability heuristics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. TECHNOLOGY STACK
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Technology Stack & Library Ecosystem")

    stack_df = pd.DataFrame([
        {"Layer": "User Interface & Routing", "Technology": "Streamlit (v1.30+)", "Specifications": "Multi-view layout, reactive state, custom dark navy CSS theme"},
        {"Layer": "Machine Learning Backend", "Technology": "Scikit-Learn (v1.3+)", "Specifications": "DecisionTreeClassifier, ColumnTransformer, Pipeline, Metrics"},
        {"Layer": "Data Processing & Ingestion", "Technology": "Pandas (v2.0+) & NumPy", "Specifications": "Automated schema detection, statistical summaries, data cleaning"},
        {"Layer": "Spreadsheet Compatibility", "Technology": "OpenPyXL (v3.1+)", "Specifications": "Native validation and parsing of Microsoft Excel (.xlsx) workbooks"},
        {"Layer": "Visual Analytics & Plots", "Technology": "Matplotlib (v3.7+)", "Specifications": "Confusion matrix heatmaps, plot_tree visualization, sensitivity curves"},
        {"Layer": "PDF Report Generation", "Technology": "ReportLab (v4.0+)", "Specifications": "Publication-grade academic diagnostic PDF documents with tables & styling"},
        {"Layer": "Database & Authentication", "Technology": "SQLite3 & Hashlib", "Specifications": "Salted PBKDF2-HMAC-SHA256 password security, user-isolated history"},
    ])
    st.dataframe(stack_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. DATASET WORKFLOW & SYSTEM ARCHITECTURE
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 End-to-End System Architecture Workflow")
    st.markdown("""
    <div style="background:#0E1626; padding:16px; border-radius:8px; border:1px solid rgba(148,163,184,0.1); color:#CBD5E1; font-size:0.86rem; line-height:1.8;">
        <b>1. Ingestion & Validation:</b> User uploads CSV/Excel file or loads verified benchmark. Automated validators reject corrupt, empty, or unsupported files.<br>
        <b>2. Target Formulation:</b> User selects discrete target or converts continuous scores into documented categories (<i>Excellent, Good, Average, Needs Improvement</i>).<br>
        <b>3. Preprocessing:</b> Median imputation for numeric data, one-hot encoding for categorical variables. Preprocessor fitted strictly on training partition.<br>
        <b>4. Model Training:</b> Stratified split partitions data (80/20). DecisionTreeClassifier is trained with user-tuned depth and leaf hyperparameters.<br>
        <b>5. Out-of-Sample Evaluation:</b> Evaluates holdout test set to compute Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.<br>
        <b>6. Dynamic Prediction & Advisory:</b> Renders custom input form, predicts performance class and probabilities, maps institutional risk, and generates PDF report.<br>
        <b>7. Persistence:</b> Prediction records and inputs saved into user's private SQLite history.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 6. LIMITATIONS, FUTURE SCOPE & DISCLAIMER
    # -------------------------------------------------------------
    col_lim, col_fut = st.columns(2)

    with col_lim:
        st.markdown('<div class="standard-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown("### ⚠️ System Limitations")
        st.markdown("""
        <ul style="color:#94A3B8; font-size:0.84rem; line-height:1.7; padding-left:18px;">
            <li><b>Cohort Representativeness:</b> Model predictions reflect the historical patterns present in the training dataset. Biases in historical data may influence outputs.</li>
            <li><b>Axis-Parallel Splits:</b> Standard Decision Trees construct orthogonal decision boundaries, which may require higher depth to model correlated features.</li>
            <li><b>Single Estimator Variance:</b> While highly interpretable, individual decision trees have higher variance than ensemble methods.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fut:
        st.markdown('<div class="standard-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown("### 🔮 Future Scope")
        st.markdown("""
        <ul style="color:#CBD5E1; font-size:0.84rem; line-height:1.7; padding-left:18px;">
            <li><b>LMS Integration:</b> Direct API connectors for Moodle, Canvas, and Blackboard for automated attendance and grade sync.</li>
            <li><b>Multi-Semester Longitudinal Tracking:</b> Time-series analysis tracking student trajectories over several academic years.</li>
            <li><b>Pruning Optimization:</b> Cost-complexity post-pruning ($\alpha$-pruning) to automatically identify the minimal optimal tree.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="standard-card" style="border-left: 3px solid #3B82F6; margin-top:16px;">
        <h4 style="margin-top:0; color:#93C5FD; font-size:0.95rem;">🏛️ Institutional Advisory Disclaimer</h4>
        <p style="color:#94A3B8; font-size:0.83rem; line-height:1.6; margin:0;">
            The Student Performance Prediction and Analysis System is an educational diagnostic and academic counseling tool.
            All classifications and risk alerts are statistical estimates designed to support faculty mentoring and student self-improvement.
            This software does not constitute an official university transcript, academic grading instrument, or formal examination record.
        </p>
    </div>
    """, unsafe_allow_html=True)
