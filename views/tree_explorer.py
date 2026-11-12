"""
Decision Tree Explorer View
Student Performance Prediction and Analysis System
Interactive Decision Tree visualization, structural metrics, split condition inspection,
and mathematical theory of tree learning, entropy, and Gini impurity.
"""

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import ml_engine
from styles import render_page_header, render_active_dataset_badge, render_metric_card


def render_tree_explorer_view():
    """Render Decision Tree Explorer interface."""
    render_page_header(
        title="Decision Tree Architecture & Rule Explorer",
        subtitle="Inspect the trained Decision Tree, navigate hierarchical split conditions, examine leaf nodes, and explore mathematical formulations.",
        badge="Model Interpretability"
    )

    render_active_dataset_badge(st.session_state)

    trained_model = st.session_state.get("trained_dt_model")
    if trained_model is None:
        saved = ml_engine.load_saved_decision_tree()
        if saved is not None:
            st.session_state.trained_dt_model = saved
            trained_model = saved

    if trained_model is None:
        st.warning("⚠️ No trained Decision Tree model found. Please train a model in the **Model Evaluation** section first.")
        return

    classifier = trained_model["classifier"]
    classes = trained_model["classes"]
    clean_features = trained_model["clean_feature_names"]

    # -------------------------------------------------------------
    # 1. STRUCTURAL ARCHITECTURE METRICS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🌲 Decision Tree Structural Attributes")
    st.markdown(
        f"<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:14px;'>"
        f"Model trained on <b>{trained_model['train_samples']} student records</b> using "
        f"<b>{trained_model['hyperparameters']['criterion'].title()}</b> criterion."
        f"</p>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            title="Total Nodes",
            value=str(trained_model["tree_node_count"]),
            subtext="Internal decision + leaf nodes",
            top_accent=True
        )
    with m2:
        render_metric_card(
            title="Max Tree Depth",
            value=str(trained_model["tree_depth"]),
            subtext="Root to deepest leaf path",
            top_accent=True
        )
    with m3:
        render_metric_card(
            title="Leaf Nodes (Partitions)",
            value=str(trained_model["tree_n_leaves"]),
            subtext="Terminal decision categories",
            top_accent=True
        )
    with m4:
        render_metric_card(
            title="Output Categories",
            value=str(len(classes)),
            subtext=", ".join(classes),
            top_accent=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. GRAPHICAL TREE VISUALIZATION
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 🌿 Graphical Decision Tree Diagram")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:12px;'>"
        "Visual representation of actual decision nodes, splitting thresholds, sample counts, and majority classes. "
        "Use the depth slider to adjust the visualization detail level."
        "</p>",
        unsafe_allow_html=True
    )

    max_viz_depth = st.slider(
        "Display Tree Depth Level",
        min_value=1,
        max_value=max(1, trained_model["tree_depth"]),
        value=min(3, max(1, trained_model["tree_depth"])),
        help="Control how many levels of branches to render simultaneously."
    )

    with st.spinner("Rendering Decision Tree graph..."):
        # Dynamic figure size based on depth
        fig_width = min(22, 10 + max_viz_depth * 3)
        fig_height = min(12, 5 + max_viz_depth * 2)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="#0E1626")
        ax.set_facecolor("#0E1626")

        plot_tree(
            classifier,
            max_depth=max_viz_depth,
            feature_names=clean_features,
            class_names=classes,
            filled=True,
            rounded=True,
            precision=2,
            fontsize=8,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

    with st.expander("📄 View Hierarchical Textual Decision Path (Full Tree)"):
        tree_text = export_text(
            classifier,
            feature_names=clean_features,
            max_depth=10
        )
        st.code(tree_text, language="text")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. MATHEMATICAL FOUNDATIONS & LEARNING CONCEPTS
    # -------------------------------------------------------------
    st.markdown('<div class="standard-card">', unsafe_allow_html=True)
    st.markdown("### 📐 Mathematical Foundations of Decision Trees")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.88rem; margin-bottom:16px;'>"
        "The Decision Tree Classifier partitions the student feature space into non-overlapping hyper-rectangles using greedy recursive binary splitting."
        "</p>",
        unsafe_allow_html=True
    )

    th1, th2 = st.columns(2)

    with th1:
        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.12); border-radius:8px; padding:16px; margin-bottom:14px;">
            <div style="color:#93C5FD; font-weight:700; font-size:0.95rem; margin-bottom:8px;">1. Gini Impurity</div>
            <p style="color:#94A3B8; font-size:0.83rem; line-height:1.6; margin-bottom:10px;">
                Measures the probability of misclassifying a randomly chosen element from the set if it were randomly labeled according to the class distribution.
            </p>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2")
        st.markdown("""
            <p style="color:#CBD5E1; font-size:0.8rem; margin-top:8px;">
                Where <i>C</i> is the number of student performance classes and <i>p<sub>i</sub></i> is the probability of class <i>i</i>.
                A Gini of <b>0.0</b> indicates a pure node.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.12); border-radius:8px; padding:16px;">
            <div style="color:#93C5FD; font-weight:700; font-size:0.95rem; margin-bottom:8px;">3. Information Gain (IG)</div>
            <p style="color:#94A3B8; font-size:0.83rem; line-height:1.6; margin-bottom:10px;">
                The reduction in entropy achieved by partitioning a node based on candidate feature <i>A</i>:
            </p>
        """, unsafe_allow_html=True)
        st.latex(r"\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)")
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

    with th2:
        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.12); border-radius:8px; padding:16px; margin-bottom:14px;">
            <div style="color:#C4B5FD; font-weight:700; font-size:0.95rem; margin-bottom:8px;">2. Shannon Entropy</div>
            <p style="color:#94A3B8; font-size:0.83rem; line-height:1.6; margin-bottom:10px;">
                Quantifies the average amount of information or unpredictability present in the target distribution:
            </p>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)")
        st.markdown("""
            <p style="color:#CBD5E1; font-size:0.8rem; margin-top:8px;">
                A uniform distribution maximizes entropy (maximum chaos), whereas a single-class node has zero entropy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0E1626; border:1px solid rgba(148,163,184,0.12); border-radius:8px; padding:16px;">
            <div style="color:#C4B5FD; font-weight:700; font-size:0.95rem; margin-bottom:8px;">4. Leaf Node Classification</div>
            <p style="color:#94A3B8; font-size:0.83rem; line-height:1.6; margin-bottom:8px;">
                Once stopping criteria are reached (e.g., <code>max_depth</code>, <code>min_samples_split</code>), each leaf assigns the majority class:
            </p>
            <div style="background:#070B14; padding:8px 12px; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#6EE7B7;">
                class_pred = argmax_c (N_c / N_leaf)
            </div>
            <p style="color:#CBD5E1; font-size:0.8rem; margin-top:8px;">
                Class probabilities are derived directly from relative frequencies <i>p<sub>c</sub> = N<sub>c</sub> / N<sub>leaf</sub></i>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
