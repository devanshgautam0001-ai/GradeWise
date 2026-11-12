"""
Machine Learning Pipeline Module — Decision Tree Classifier
Student Performance Prediction and Analysis System

Provides complete machine learning lifecycle for student performance classification:
1. Target column detection, validation & categorical conversion
2. Leakage-free preprocessing (ColumnTransformer + Pipeline)
3. Decision Tree Classifier training with configurable hyperparameters
4. Comprehensive evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
5. Decision Tree inspection & visualization data extraction
6. Dynamic single & batch prediction with confidence probabilities
7. Rule-based Academic Risk mapping (clearly decoupled from ML tree)
8. What-If sensitivity simulation using actual model inferences
"""

import os
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

MODEL_BUNDLE_FILE = "trained_dt_model.pkl"

DEFAULT_SCORE_BINS = [0.0, 50.0, 70.0, 85.0, 100.0]
DEFAULT_PERFORMANCE_LABELS = ["Needs Improvement", "Average", "Good", "Excellent"]

# Standard rule-based risk definitions mapping performance category to risk tier
RISK_RULES_MAPPING = {
    "Excellent": {
        "tier": "Low Risk",
        "color": "#10B981",
        "description": "High mastery trajectory. Optimal academic standing with negligible risk of course failure."
    },
    "Good": {
        "tier": "Low Risk",
        "color": "#10B981",
        "description": "Above-average academic trajectory. Consistent study performance with stable pass margins."
    },
    "Average": {
        "tier": "Medium Risk",
        "color": "#F59E0B",
        "description": "Borderline course standing. Vulnerable to grade degradation without proactive study routine reinforcement."
    },
    "Needs Improvement": {
        "tier": "High Risk",
        "color": "#EF4444",
        "description": "Critical academic risk. High probability of failing required benchmarks. Immediate intervention advised."
    },
    "Fail": {
        "tier": "High Risk",
        "color": "#EF4444",
        "description": "Below passing threshold. Mandatory academic remediation and tutoring required."
    },
    "Pass": {
        "tier": "Low Risk",
        "color": "#10B981",
        "description": "Meets academic progression criteria."
    }
}


# =====================================================================
# 1. COLUMN DETECTION & TARGET VALIDATION
# =====================================================================

def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect DataFrame schema and identify numerical, categorical, and potential target columns.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    potential_targets = []
    # Any categorical column is a candidate target
    for c in categorical_cols:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 20:
            potential_targets.append(c)

    # Any numerical column with <= 10 unique integer values could also be a discrete target
    for c in numerical_cols:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 10:
            potential_targets.append(c)

    return {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "potential_targets": potential_targets,
        "all_cols": df.columns.tolist()
    }


def create_performance_categories(
    df: pd.DataFrame,
    score_col: str,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    new_col_name: str = "performance_category"
) -> Tuple[pd.DataFrame, str]:
    """
    Convert a continuous numerical score column into documented discrete academic performance categories.
    Returns (modified_df_copy, new_column_name).
    """
    if bins is None:
        bins = DEFAULT_SCORE_BINS
    if labels is None:
        labels = DEFAULT_PERFORMANCE_LABELS

    df_copy = df.copy()
    scores = df_copy[score_col].astype(float)

    # Ensure bins cover data range
    actual_min = scores.min()
    actual_max = scores.max()
    adjusted_bins = list(bins)
    if actual_min < adjusted_bins[0]:
        adjusted_bins[0] = float(np.floor(actual_min))
    if actual_max > adjusted_bins[-1]:
        adjusted_bins[-1] = float(np.ceil(actual_max))

    df_copy[new_col_name] = pd.cut(
        scores,
        bins=adjusted_bins,
        labels=labels,
        include_lowest=True,
        ordered=True
    ).astype(str)

    return df_copy, new_col_name


def validate_target(df: pd.DataFrame, target_col: str) -> Tuple[bool, str, Dict[str, int]]:
    """
    Validate that target column exists, is non-empty, and has at least 2 distinct classes.
    """
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset.", {}

    series = df[target_col].dropna()
    if series.empty:
        return False, f"Target column '{target_col}' contains only null values.", {}

    counts = series.value_counts().to_dict()
    n_classes = len(counts)

    if n_classes < 2:
        return (
            False,
            f"Target column '{target_col}' contains only {n_classes} class ({list(counts.keys())}). "
            "Classification requires at least 2 distinct performance classes.",
            counts
        )

    return True, f"Target '{target_col}' is valid with {n_classes} classes.", counts


# =====================================================================
# 2. PREPROCESSING PIPELINE (ColumnTransformer)
# =====================================================================

def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Build a leakage-free Scikit-Learn ColumnTransformer.
    - Numerical: Median imputation + Standard scaling
    - Categorical: Most frequent imputation + One-Hot Encoding (ignore unknown at test time)
    """
    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def get_transformed_feature_names(preprocessor: ColumnTransformer, numerical_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Extract clean feature names generated by ColumnTransformer."""
    feature_names = []
    try:
        # sklearn 1.0+ get_feature_names_out
        raw_names = preprocessor.get_feature_names_out()
        for name in raw_names:
            # Clean up prefixes like num__ or cat__
            clean_name = name.replace("num__", "").replace("cat__", "")
            feature_names.append(clean_name)
    except Exception:
        # Fallback manual reconstruction
        feature_names.extend(numerical_cols)
        if categorical_cols and "cat" in preprocessor.named_transformers_:
            try:
                ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_names = ohe.get_feature_names_out(categorical_cols)
                feature_names.extend(cat_names.tolist())
            except Exception:
                feature_names.extend(categorical_cols)
    return feature_names


# =====================================================================
# 3. DECISION TREE CLASSIFIER TRAINING
# =====================================================================

def train_decision_tree(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Train a genuine Scikit-Learn Decision Tree Classifier on the active dataset.
    Implements proper train-test split, leakage-free ColumnTransformer, and multiclass metrics.
    """
    if config is None:
        config = {}

    max_depth = config.get("max_depth", 5)
    min_samples_split = config.get("min_samples_split", 4)
    min_samples_leaf = config.get("min_samples_leaf", 2)
    criterion = config.get("criterion", "gini")
    test_size = float(config.get("test_size", 0.2))
    random_state = int(config.get("random_state", 42))

    # Basic validations
    if not feature_cols:
        return False, "At least one feature column must be selected.", None

    valid, msg, class_counts = validate_target(df, target_col)
    if not valid:
        return False, msg, None

    # Filter out rows where target is missing
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 10:
        return False, f"Dataset has only {len(clean_df)} complete rows; minimum 10 rows required for training.", None

    # Drop duplicate rows within feature set to avoid test contamination
    clean_df = clean_df.drop_duplicates(subset=feature_cols + [target_col])

    X = clean_df[feature_cols].copy()
    y = clean_df[target_col].astype(str)

    # Identify numerical and categorical features within selected feature_cols
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Determine class names and check if stratified split is possible
    unique_classes, counts = np.unique(y, return_counts=True)
    class_names = sorted(list(unique_classes))

    # Stratify requires at least 2 samples per class
    min_class_count = counts.min()
    use_stratify = y if min_class_count >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=use_stratify
        )
    except Exception:
        # Fallback to unstratified if split fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )

    # 1. Build and fit preprocessor ON TRAINING DATA ONLY
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    try:
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
    except Exception as e:
        return False, f"Preprocessing error: {str(e)}", None

    transformed_feature_names = get_transformed_feature_names(
        preprocessor, numerical_cols, categorical_cols
    )

    # 2. Initialize and train genuine Decision Tree Classifier
    dt_model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth if max_depth and max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    dt_model.fit(X_train_trans, y_train)

    # 3. Model Evaluation on held-out test data
    y_pred = dt_model.predict(X_test_trans)
    y_pred_proba = None
    if hasattr(dt_model, "predict_proba"):
        try:
            y_pred_proba = dt_model.predict_proba(X_test_trans)
        except Exception:
            pass

    # Ensure class labels line up with dt_model.classes_
    model_classes = list(dt_model.classes_)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=model_classes)
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Compute training set accuracy to observe overfitting/underfitting
    y_train_pred = dt_model.predict(X_train_trans)
    train_acc = float(accuracy_score(y_train, y_train_pred))

    # 4. Feature Importance Calculation from the trained Decision Tree
    raw_importances = dt_model.feature_importances_
    importance_records = []
    for fname, imp in zip(transformed_feature_names, raw_importances):
        importance_records.append({"Feature": fname, "Importance": float(imp)})

    importance_df = pd.DataFrame(importance_records).sort_values(by="Importance", ascending=False)
    if not importance_df.empty and importance_df["Importance"].sum() > 0:
        importance_df["Percentage"] = (importance_df["Importance"] / importance_df["Importance"].sum()) * 100
    else:
        importance_df["Percentage"] = 0.0

    # Aggregate importance back to original feature columns for high-level ranking
    orig_importance_dict = {f: 0.0 for f in feature_cols}
    for _, row in importance_df.iterrows():
        fname = row["Feature"]
        f_imp = row["Importance"]
        # Find which original feature this corresponds to
        matched = False
        for orig in feature_cols:
            if fname == orig or fname.startswith(f"{orig}_"):
                orig_importance_dict[orig] += f_imp
                matched = True
                break
        if not matched and fname in orig_importance_dict:
            orig_importance_dict[fname] += f_imp

    orig_importance_df = pd.DataFrame([
        {"Feature": k, "Importance": float(v)} for k, v in orig_importance_dict.items()
    ]).sort_values(by="Importance", ascending=False)
    total_orig = orig_importance_df["Importance"].sum()
    orig_importance_df["Percentage"] = (
        (orig_importance_df["Importance"] / total_orig) * 100 if total_orig > 0 else 0.0
    )

    # 5. Extract structural tree info
    tree_ = dt_model.tree_
    tree_info = {
        "n_nodes": int(tree_.node_count),
        "max_depth_actual": int(tree_.max_depth),
        "n_leaves": int(tree_.n_leaves),
        "n_classes": len(model_classes),
        "criterion": criterion,
        "n_features_in": len(transformed_feature_names)
    }

    # Generate text rules
    try:
        rules_text = export_text(
            dt_model,
            feature_names=transformed_feature_names,
            max_depth=min(4, tree_.max_depth)
        )
    except Exception:
        rules_text = "Decision rules text could not be exported."

    # Assemble comprehensive model bundle
    bundle: Dict[str, Any] = {
        "model": dt_model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "transformed_feature_names": transformed_feature_names,
        "target_col": target_col,
        "class_names": model_classes,
        "config": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "test_size": test_size,
            "random_state": random_state
        },
        "train_size": len(X_train),
        "test_size_actual": len(X_test),
        "total_samples": len(clean_df),
        "accuracy": round(acc, 4),
        "train_accuracy": round(train_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report_dict": clf_report,
        "feature_importances_transformed": importance_df.reset_index(drop=True),
        "feature_importances_original": orig_importance_df.reset_index(drop=True),
        "tree_info": tree_info,
        "tree_rules_text": rules_text,
        # Store test comparisons for evaluation view
        "test_actual": y_test.tolist(),
        "test_predicted": y_pred.tolist(),
    }

    # Save to disk automatically
    save_model_bundle(bundle)

    success_msg = (
        f"Decision Tree Classifier trained successfully on {len(X_train):,} training records. "
        f"Test Accuracy: {acc*100:.2f}%, F1 Score: {f1:.4f} across {len(model_classes)} categories."
    )
    return True, success_msg, bundle


# =====================================================================
# 4. PERSISTENCE (SAVE / LOAD BUNDLE)
# =====================================================================

def save_model_bundle(bundle: Dict[str, Any], filepath: str = MODEL_BUNDLE_FILE) -> bool:
    """Serialize and save trained model bundle to disk."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)
        return True
    except Exception:
        return False


def load_model_bundle(filepath: str = MODEL_BUNDLE_FILE) -> Optional[Dict[str, Any]]:
    """Load serialized model bundle from disk."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "model" in bundle and "preprocessor" in bundle:
            return bundle
        return None
    except Exception:
        return None


# =====================================================================
# 5. PREDICTION INTERFACE (Single & Batch)
# =====================================================================

def predict_single(bundle: Dict[str, Any], input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict performance category for a single student profile using the trained pipeline.
    Uses exact same feature order and ColumnTransformer fitted during model training.
    """
    model: DecisionTreeClassifier = bundle["model"]
    preprocessor: ColumnTransformer = bundle["preprocessor"]
    feature_cols: List[str] = bundle["feature_cols"]
    classes: List[str] = bundle["class_names"]

    # Build single-row DataFrame aligned with training feature columns
    ordered_values = {}
    for col in feature_cols:
        val = input_dict.get(col, 0.0)
        ordered_values[col] = [val]

    input_df = pd.DataFrame(ordered_values, columns=feature_cols)

    # Transform using training preprocessor (handles imputations, OHE, scaling)
    transformed = preprocessor.transform(input_df)

    # Predict class
    pred_class = str(model.predict(transformed)[0])

    # Predict probabilities if supported
    probabilities = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(transformed)[0]
            probabilities = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probas)}
            confidence = round(float(probabilities.get(pred_class, 1.0)), 4)
        except Exception:
            pass

    # Map to rule-based academic risk tier
    risk_info = classify_risk_from_prediction(pred_class, classes)

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "risk_tier": risk_info["tier"],
        "risk_color": risk_info["color"],
        "risk_description": risk_info["description"],
        "input_features": input_dict
    }


def predict_batch(bundle: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a multi-row DataFrame."""
    model: DecisionTreeClassifier = bundle["model"]
    preprocessor: ColumnTransformer = bundle["preprocessor"]
    feature_cols: List[str] = bundle["feature_cols"]

    # Filter columns to feature set
    X_input = df[feature_cols].copy()
    transformed = preprocessor.transform(X_input)
    predictions = model.predict(transformed)

    res_df = df.copy()
    res_df["Predicted_Class"] = predictions
    res_df["Risk_Tier"] = [
        classify_risk_from_prediction(p, bundle["class_names"])["tier"]
        for p in predictions
    ]
    return res_df


# =====================================================================
# 6. RULE-BASED ACADEMIC RISK CLASSIFICATION (Separated from Tree)
# =====================================================================

def classify_risk_from_prediction(
    predicted_class: str,
    class_names: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Map predicted performance class to documented institutional risk tiers.
    Clearly decoupled from the statistical Decision Tree model:
    - Excellent / Good / Pass with Distinction -> Low Risk (Green)
    - Average / Moderate -> Medium Risk (Amber)
    - Needs Improvement / Fail / Poor -> High Risk (Crimson)
    """
    clean_cls = predicted_class.strip()

    # Exact or substring match in documented rule mapping
    for key, rule in RISK_RULES_MAPPING.items():
        if key.lower() in clean_cls.lower():
            return rule

    # Fallback heuristic for arbitrary user-uploaded category labels
    lower_cls = clean_cls.lower()
    if any(w in lower_cls for w in ["a", "high", "top", "distinction", "honor", "pass"]):
        return {
            "tier": "Low Risk",
            "color": "#10B981",
            "description": "Favorable academic progress. Student meets key proficiency benchmarks."
        }
    elif any(w in lower_cls for w in ["mid", "medium", "c", "moderate", "fair"]):
        return {
            "tier": "Medium Risk",
            "color": "#F59E0B",
            "description": "Moderate academic standing. Performance requires targeted review."
        }
    else:
        return {
            "tier": "High Risk",
            "color": "#EF4444",
            "description": "Vulnerable academic position. Structured intervention recommended."
        }


# =====================================================================
# 7. WHAT-IF SENSITIVITY SIMULATOR
# =====================================================================

def simulate_what_if(
    bundle: Dict[str, Any],
    baseline_inputs: Dict[str, Any],
    vary_feature: str,
    values_range: List[Any]
) -> pd.DataFrame:
    """
    Run parametric sensitivity simulation across a range of values for a given feature.
    Evaluates real predictions from the actual trained Decision Tree model.
    """
    records = []
    base_dict = baseline_inputs.copy()

    for val in values_range:
        current = base_dict.copy()
        current[vary_feature] = val
        pred_res = predict_single(bundle, current)

        rec = {
            vary_feature: val,
            "Predicted_Category": pred_res["predicted_class"],
            "Confidence": round(pred_res["confidence"] * 100, 1),
            "Risk_Tier": pred_res["risk_tier"],
        }
        # Add individual class probabilities
        for cname, pval in pred_res["probabilities"].items():
            rec[f"Prob_{cname}"] = round(pval * 100, 1)

        records.append(rec)

    return pd.DataFrame(records)


# =====================================================================
# 8. DECISION TREE EXPLORER UTILITIES
# =====================================================================

def get_tree_structure(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract internal node arrays from the fitted Scikit-Learn DecisionTree.
    Used for custom rendering of root nodes, decision branches, splits, and leaves.
    """
    model: DecisionTreeClassifier = bundle["model"]
    tree_ = model.tree_
    feature_names = bundle["transformed_feature_names"]
    classes = bundle["class_names"]

    # Extract nodes
    nodes = []
    for i in range(tree_.node_count):
        is_leaf = tree_.children_left[i] == tree_.children_right[i]
        feat_idx = tree_.feature[i]
        feat_name = feature_names[feat_idx] if not is_leaf and 0 <= feat_idx < len(feature_names) else None
        threshold = float(tree_.threshold[i]) if not is_leaf else None
        impurity = float(tree_.impurity[i])
        n_samples = int(tree_.n_node_samples[i])

        # Value array shape: (1, n_classes)
        class_counts = tree_.value[i][0].tolist()
        pred_class_idx = int(np.argmax(class_counts))
        pred_class = classes[pred_class_idx]

        nodes.append({
            "node_id": i,
            "is_leaf": is_leaf,
            "feature": feat_name,
            "threshold": round(threshold, 3) if threshold is not None else None,
            "impurity": round(impurity, 4),
            "n_samples": n_samples,
            "predicted_class": pred_class,
            "class_distribution": dict(zip(classes, [int(c) for c in class_counts])),
            "left_child": int(tree_.children_left[i]),
            "right_child": int(tree_.children_right[i]),
        })

    return {
        "n_nodes": int(tree_.node_count),
        "max_depth": int(tree_.max_depth),
        "n_leaves": int(tree_.n_leaves),
        "criterion": bundle["config"]["criterion"],
        "nodes": nodes
    }


def get_decision_tree_concepts() -> Dict[str, Dict[str, str]]:
    """Educational reference material explaining Decision Tree theory."""
    return {
        "Decision Tree Learning": {
            "title": "Decision Tree Learning",
            "body": (
                "A non-parametric supervised learning algorithm that partitions feature space into "
                "axis-aligned hyper-rectangles using hierarchical boolean decision rules. Each internal node "
                "tests an attribute, each branch represents an outcome of the test, and each leaf node "
                "holds a class label representing the majority vote of student observations in that subspace."
            )
        },
        "Gini Impurity": {
            "title": "Gini Impurity",
            "body": (
                "Measures the probability of a randomly chosen student element being incorrectly labeled if it "
                "were randomly labeled according to the class distribution in the node. "
                "Formula: I_G(p) = 1 - Σ (p_i)². A Gini of 0.0 indicates a perfectly pure leaf where all students "
                "share the identical performance classification."
            )
        },
        "Entropy": {
            "title": "Entropy & Information Theory",
            "body": (
                "Quantifies the average level of information, surprise, or uncertainty in a node's class distribution. "
                "Formula: H(S) = - Σ p_i * log₂(p_i). When student records are evenly distributed across categories, "
                "entropy is maximized; when all students belong to one category, entropy is zero."
            )
        },
        "Information Gain": {
            "title": "Information Gain",
            "body": (
                "The expected reduction in entropy resulting from splitting a node on an academic attribute A: "
                "IG(S, A) = H(S) - Σ (|S_v| / |S|) * H(S_v). The Decision Tree algorithm greedily selects the feature "
                "and numerical threshold that maximizes Information Gain at each branch."
            )
        },
        "Classification Mechanics": {
            "title": "Hierarchical Classification",
            "body": (
                "To classify a student, the algorithm evaluates conditions from the root node downwards "
                "(e.g., 'study_hours <= 4.5'). The student profile follows matching branches until reaching a terminal "
                "leaf node. The leaf provides both the categorical prediction and empirical class probabilities."
            )
        }
    }


def get_feature_importance(bundle: Dict[str, Any], use_original: bool = True) -> pd.DataFrame:
    """Return sorted feature importance DataFrame from trained Decision Tree model."""
    if use_original and "feature_importances_original" in bundle:
        return bundle["feature_importances_original"].copy()
    elif "feature_importances_transformed" in bundle:
        return bundle["feature_importances_transformed"].copy()
    return pd.DataFrame()


def get_tree_rules_text(bundle: Dict[str, Any], max_depth: int = 4) -> str:
    """Return text export of the decision tree rules."""
    model = bundle.get("model")
    feature_names = bundle.get("transformed_feature_names")
    if model is not None and feature_names is not None:
        try:
            return export_text(model, feature_names=feature_names, max_depth=max_depth)
        except Exception:
            pass
    return bundle.get("tree_rules_text", "Rules not available.")


def export_tree_plot_data(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Return objects required to plot tree using sklearn.tree.plot_tree."""
    return {
        "model": bundle.get("model"),
        "feature_names": bundle.get("transformed_feature_names"),
        "class_names": bundle.get("class_names")
    }

