"""
Decision Tree Machine Learning Engine
Student Performance Prediction and Analysis System
Full preprocessing, Decision Tree Classifier, evaluation metrics, and inference pipeline.
"""

import os
import time
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

MODEL_STORAGE_FILE = "decision_tree_model.pkl"

DEFAULT_SCORE_THRESHOLDS = {
    "Excellent": 80.0,
    "Good": 65.0,
    "Average": 50.0,
}


def categorize_numeric_scores(
    series: pd.Series,
    thresholds: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Transform continuous numerical grades into documented student performance categories.
    Default thresholds:
      - >= 80.0: Excellent
      - >= 65.0 and < 80.0: Good
      - >= 50.0 and < 65.0: Average
      - < 50.0: Needs Improvement
    """
    thresh = thresholds or DEFAULT_SCORE_THRESHOLDS
    t_exc = float(thresh.get("Excellent", 80.0))
    t_good = float(thresh.get("Good", 65.0))
    t_avg = float(thresh.get("Average", 50.0))

    def _map_val(v):
        if pd.isna(v):
            return "Needs Improvement"
        try:
            val = float(v)
        except (ValueError, TypeError):
            return "Needs Improvement"

        if val >= t_exc:
            return "Excellent"
        elif val >= t_good:
            return "Good"
        elif val >= t_avg:
            return "Average"
        else:
            return "Needs Improvement"

    return series.apply(_map_val)


def inspect_candidate_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze all columns in a dataset to identify candidate targets and features.
    """
    candidates = {
        "all_columns": list(df.columns),
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
        "recommended_targets": [],
        "id_like_columns": [],
    }

    for col in df.columns:
        col_lower = str(col).lower()
        # ID-like columns (high cardinality or naming convention)
        if "id" in col_lower or "name" in col_lower or "roll" in col_lower or "index" in col_lower:
            candidates["id_like_columns"].append(col)

        # Performance or target candidates
        if any(keyword in col_lower for keyword in ["score", "grade", "performance", "result", "target", "category", "pass", "outcome"]):
            candidates["recommended_targets"].append(col)

    if not candidates["recommended_targets"]:
        # Fallback: last column
        candidates["recommended_targets"] = [df.columns[-1]] if len(df.columns) > 0 else []

    return candidates


def train_decision_tree_pipeline(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    is_numeric_target: bool = False,
    score_thresholds: Optional[Dict[str, float]] = None,
    criterion: str = "gini",
    max_depth: Optional[int] = 6,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Train a Decision Tree Classifier using Scikit-Learn Pipeline and ColumnTransformer.
    - Fits preprocessing exclusively on training split (no data leakage).
    - Uses stratified split when target class frequencies permit.
    - Encodes categorical variables and imputes missing values.
    - Computes multiclass accuracy, precision, recall, f1, and confusion matrix.
    """
    start_time = time.time()

    if df is None or df.empty:
        return False, "Cannot train model: the active dataset is empty.", None

    if target_column not in df.columns:
        return False, f"Target column '{target_column}' does not exist in dataset.", None

    if not feature_columns:
        return False, "At least one predictive feature must be selected for training.", None

    working_df = df.copy()

    # Target processing
    if is_numeric_target:
        target_series = categorize_numeric_scores(working_df[target_column], score_thresholds)
    else:
        target_series = working_df[target_column].astype(str).str.strip()

    # Drop missing target rows
    valid_mask = target_series.notna() & (target_series != "")
    if valid_mask.sum() < 10:
        return False, "Insufficient non-null target samples for model training (fewer than 10).", None

    X = working_df.loc[valid_mask, feature_columns]
    y = target_series[valid_mask]

    # Validate classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        return False, f"Target contains only 1 unique class ({unique_classes[0]}). At least 2 distinct classes required.", None

    # Determine numerical vs categorical among selected features
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_columns if c not in num_cols]

    # Train-test split (stratified if each class has >= 2 instances)
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2
    stratify_target = y if can_stratify else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target
        )
    except Exception as split_err:
        # Fallback to non-stratified split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    # Build Preprocessing Pipeline
    transformers = []
    if num_cols:
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", dt_classifier)
    ])

    # Fit pipeline strictly on X_train (Zero Data Leakage)
    full_pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_pred = full_pipeline.predict(X_test)

    # Evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    classes_list = list(full_pipeline.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes_list)
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extract feature names and feature importances
    try:
        raw_feature_names = preprocessor.get_feature_names_out()
        clean_feature_names = [f.split("__")[-1] for f in raw_feature_names]
    except Exception:
        clean_feature_names = [f"feat_{i}" for i in range(len(dt_classifier.feature_importances_))]

    importances = dt_classifier.feature_importances_
    total_imp = importances.sum()
    pct_imp = (importances / total_imp * 100) if total_imp > 0 else np.zeros_like(importances)

    imp_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Importance": importances,
        "Percentage": pct_imp,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Tree architecture attributes
    tree_depth = int(dt_classifier.get_depth())
    tree_node_count = int(dt_classifier.tree_.node_count)
    tree_n_leaves = int(dt_classifier.get_n_leaves())

    training_time = round(time.time() - start_time, 3)

    result_payload = {
        "pipeline": full_pipeline,
        "classifier": dt_classifier,
        "preprocessor": preprocessor,
        "classes": classes_list,
        "feature_columns": feature_columns,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "clean_feature_names": clean_feature_names,
        "feature_importance_df": imp_df,
        "target_column": target_column,
        "is_numeric_target": is_numeric_target,
        "score_thresholds": score_thresholds or DEFAULT_SCORE_THRESHOLDS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X),
        "stratified_split": can_stratify,
        "accuracy": acc,
        "precision_weighted": prec_weighted,
        "precision_macro": prec_macro,
        "recall_weighted": rec_weighted,
        "recall_macro": rec_macro,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "y_test_actual": list(y_test),
        "y_test_pred": list(y_pred),
        "tree_depth": tree_depth,
        "tree_node_count": tree_node_count,
        "tree_n_leaves": tree_n_leaves,
        "hyperparameters": {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "test_size": test_size,
        },
        "training_time_sec": training_time,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save to disk for persistence
    try:
        with open(MODEL_STORAGE_FILE, "wb") as f:
            pickle.dump(result_payload, f)
    except Exception:
        pass

    return True, f"Decision Tree Classifier successfully trained on {len(X_train)} samples ({acc*100:.1f}% test accuracy).", result_payload


def load_saved_decision_tree() -> Optional[Dict[str, Any]]:
    """Load previously trained Decision Tree model payload from disk."""
    if os.path.exists(MODEL_STORAGE_FILE):
        try:
            with open(MODEL_STORAGE_FILE, "rb") as f:
                payload = pickle.load(f)
                if isinstance(payload, dict) and "pipeline" in payload:
                    return payload
        except Exception:
            pass
    return None


def predict_single_student(
    pipeline: Pipeline,
    feature_columns: List[str],
    input_values: Dict[str, Any]
) -> Tuple[str, Dict[str, float]]:
    """
    Predict performance category and multiclass probability distribution.
    Uses exact same feature ordering and ColumnTransformer preprocessing.
    """
    row_data = {col: input_values.get(col, 0.0) for col in feature_columns}
    input_df = pd.DataFrame([row_data], columns=feature_columns)

    pred_category = str(pipeline.predict(input_df)[0])

    # Probability distribution
    probs = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            prob_arr = pipeline.predict_proba(input_df)[0]
            for cls_name, p in zip(pipeline.classes_, prob_arr):
                probs[cls_name] = round(float(p) * 100, 1)
        except Exception:
            probs = {pred_category: 100.0}
    else:
        probs = {pred_category: 100.0}

    return pred_category, probs


def evaluate_academic_risk_tier(
    predicted_category: str,
    input_values: Dict[str, Any]
) -> Tuple[str, str, str, str]:
    """
    Transparent, documented academic risk classification rules based on
    predicted performance class and key academic indicators.
    Returns (tier_label, badge_class, hex_color, rule_explanation).
    """
    cat = str(predicted_category).lower()
    attendance = float(input_values.get("attendance", 100.0))
    study_hours = float(input_values.get("study_hours", 5.0))

    # Rule 1: Needs Improvement -> Always High Risk
    if "needs" in cat or "fail" in cat or "poor" in cat or "below" in cat:
        return (
            "High Academic Risk",
            "badge-risk-high",
            "#EF4444",
            "Rule Applied: Model predicts 'Needs Improvement'. Immediate faculty remediation and academic intervention required."
        )

    # Rule 2: Average -> High Risk if attendance < 70% or study < 3 hrs; else Medium Risk
    elif "average" in cat or "moderate" in cat or "pass" in cat:
        if attendance < 70.0:
            return (
                "High Academic Risk",
                "badge-risk-high",
                "#EF4444",
                f"Rule Applied: Predicted 'Average' compounded by critical attendance lag ({attendance:.1f}% < 70%). Elevates to High Risk."
            )
        elif study_hours < 3.0:
            return (
                "High Academic Risk",
                "badge-risk-high",
                "#EF4444",
                f"Rule Applied: Predicted 'Average' compounded by severe study deficit ({study_hours:.1f} hrs/day < 3.0 hrs). Elevates to High Risk."
            )
        else:
            return (
                "Medium Academic Risk",
                "badge-risk-medium",
                "#F59E0B",
                "Rule Applied: Predicted 'Average' with stable attendance. Student requires continuous monitoring to prevent grade dips."
            )

    # Rule 3: Good -> Medium Risk if attendance < 65%; else Low Risk
    elif "good" in cat or "above" in cat:
        if attendance < 65.0:
            return (
                "Medium Academic Risk",
                "badge-risk-medium",
                "#F59E0B",
                f"Rule Applied: Predicted 'Good' but attendance is marginal ({attendance:.1f}% < 65%). Risk tier adjusted to Medium."
            )
        else:
            return (
                "Low Academic Risk",
                "badge-risk-low",
                "#10B981",
                "Rule Applied: Predicted 'Good' with healthy academic indicators. Minimal probability of failure."
            )

    # Rule 4: Excellent -> Low Academic Risk
    elif "excellent" in cat or "distinction" in cat or "high" in cat:
        return (
            "Low Academic Risk",
            "badge-risk-low",
            "#10B981",
            "Rule Applied: Model predicts 'Excellent'. Student demonstrates robust foundational mastery and optimal trajectory."
        )

    # Fallback Rule
    else:
        return (
            "Medium Academic Risk",
            "badge-risk-medium",
            "#F59E0B",
            f"Rule Applied: Standard risk profile for category '{predicted_category}'."
        )


def simulate_what_if_trajectory(
    pipeline: Pipeline,
    feature_columns: List[str],
    base_inputs: Dict[str, Any],
    param_name: str,
    param_range: List[float]
) -> pd.DataFrame:
    """
    Simulate predicted performance categories and class probabilities across a numerical parameter range.
    """
    records = []
    classes = list(pipeline.classes_)

    for val in param_range:
        current_row = base_inputs.copy()
        current_row[param_name] = val

        pred_cat, probs = predict_single_student(pipeline, feature_columns, current_row)
        tier, _, _, _ = evaluate_academic_risk_tier(pred_cat, current_row)

        record = {
            param_name: val,
            "Predicted_Category": pred_cat,
            "Risk_Tier": tier,
        }
        for cls in classes:
            record[f"Prob_{cls}"] = probs.get(cls, 0.0)

        records.append(record)

    return pd.DataFrame(records)

