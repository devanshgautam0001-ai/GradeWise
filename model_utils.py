"""
Model Utilities and Machine Learning Inference Module
Student Performance Prediction and Analysis System
"""

import os
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_FILE = "best_model.pkl"
DEFAULT_DATASET_FILE = "advanced_student_data.csv"

DEFAULT_FEATURES = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_score",
    "social_media_usage",
    "stress_level",
    "preparation_level",
    "extra_activities",
]
DEFAULT_TARGET = "final_score"

FEATURE_METADATA = {
    "study_hours": {"label": "Daily Study Hours", "min": 0.0, "max": 14.0, "default": 5.0, "step": 0.5, "unit": "hrs/day"},
    "attendance": {"label": "Attendance Rate", "min": 0.0, "max": 100.0, "default": 75.0, "step": 1.0, "unit": "%"},
    "sleep_hours": {"label": "Daily Sleep Duration", "min": 3.0, "max": 12.0, "default": 7.0, "step": 0.5, "unit": "hrs/day"},
    "previous_score": {"label": "Previous Term Score", "min": 0.0, "max": 100.0, "default": 65.0, "step": 1.0, "unit": "marks"},
    "social_media_usage": {"label": "Daily Social Media Usage", "min": 0.0, "max": 12.0, "default": 2.5, "step": 0.5, "unit": "hrs/day"},
    "stress_level": {"label": "Perceived Stress Scale", "min": 1, "max": 10, "default": 5, "step": 1, "unit": "1-10 scale"},
    "preparation_level": {"label": "Exam Preparation Level", "min": 1, "max": 5, "default": 3, "step": 1, "unit": "1-5 scale"},
    "extra_activities": {"label": "Extracurricular Activities", "min": 0, "max": 25, "default": 5, "step": 1, "unit": "hrs/week"},
}


def train_baseline_model(
    df: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    target: str = DEFAULT_TARGET
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Train a Random Forest Regressor on the dataset.
    Returns (model, feature_names, metrics_dict).
    """
    if df is None:
        df = pd.read_csv(DEFAULT_DATASET_FILE)

    if features is None:
        features = [col for col in DEFAULT_FEATURES if col in df.columns]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, test_pred))
    mse = float(mean_squared_error(y_test, test_pred))
    rmse = float(np.sqrt(mse))
    r2_test = float(r2_score(y_test, test_pred))
    r2_train = float(r2_score(y_train, train_pred))

    metrics = {
        "mae": round(mae, 3),
        "mse": round(mse, 3),
        "rmse": round(rmse, 3),
        "r2_test": round(r2_test, 4),
        "r2_train": round(r2_train, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(features),
        "features": features,
    }

    # Save to disk for fast reuse
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, features, metrics), f)
    except Exception:
        pass

    return model, features, metrics


def load_model_and_metrics() -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Load pre-trained model and evaluation metrics, or train if missing.
    """
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, tuple) and len(loaded) >= 2:
                model, features = loaded[0], loaded[1]
                metrics = loaded[2] if len(loaded) > 2 and isinstance(loaded[2], dict) else {}
                if not metrics:
                    # Compute metrics quickly
                    _, _, metrics = train_baseline_model()
                return model, features, metrics
        except Exception:
            pass

    return train_baseline_model()


def classify_academic_risk(score: float) -> Tuple[str, str, str, str]:
    """
    Classify academic performance risk tier.
    Returns (tier_label, badge_class, hex_color, detailed_description).
    """
    clamped_score = max(0.0, min(100.0, float(score)))

    if clamped_score >= 75.0:
        return (
            "Low Academic Risk",
            "risk-low",
            "#10B981",
            "Optimal academic trajectory. Student demonstrates strong foundational readiness with minimal likelihood of course failure."
        )
    elif clamped_score >= 50.0:
        return (
            "Moderate Academic Risk",
            "risk-medium",
            "#F59E0B",
            "Borderline academic trajectory. Student is vulnerable to grade dips without proactive study habit reinforcement and attendance monitoring."
        )
    else:
        return (
            "High Academic Risk",
            "risk-high",
            "#EF4444",
            "Critical academic risk. Urgent academic intervention required: mandatory faculty tutoring, attendance recovery, and structured study regimen."
        )


def predict_student_score(
    model: Any,
    features: List[str],
    input_values: Dict[str, float]
) -> Tuple[float, str, str, str, str]:
    """
    Predict student score and return risk profile.
    Returns (score, tier_label, badge_class, hex_color, description).
    """
    ordered_values = [input_values.get(f, 0.0) for f in features]
    input_df = pd.DataFrame([ordered_values], columns=features)
    pred_raw = float(model.predict(input_df)[0])
    clamped_score = round(max(0.0, min(100.0, pred_raw)), 2)

    tier, badge, color, desc = classify_academic_risk(clamped_score)
    return clamped_score, tier, badge, color, desc


def get_feature_importance_df(model: Any, features: List[str]) -> pd.DataFrame:
    """Return sorted feature importance DataFrame."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({
            "Feature": [FEATURE_METADATA.get(f, {}).get("label", f) for f in features],
            "Raw_Feature": features,
            "Importance": importances,
            "Percentage": (importances / importances.sum()) * 100,
        }).sort_values(by="Importance", ascending=False)
        return df.reset_index(drop=True)
    return pd.DataFrame()


def simulate_what_if_parameter(
    model: Any,
    features: List[str],
    baseline_inputs: Dict[str, float],
    target_param: str,
    param_range: List[float]
) -> pd.DataFrame:
    """
    Generate sensitivity curve data by altering one parameter across a range.
    """
    records = []
    base_row = {f: baseline_inputs.get(f, 0.0) for f in features}

    for val in param_range:
        current_row = base_row.copy()
        current_row[target_param] = val
        input_df = pd.DataFrame([[current_row[f] for f in features]], columns=features)
        pred = round(float(model.predict(input_df)[0]), 2)
        clamped = max(0.0, min(100.0, pred))
        tier, _, _, _ = classify_academic_risk(clamped)
        records.append({
            target_param: val,
            "Predicted_Score": clamped,
            "Risk_Tier": tier
        })

    return pd.DataFrame(records)

