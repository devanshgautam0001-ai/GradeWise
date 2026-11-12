"""
Dataset Management, Ingestion, and Validation Module
Student Performance Prediction and Analysis System
"""

import os
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np

DEFAULT_SAMPLE_PATH = "advanced_student_data.csv"
SECONDARY_SAMPLE_PATH = "student_data.csv"


def load_sample_dataset(filepath: str = DEFAULT_SAMPLE_PATH) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """Load default verified sample dataset from disk."""
    if not os.path.exists(filepath):
        return False, f"Default sample file '{filepath}' not found.", None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return False, f"Sample file '{filepath}' is empty.", None
        return True, "Loaded sample dataset successfully.", df
    except Exception as e:
        return False, f"Failed to load sample dataset: {str(e)}", None


def validate_and_load_file(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Validate and ingest uploaded dataset (CSV or Excel XLSX).
    Implements strict validation:
    - Rejects unsupported extensions
    - Rejects empty files (0 bytes or 0 rows)
    - Gracefully catches corrupted files
    - Identifies column schemas
    """
    metadata: Dict[str, Any] = {
        "filename": getattr(uploaded_file, "name", "uploaded_dataset"),
        "size_bytes": getattr(uploaded_file, "size", 0),
        "extension": "",
        "error_type": None,
    }

    if uploaded_file is None:
        return False, "No file was provided for upload.", None, metadata

    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    metadata["extension"] = ext

    # 1. Format validation
    if ext not in [".csv", ".xlsx", ".xls"]:
        metadata["error_type"] = "unsupported_format"
        return (
            False,
            f"Unsupported file format '{ext}'. Only CSV (.csv) and Excel (.xlsx, .xls) files are accepted.",
            None,
            metadata,
        )

    # 2. File size validation (empty file check)
    if metadata["size_bytes"] == 0:
        metadata["error_type"] = "empty_file"
        return (
            False,
            "The selected file is empty (0 bytes). Please upload a valid dataset containing student records.",
            None,
            metadata,
        )

    # 3. File parsing with corruption handling
    try:
        uploaded_file.seek(0)
        if ext == ".csv":
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

    except pd.errors.EmptyDataError:
        metadata["error_type"] = "empty_data"
        return False, "The file contains no columns or records (EmptyDataError).", None, metadata
    except pd.errors.ParserError as pe:
        metadata["error_type"] = "parser_error"
        return (
            False,
            f"CSV parsing error: The file is corrupted or improperly delimited. Details: {str(pe)}",
            None,
            metadata,
        )
    except Exception as e:
        metadata["error_type"] = "corrupted_file"
        return (
            False,
            f"Could not read the uploaded file. It may be corrupted or encrypted. Error: {str(e)}",
            None,
            metadata,
        )

    # 4. Content sanity check
    if df is None or df.empty:
        metadata["error_type"] = "zero_rows"
        return False, "The dataset was parsed but contains 0 rows of data.", None, metadata

    if len(df.columns) == 0:
        metadata["error_type"] = "zero_columns"
        return False, "The dataset contains no headers or columns.", None, metadata

    # Clean column headers
    df.columns = [str(col).strip() for col in df.columns]

    return True, f"Dataset '{filename}' successfully validated and loaded.", df, metadata


def compute_dataset_metrics(
    df: pd.DataFrame, filename: str, source_type: str = "Uploaded"
) -> Dict[str, Any]:
    """
    Generate comprehensive preview and analytical metrics for any dataset.
    """
    row_count, col_count = df.shape
    columns = list(df.columns)

    # Data types
    dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Numerical vs categorical column detection
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    other_cols = [c for c in columns if c not in num_cols and c not in cat_cols]

    # Missing values
    missing_series = df.isnull().sum()
    missing_counts = missing_series.to_dict()
    missing_percents = {
        col: round((count / row_count) * 100, 2) for col, count in missing_counts.items()
    }
    total_missing_cells = int(missing_series.sum())
    total_cells = row_count * col_count if row_count and col_count else 1
    total_missing_pct = round((total_missing_cells / total_cells) * 100, 2)

    # Duplicate records
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / row_count) * 100, 2) if row_count else 0.0

    # Summary statistics
    num_summary = None
    if num_cols:
        num_summary = df[num_cols].describe().T
        # Round numerical summary for clean display
        num_summary = num_summary.round(3)

    cat_summary = None
    if cat_cols:
        cat_summary = df[cat_cols].describe().T

    # Memory footprint
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = round(memory_bytes / (1024 * 1024), 3)

    # Missing values breakdown table
    missing_df = pd.DataFrame({
        "Column": columns,
        "Data Type": [dtypes_map[c] for c in columns],
        "Missing Count": [missing_counts[c] for c in columns],
        "Missing (%)": [missing_percents[c] for c in columns],
        "Unique Values": [df[c].nunique(dropna=True) for c in columns],
    })

    return {
        "filename": filename,
        "source_type": source_type,
        "row_count": row_count,
        "column_count": col_count,
        "columns": columns,
        "dtypes": dtypes_map,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "other_columns": other_cols,
        "missing_counts": missing_counts,
        "missing_percents": missing_percents,
        "total_missing_cells": total_missing_cells,
        "total_missing_pct": total_missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "first_10_records": df.head(10),
        "numerical_summary": num_summary,
        "categorical_summary": cat_summary,
        "missing_breakdown_df": missing_df,
        "memory_mb": memory_mb,
    }


def init_dataset_state(session_state):
    """Ensure active dataset is initialized in Streamlit session state."""
    if "active_df" not in session_state or session_state.active_df is None:
        ok, msg, sample_df = load_sample_dataset()
        if ok and sample_df is not None:
            session_state.active_df = sample_df
            session_state.active_dataset_name = "advanced_student_data.csv"
            session_state.active_dataset_type = "Sample"
        else:
            session_state.active_df = None
            session_state.active_dataset_name = "No dataset active"
            session_state.active_dataset_type = "None"

    if "dataset_history" not in session_state:
        session_state.dataset_history = []


def set_active_dataset(session_state, df: pd.DataFrame, name: str, dataset_type: str = "Uploaded"):
    """Set the active dataset in session state without modifying original files on disk."""
    session_state.active_df = df.copy()
    session_state.active_dataset_name = name
    session_state.active_dataset_type = dataset_type


def reset_to_sample(session_state, filepath: str = DEFAULT_SAMPLE_PATH) -> Tuple[bool, str]:
    """Revert active dataset to the default verified sample."""
    ok, msg, df = load_sample_dataset(filepath)
    if ok and df is not None:
        set_active_dataset(session_state, df, os.path.basename(filepath), "Sample")
        return True, "Reverted to verified sample dataset."
    return False, msg

