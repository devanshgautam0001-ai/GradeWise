"""
Authentication and SQLite Database Module
Student Performance Prediction and Analysis System
PBKDF2 Password Hashing, Session Management, and User Prediction History.
"""

import sqlite3
import hashlib
import os
import hmac
import json
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd

DB_NAME = "users.db"
PBKDF2_ITERATIONS = 100000


def _hash_password(password: str) -> str:
    """
    Generate a secure PBKDF2-HMAC-SHA256 password hash with a random salt.
    Format: pbkdf2_sha256$iterations$salt$hash
    """
    salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${key.hex()}"


def _verify_password(stored_hash: str, password: str) -> Tuple[bool, bool]:
    """
    Verify password against stored hash.
    Returns (is_valid, needs_upgrade).
    Supports backward compatibility with legacy plain SHA-256 hashes.
    """
    if not stored_hash or not password:
        return False, False

    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            parts = stored_hash.split("$")
            if len(parts) != 4:
                return False, False
            iterations = int(parts[1])
            salt = parts[2]
            expected_key = parts[3]
            derived_key = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                iterations
            )
            is_valid = hmac.compare_digest(derived_key.hex(), expected_key)
            return is_valid, False
        except Exception:
            return False, False
    else:
        # Legacy SHA256 hex string check
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        is_valid = hmac.compare_digest(stored_hash, legacy_hash)
        return is_valid, is_valid


def init_db():
    """Initialize database tables and run schema migrations safely."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT ''
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            study_hours REAL DEFAULT 0.0,
            attendance REAL DEFAULT 0.0,
            predicted_score REAL DEFAULT 0.0,
            risk TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            performance_category TEXT DEFAULT '',
            inputs_json TEXT DEFAULT '{}',
            model_config_json TEXT DEFAULT '{}'
        )
    """)

    conn.commit()

    # Column migrations for existing databases
    existing_cols = [c[1] for c in cur.execute("PRAGMA table_info(predictions);").fetchall()]
    new_cols = [
        ("performance_category", "TEXT DEFAULT ''"),
        ("inputs_json", "TEXT DEFAULT '{}'"),
        ("model_config_json", "TEXT DEFAULT '{}'"),
        ("created_at", "TEXT DEFAULT ''"),
    ]

    for col_name, col_def in new_cols:
        if col_name not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_def}")
                conn.commit()
            except Exception:
                pass

    conn.close()


def create_user(username: str, password: str) -> Tuple[bool, str]:
    """Register a new user with PBKDF2 hashed password."""
    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        secure_hash = _hash_password(password)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, secure_hash, now_str),
        )
        conn.commit()
        conn.close()
        return True, "Account registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose a different username."
    except Exception as e:
        return False, f"Registration error: {str(e)}"


def verify_user(username: str, password: str) -> bool:
    """
    Authenticate user. If an old plain SHA-256 hash matches, seamlessly
    rehash and upgrade the record to PBKDF2-HMAC-SHA256 with salt.
    """
    username = username.strip()
    if not username or not password:
        return False

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return False

    user_id, stored_hash = row[0], row[1]
    is_valid, needs_upgrade = _verify_password(stored_hash, password)

    if is_valid and needs_upgrade:
        new_hash = _hash_password(password)
        cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        conn.commit()

    conn.close()
    return is_valid


def save_prediction(
    username: str,
    study_hours: float = 0.0,
    attendance: float = 0.0,
    score: float = 0.0,
    risk: str = "",
    performance_category: str = "",
    inputs_dict: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None
):
    """
    Save student prediction audit record into SQLite.
    Stores full input payload and model hyperparameters as JSON.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inputs_json = json.dumps(inputs_dict or {})
    config_json = json.dumps(model_config or {})

    # If performance_category not provided, extract from risk if present
    if not performance_category and " (" in risk:
        performance_category = risk.split(" (")[0]

    cur.execute("""
        INSERT INTO predictions (
            username, study_hours, attendance, predicted_score, risk,
            created_at, performance_category, inputs_json, model_config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username, study_hours, attendance, score, risk,
        now_str, performance_category, inputs_json, config_json
    ))

    conn.commit()
    conn.close()


def get_user_history(
    username: str,
    search_query: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve prediction history strictly scoped to the authenticated user.
    Supports optional text search across performance category, risk tier, or timestamp.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    if search_query and search_query.strip():
        q = f"%{search_query.strip().lower()}%"
        cur.execute("""
            SELECT id, username, study_hours, attendance, predicted_score, risk,
                   created_at, performance_category, inputs_json, model_config_json
            FROM predictions
            WHERE username = ?
              AND (LOWER(performance_category) LIKE ? OR LOWER(risk) LIKE ? OR created_at LIKE ?)
            ORDER BY id DESC
            LIMIT ?
        """, (username, q, q, q, limit))
    else:
        cur.execute("""
            SELECT id, username, study_hours, attendance, predicted_score, risk,
                   created_at, performance_category, inputs_json, model_config_json
            FROM predictions
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ?
        """, (username, limit))

    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        try:
            inputs = json.loads(r[8]) if r[8] else {}
        except Exception:
            inputs = {}

        try:
            config = json.loads(r[9]) if r[9] else {}
        except Exception:
            config = {}

        cat = r[7] if r[7] else (r[5].split(" (")[0] if " (" in r[5] else r[5])

        results.append({
            "id": r[0],
            "username": r[1],
            "study_hours": r[2],
            "attendance": r[3],
            "predicted_score": r[4],
            "risk": r[5],
            "created_at": r[6] if r[6] else "Earlier",
            "performance_category": cat,
            "inputs": inputs,
            "model_config": config,
        })

    return results


def delete_user_prediction(record_id: int, username: str) -> bool:
    """
    Delete a specific prediction record belonging to the authenticated user.
    Enforces user isolation via username clause.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions WHERE id = ? AND username = ?", (record_id, username))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted
    except Exception:
        return False


def clear_user_history(username: str) -> bool:
    """Clear all prediction records for the authenticated user."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def get_user_stats(username: str) -> Dict[str, Any]:
    """Compute aggregate prediction statistics for user dashboard."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*), AVG(study_hours), AVG(attendance)
        FROM predictions
        WHERE username = ?
    """, (username,))
    row = cur.fetchone()

    total = row[0] if row and row[0] is not None else 0
    avg_study = round(row[1], 1) if row and row[1] is not None else 0.0
    avg_att = round(row[2], 1) if row and row[2] is not None else 0.0

    # Get latest prediction category and risk
    cur.execute("""
        SELECT performance_category, risk, created_at
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
        LIMIT 1
    """, (username,))
    latest = cur.fetchone()

    latest_cat = latest[0] if latest and latest[0] else "None"
    latest_risk = latest[1] if latest and latest[1] else "None"
    latest_time = latest[2] if latest and latest[2] else "-"

    conn.close()
    return {
        "total_predictions": total,
        "avg_study_hours": avg_study,
        "avg_attendance": avg_att,
        "latest_category": latest_cat,
        "latest_risk": latest_risk,
        "latest_time": latest_time,
    }


def get_user_trend_data(username: str) -> pd.DataFrame:
    """Return historical prediction trend DataFrame for charting."""
    records = get_user_history(username, limit=50)
    if not records:
        return pd.DataFrame()

    rows = []
    for r in reversed(records):  # Chronological order
        rows.append({
            "Assessment ID": f"#{r['id']}",
            "Date": r["created_at"][:10] if len(r["created_at"]) >= 10 else r["created_at"],
            "Study Hours": r["study_hours"],
            "Attendance": r["attendance"],
            "Category": r["performance_category"],
            "Risk": r["risk"],
        })

    return pd.DataFrame(rows)
