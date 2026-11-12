"""
Theme and Styling Module
Student Performance Prediction and Analysis System
Clean White + Blue/Black Theme
"""

import streamlit as st


def apply_theme():
    """Inject clean white and blue/black academic analytics CSS theme."""
    st.markdown("""
    <style>
    /* =========================================================
       GLOBAL RESET & THEME TOKENS
       ========================================================= */
    :root {
        --bg-main: #FFFFFF;
        --bg-surface: #F8F9FA;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F1F5F9;
        --border-subtle: #E2E8F0;
        --border-focus: #2563EB;
        --primary-blue: #2563EB;
        --primary-blue-light: #3B82F6;
        --accent-violet: #2563EB;
        --accent-violet-light: #3B82F6;
        --text-white: #1E293B;
        --text-primary: #334155;
        --text-secondary: #64748B;
        --text-muted: #94A3B8;
        --risk-low: #16A34A;
        --risk-medium: #D97706;
        --risk-high: #DC2626;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* =========================================================
       SIDEBAR STYLING
       ========================================================= */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    .sidebar-brand-box {
        padding: 16px 12px;
        background: linear-gradient(180deg, rgba(37, 99, 235, 0.08) 0%, transparent 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        margin-bottom: 12px;
    }

    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        letter-spacing: -0.01em;
        line-height: 1.3;
        margin: 0;
    }

    .sidebar-brand-subtitle {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .sidebar-user-card {
        padding: 10px 12px;
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .sidebar-nav-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-weight: 700;
        margin-top: 14px;
        margin-bottom: 6px;
        padding-left: 4px;
    }

    /* =========================================================
       TYPOGRAPHY & HEADINGS
       ========================================================= */
    h1, h2, h3, h4, h5, h6 {
        color: #1E293B !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, label, li {
        color: var(--text-primary);
    }

    .page-header-container {
        margin-bottom: 1.6rem;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 1rem;
    }

    .page-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E293B;
        margin: 0;
        line-height: 1.2;
    }

    .page-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* =========================================================
       CARDS & CONTAINERS
       ========================================================= */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 18px 20px;
        transition: border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        border-color: var(--primary-blue);
    }

    .metric-card-top-accent {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-blue);
    }

    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E293B;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .metric-subtext {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 6px;
    }

    .standard-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    /* =========================================================
       ACTIVE DATASET BANNER
       ========================================================= */
    .active-dataset-banner {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .active-dataset-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .active-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #16A34A;
        display: inline-block;
    }

    /* =========================================================
       BADGES & STATUS PILLS
       ========================================================= */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-blue {
        background-color: rgba(37, 99, 235, 0.1);
        color: #2563EB;
        border: 1px solid rgba(37, 99, 235, 0.25);
    }

    .badge-violet {
        background-color: rgba(37, 99, 235, 0.1);
        color: #1D4ED8;
        border: 1px solid rgba(37, 99, 235, 0.25);
    }

    .badge-risk-low {
        background-color: rgba(22, 163, 74, 0.1);
        color: #16A34A;
        border: 1px solid rgba(22, 163, 74, 0.25);
    }

    .badge-risk-medium {
        background-color: rgba(217, 119, 6, 0.1);
        color: #D97706;
        border: 1px solid rgba(217, 119, 6, 0.25);
    }

    .badge-risk-high {
        background-color: rgba(220, 38, 38, 0.1);
        color: #DC2626;
        border: 1px solid rgba(220, 38, 38, 0.25);
    }

    /* =========================================================
       FORM INPUTS & STREAMLIT OVERRIDES
       ========================================================= */
    .stTextInput input, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }

    /* Primary Buttons */
    .stButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        transition: background 0.15s ease !important;
    }

    .stButton > button:hover {
        background: #1D4ED8 !important;
    }

    .stButton > button:active {
        background: #1E40AF !important;
    }

    /* Tables & Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #2563EB !important;
    }

    /* File uploader box */
    [data-testid="stFileUploader"] {
        background-color: #F8F9FA;
        border: 2px dashed rgba(37, 99, 235, 0.3);
        border-radius: 10px;
        padding: 18px;
    }

    /* App Footer */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-subtle);
        color: var(--text-muted);
        font-size: 0.82rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge: str = "Academic Analytics"):
    """Render a consistent page header with subtitle and badge."""
    st.markdown(f"""
    <div class="page-header-container">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
            <span class="badge badge-blue">{badge}</span>
        </div>
        <h1 class="page-title">{title}</h1>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_active_dataset_badge(session_state):
    """Render a clean active dataset status banner at the top of analytical pages."""
    name = getattr(session_state, "active_dataset_name", "None")
    dtype = getattr(session_state, "active_dataset_type", "None")
    df = getattr(session_state, "active_df", None)

    if df is not None:
        rows, cols = df.shape
        badge_style = "badge-blue" if dtype == "Sample" else "badge-violet"
        st.markdown(f"""
        <div class="active-dataset-banner">
            <div class="active-dataset-info">
                <span class="active-dot"></span>
                <div>
                    <span style="font-size:0.85rem; font-weight:700; color:#1E293B;">Active Dataset:</span>
                    <span style="font-size:0.85rem; color:#475569; margin-left:6px; font-family:monospace;">{name}</span>
                </div>
            </div>
            <div style="display:flex; gap:8px; align-items:center;">
                <span class="badge {badge_style}">{dtype} Mode</span>
                <span class="badge badge-blue">{rows:,} Rows &times; {cols} Columns</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="active-dataset-banner" style="border-color: rgba(220, 38, 38, 0.3); background: rgba(220, 38, 38, 0.04);">
            <div class="active-dataset-info">
                <span class="active-dot" style="background-color:#DC2626;"></span>
                <span style="font-size:0.85rem; font-weight:600; color:#DC2626;">No Dataset Currently Loaded</span>
            </div>
            <span class="badge badge-risk-high">Action Required in Dataset Management</span>
        </div>
        """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtext: str = "", top_accent: bool = True):
    """Render an executive dashboard metric card."""
    accent_html = '<div class="metric-card-top-accent"></div>' if top_accent else ''
    st.markdown(f"""
    <div class="metric-card">
        {accent_html}
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)
