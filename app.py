"""
PredicX | Enterprise Customer Churn & Predictive ML Analytics Platform
Author: Lead Machine Learning Engineer & Senior Streamlit UI/UX Developer
Stack: Streamlit, Scikit-Learn, Plotly, Pandas, NumPy, SQLite
"""

import datetime
import os
import sqlite3
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & EXECUTIVE DARK CSS
# ==============================================================================

st.set_page_config(
    page_title="PredicX | Enterprise Churn & ML Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Global Font & Tone */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #94A3B8;
}

/* Carbon Onyx Base Canvas */
.stApp {
    background: radial-gradient(circle at 18% 12%, #111827 0%, #0B0F19 65%, #05070D 100%);
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0B0F19 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Streamlit Header & Deploy Button Adjustments */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3.2rem !important;
}

.stDeployButton, [data-testid="stToolbar"] {
    visibility: hidden;
    height: 0px;
}

/* Sidebar Toggle Arrow - Prominently visible and glowing */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    background: rgba(30, 41, 59, 0.95) !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
    border-radius: 8px !important;
    color: #818CF8 !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.35) !important;
    padding: 4px !important;
    z-index: 999999 !important;
}

/* Container Padding - Ensures header has breathing room and is not clipped */
.block-container {
    padding-top: 4.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 96% !important;
}

/* Glassmorphism KPI Card */
.kpi-card {
    background: rgba(17, 24, 39, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45), 0 8px 10px -6px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 0.5rem;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 16px 30px -8px rgba(99, 102, 241, 0.18);
}

.kpi-title {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.45rem;
    font-feature-settings: "tnum" 1;
}

.kpi-footer {
    font-size: 0.77rem;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-crimson {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.2);
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-indigo {
    background: rgba(99, 102, 241, 0.15);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

/* Glassmorphic Panel Container */
.glass-panel {
    background: rgba(17, 24, 39, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.35);
}

/* Styled Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.65);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 600;
    font-size: 0.88rem;
    border: none;
    background-color: transparent;
    transition: all 0.2s ease;
    padding: 0 1.2rem;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC;
    background: rgba(255, 255, 255, 0.04);
}

.stTabs [aria-selected="true"] {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
}

/* Smooth Dark Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0F19;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 2. DATA ARCHITECTURE & SQLITE PERSISTENCE ENGINE
# ==============================================================================

DB_PATH = "customer_analytics.db"


def init_database(force_reseed: bool = False) -> None:
    """
    Initializes the SQLite database. If the table 'customers' does not exist,
    contains fewer than 800 records, or force_reseed is True, it generates and
    persists realistic synthetic customer behavioral data.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if force_reseed:
        cursor.execute("DROP TABLE IF EXISTS customers")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            tenure_months INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            contract_type TEXT,
            tech_support_tickets INTEGER,
            payment_method TEXT,
            usage_gb REAL,
            churn INTEGER
        )
    """)

    cursor.execute("SELECT COUNT(*) FROM customers")
    count = cursor.fetchone()[0]

    if count < 800:
        np.random.seed(42)
        n_samples = 1350

        customer_ids = [f"CUST-{10001 + i}" for i in range(n_samples)]

        # Tenure: realistic exponential distribution with long tail (1 to 72 months)
        tenure = np.clip(
            np.random.exponential(scale=22, size=n_samples).astype(int) + 1, 1, 72
        )

        # Contract commitments
        contracts = np.random.choice(
            ["Month-to-month", "1-Year", "2-Year"],
            size=n_samples,
            p=[0.55, 0.25, 0.20],
        )

        # Payment methods
        payments = np.random.choice(
            [
                "Electronic check",
                "Credit card (auto)",
                "Bank transfer (auto)",
                "Mailed check",
            ],
            size=n_samples,
            p=[0.38, 0.26, 0.24, 0.12],
        )

        # Monthly charges ($22 to $118, higher for Month-to-month contracts)
        monthly_charges = np.round(
            np.random.uniform(22.0, 114.0, size=n_samples)
            + (contracts == "Month-to-month") * 8.0,
            2,
        )

        # Total charges ~ tenure * monthly charges with slight variance
        variance_factor = np.random.uniform(0.96, 1.04, size=n_samples)
        total_charges = np.round(tenure * monthly_charges * variance_factor, 2)

        # Technical Support Tickets (Poisson distribution, higher under Month-to-month)
        tickets_lambda = 0.9 + (contracts == "Month-to-month") * 1.4
        tech_tickets = np.clip(
            np.random.poisson(lam=tickets_lambda, size=n_samples), 0, 9
        )

        # Usage in GB: normal distribution centered at 320 GB
        usage_gb = np.round(
            np.clip(np.random.normal(320, 140, size=n_samples), 15.0, 920.0), 1
        )

        # Realistic Churn modeling calibrated to ~26.5% baseline
        logit = (
            -3.05
            + 0.034 * (monthly_charges - 65.0)
            - 0.062 * (tenure - 20.0)
            + 0.54 * tech_tickets
            + 1.35 * (contracts == "Month-to-month")
            - 0.85 * (contracts == "1-Year")
            - 1.85 * (contracts == "2-Year")
            + 0.65 * (payments == "Electronic check")
            - 0.35 * (payments == "Credit card (auto)")
            - 0.0012 * (usage_gb - 300.0)
        )

        churn_prob = 1.0 / (1.0 + np.exp(-logit))
        churn = (np.random.rand(n_samples) < churn_prob).astype(int)

        records = [
            (
                customer_ids[i],
                int(tenure[i]),
                float(monthly_charges[i]),
                float(total_charges[i]),
                str(contracts[i]),
                int(tech_tickets[i]),
                str(payments[i]),
                float(usage_gb[i]),
                int(churn[i]),
            )
            for i in range(n_samples)
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO customers (
                customer_id, tenure_months, monthly_charges, total_charges,
                contract_type, tech_support_tickets, payment_method, usage_gb, churn
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            records,
        )

        conn.commit()

    conn.close()


def query_dataframe(query: str, params: tuple = ()) -> pd.DataFrame:
    """Helper executing SQL queries returning a pandas DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_sql_analytics():
    """
    Executes advanced SQL aggregations for churn rate, CLV approximations,
    and RFM-style customer activity segmentation.
    """
    # 1. Churn Rate by Contract Tier & Payment Method
    query_contract_payment = """
        SELECT 
            contract_type,
            payment_method,
            COUNT(*) AS total_customers,
            SUM(churn) AS churned_count,
            ROUND(AVG(churn) * 100.0, 1) AS churn_rate_pct,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_bill,
            ROUND(SUM(monthly_charges * 12), 2) AS total_arr
        FROM customers
        GROUP BY contract_type, payment_method
        ORDER BY churn_rate_pct DESC
    """
    df_contract_payment = query_dataframe(query_contract_payment)

    # 2. Customer Lifetime Value (CLV) & Tenure Dynamics
    query_clv = """
        SELECT 
            contract_type,
            COUNT(*) AS customer_count,
            ROUND(AVG(tenure_months), 1) AS avg_tenure_months,
            ROUND(AVG(total_charges), 2) AS avg_historical_clv,
            ROUND(AVG(monthly_charges * 12 * (1.0 - churn + 0.15)), 2) AS projected_future_arr
        FROM customers
        GROUP BY contract_type
        ORDER BY avg_historical_clv DESC
    """
    df_clv = query_dataframe(query_clv)

    # 3. RFM-style Behavioral Segmentation Engine
    # R (Recency/Activity approximated by Usage GB), F (Interaction/Tech Tickets), M (Monetary Total Charges)
    query_rfm = """
        WITH customer_scores AS (
            SELECT 
                customer_id,
                tenure_months,
                monthly_charges,
                total_charges,
                usage_gb,
                tech_support_tickets,
                contract_type,
                churn,
                NTILE(3) OVER (ORDER BY usage_gb ASC) AS activity_score,
                NTILE(3) OVER (ORDER BY tech_support_tickets DESC) AS friction_score,
                NTILE(3) OVER (ORDER BY monthly_charges ASC) AS monetary_score
            FROM customers
        )
        SELECT 
            CASE 
                WHEN churn = 0 AND monetary_score = 3 AND activity_score = 3 THEN 'VIP Champions'
                WHEN churn = 0 AND activity_score >= 2 AND friction_score <= 2 THEN 'Stable Core Accounts'
                WHEN churn = 1 THEN 'Churned Accounts'
                WHEN friction_score = 3 AND contract_type = 'Month-to-month' THEN 'High Risk / Severe Friction'
                WHEN activity_score = 1 AND monetary_score <= 2 THEN 'Dormant / Low Engagement'
                ELSE 'Standard Retained'
            END AS rfm_segment,
            COUNT(*) AS count_customers,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
            ROUND(AVG(tech_support_tickets), 1) AS avg_tickets,
            ROUND(AVG(churn) * 100.0, 1) AS segment_churn_pct,
            ROUND(SUM(monthly_charges * 12), 2) AS segment_arr
        FROM customer_scores
        GROUP BY rfm_segment
        ORDER BY segment_arr DESC
    """
    df_rfm = query_dataframe(query_rfm)

    return df_contract_payment, df_clv, df_rfm


# ==============================================================================
# 3. MACHINE LEARNING ENGINE (TRAINING & INFERENCE CACHE)
# ==============================================================================

@st.cache_resource(show_spinner="Training Machine Learning Pipeline...")
def train_churn_model():
    """
    Trains an end-to-end Scikit-Learn RandomForestClassifier pipeline
    with ColumnTransformer preprocessing. Generates metrics, feature importance,
    and returns full customer database predictions.
    """
    df = query_dataframe("SELECT * FROM customers")

    features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "tech_support_tickets",
        "payment_method",
        "usage_gb",
    ]
    target = "churn"

    X = df[features]
    y = df[target]

    num_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "tech_support_tickets",
        "usage_gb",
    ]
    cat_cols = ["contract_type", "payment_method"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                cat_cols,
            ),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # Test set predictions & probabilities
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    # Feature Importances extraction
    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    rf_fitted = pipeline.named_steps["classifier"]

    feature_names_raw = preprocessor_fitted.get_feature_names_out()
    feature_names_clean = []
    for name in feature_names_raw:
        clean = (
            name.replace("num__", "")
            .replace("cat__contract_type_", "Contract: ")
            .replace("cat__payment_method_", "Payment: ")
            .replace("tenure_months", "Tenure (Months)")
            .replace("monthly_charges", "Monthly Bill ($)")
            .replace("total_charges", "Total Charges ($)")
            .replace("tech_support_tickets", "Support Tickets")
            .replace("usage_gb", "Data Usage (GB)")
        )
        feature_names_clean.append(clean)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names_clean,
            "importance": rf_fitted.feature_importances_,
        }
    ).sort_values(by="importance", ascending=True)

    # Score full database for Intervention Queue
    all_prob = pipeline.predict_proba(X)[:, 1]
    df_scored = df.copy()
    df_scored["churn_prob"] = np.round(all_prob, 3)
    df_scored["churn_prob_pct"] = np.round(all_prob * 100.0, 1)

    def assign_risk_tier(p):
        if p >= 0.70:
            return "Critical Risk (>70%)"
        elif p >= 0.40:
            return "Elevated Risk (40-70%)"
        else:
            return "Low Risk (<40%)"

    df_scored["risk_tier"] = df_scored["churn_prob"].apply(assign_risk_tier)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "importance_df": importance_df,
        "df_scored": df_scored,
        "features": features,
    }


# ==============================================================================
# 4. PLOTLY THEME CONFIGURATION
# ==============================================================================

def apply_executive_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Standardizes plotly figures with refined dark glassmorphism styling."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=13, color="#F8FAFC", family="Plus Jakarta Sans"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        font=dict(color="#94A3B8", family="Plus Jakarta Sans", size=11),
        margin=dict(l=20, r=20, t=45, b=25),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Plus Jakarta Sans",
            bordercolor="#475569",
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#CBD5E1"),
        ),
    )
    return fig


# ==============================================================================
# 5. INITIALIZATION & DATA LOADING
# ==============================================================================

init_database()
ml_bundle = train_churn_model()
pipeline = ml_bundle["pipeline"]
metrics = ml_bundle["metrics"]
importance_df = ml_bundle["importance_df"]
df_scored = ml_bundle["df_scored"]

df_contract_payment, df_clv, df_rfm = get_sql_analytics()

# Global KPI Calculations
total_customers = len(df_scored)
churned_customers = df_scored["churn"].sum()
global_churn_rate = (churned_customers / total_customers) * 100.0

# Total Annual Revenue at Risk (ARR): Sum of ARR (monthly_charges * 12) for accounts with churn_prob >= 0.70 or already churned
critical_accounts_df = df_scored[
    (df_scored["churn_prob"] >= 0.70) | (df_scored["churn"] == 1)
]
total_arr_at_risk = (critical_accounts_df["monthly_charges"] * 12).sum()

critical_active_count = len(
    df_scored[(df_scored["churn_prob"] >= 0.70) & (df_scored["churn"] == 0)]
)

# ==============================================================================
# 6. INTERACTIVE SIDEBAR: WHAT-IF SIMULATION & PLATFORM CONTROLS
# ==============================================================================

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%); width: 34px; height: 34px; border-radius: 9px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);">
                <span style="font-size: 18px; color: white;">⚡</span>
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">PredicX ML</h3>
                <span style="font-size: 0.7rem; color: #64748B;">SIMULATION LAB</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 0.6rem 0.8rem; margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.72rem; color: #94A3B8;">Engine Status</span>
                <span class="badge badge-emerald">Online</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span style="font-size: 0.72rem; color: #94A3B8;">Model Pipeline</span>
                <span style="font-size: 0.72rem; color: #CBD5E1; font-weight: 600;">Random Forest</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Customer Feature Vector</p>", unsafe_allow_html=True)

    preset_choice = st.selectbox(
        "Simulation Preset",
        options=[
            "Custom What-If",
            "High-Risk Attrition Candidate",
            "Loyal Long-Term Champion",
            "New High-Touch Onboarding",
        ],
        index=0,
        help="Quickly populate feature vectors with representative behavioral personas.",
    )

    # Defaults according to preset
    if preset_choice == "High-Risk Attrition Candidate":
        def_tenure, def_contract, def_monthly, def_tickets, def_payment, def_usage = 3, "Month-to-month", 98.0, 4, "Electronic check", 180.0
    elif preset_choice == "Loyal Long-Term Champion":
        def_tenure, def_contract, def_monthly, def_tickets, def_payment, def_usage = 48, "2-Year", 55.0, 0, "Credit card (auto)", 520.0
    elif preset_choice == "New High-Touch Onboarding":
        def_tenure, def_contract, def_monthly, def_tickets, def_payment, def_usage = 4, "1-Year", 65.0, 1, "Bank transfer (auto)", 310.0
    else:
        def_tenure, def_contract, def_monthly, def_tickets, def_payment, def_usage = 8, "Month-to-month", 88.5, 3, "Electronic check", 280.0

    sb_tenure = st.slider(
        "Tenure (Months)",
        min_value=1,
        max_value=72,
        value=def_tenure,
        step=1,
        help="Account tenure with service in months.",
    )

    sb_contract = st.selectbox(
        "Contract Commitment",
        options=["Month-to-month", "1-Year", "2-Year"],
        index=["Month-to-month", "1-Year", "2-Year"].index(def_contract),
        help="Agreement terms significantly govern churn probability.",
    )

    sb_monthly = st.slider(
        "Monthly Charges ($)",
        min_value=20.0,
        max_value=125.0,
        value=float(def_monthly),
        step=0.5,
        help="Current recurring monthly bill.",
    )

    sb_tickets = st.slider(
        "Support Tickets Filed",
        min_value=0,
        max_value=9,
        value=def_tickets,
        step=1,
        help="Total friction/support incidents reported.",
    )

    sb_payment = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Credit card (auto)",
            "Bank transfer (auto)",
            "Mailed check",
        ],
        index=[
            "Electronic check",
            "Credit card (auto)",
            "Bank transfer (auto)",
            "Mailed check",
        ].index(def_payment),
    )

    sb_usage = st.slider(
        "Data Usage (GB)",
        min_value=15.0,
        max_value=950.0,
        value=float(def_usage),
        step=5.0,
        help="Data transfer volume in previous billing cycle.",
    )

    calc_total = round(sb_tenure * sb_monthly, 2)
    sb_total = st.number_input(
        "Cumulative Spend ($)",
        min_value=0.0,
        max_value=15000.0,
        value=float(calc_total),
        step=20.0,
    )

    st.markdown("---")

    # Database Maintenance Action
    if st.button("🔄 Reseed Database & Retrain Model", use_container_width=True):
        init_database(force_reseed=True)
        st.cache_resource.clear()
        st.success("Database regenerated and model retrained!")
        st.rerun()


# ==============================================================================
# 7. MAIN HEADER & EXECUTIVE KPI SCORECARDS
# ==============================================================================

st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(17, 24, 39, 0.55); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.15rem 1.6rem; margin-bottom: 1.25rem; box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%); width: 42px; height: 42px; border-radius: 11px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45); flex-shrink: 0;">
                <span style="font-size: 22px; color: white;">⚡</span>
            </div>
            <div>
                <h1 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; line-height: 1.2;">
                    PredicX <span style="font-weight: 400; color: #64748B; font-size: 1.25rem;">| Customer Churn & Predictive ML Analytics</span>
                </h1>
                <p style="margin: 4px 0 0 0; font-size: 0.81rem; color: #94A3B8;">
                    Production Random Forest Pipeline • SQLite Analytics Engine • Real-Time Intervention Queue
                </p>
            </div>
        </div>
        <div style="text-align: right; flex-shrink: 0;">
            <span class="badge badge-indigo">SQLite Engine Synchronized</span>
            <div style="font-size: 0.74rem; color: #64748B; margin-top: 5px; font-family: 'JetBrains Mono', monospace;">
                {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 4 High-Impact KPI Scorecards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    benchmark_variance = global_churn_rate - 18.0
    badge_style = "badge-crimson" if benchmark_variance > 0 else "badge-emerald"
    badge_text = (
        f"+{benchmark_variance:.1f}% vs Ind."
        if benchmark_variance > 0
        else f"{benchmark_variance:.1f}% vs Ind."
    )

    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <span>Global Churn Rate</span>
                <span class="badge {badge_style}">{badge_text}</span>
            </div>
            <div class="kpi-value">{global_churn_rate:.1f}%</div>
            <div class="kpi-footer">
                <span style="color: #64748B;">Baseline: 18.0% | {churned_customers:,} accounts churned</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <span>Annual ARR at Risk</span>
                <span class="badge badge-crimson">Immediate Threat</span>
            </div>
            <div class="kpi-value">${total_arr_at_risk:,.0f}</div>
            <div class="kpi-footer">
                <span style="color: #64748B;">Across {len(critical_accounts_df):,} critical & churned accounts</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    roc_val = metrics["roc_auc"]
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <span>Model ROC-AUC Score</span>
                <span class="badge badge-emerald">Optimal Fit</span>
            </div>
            <div class="kpi-value">{roc_val:.3f}</div>
            <div class="kpi-footer">
                <span style="color: #64748B;">Precision: {metrics['precision']*100:.1f}% | Recall: {metrics['recall']*100:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">
                <span>Active Retention Alerts</span>
                <span class="badge badge-amber">&gt; 70% Probability</span>
            </div>
            <div class="kpi-value">{critical_active_count:,}</div>
            <div class="kpi-footer">
                <span style="color: #64748B;">Active customers requiring immediate outreach</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 8. ANALYTICAL TABS
# ==============================================================================

tab_overview, tab_simulator, tab_queue = st.tabs([
    "📊  Executive Churn Overview",
    "🧪  Individual Customer Simulator (What-If)",
    "🚨  High-Risk Intervention Queue",
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE CHURN OVERVIEW
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns([1.1, 0.9])

    with row1_c1:
        # Chart 1: Feature Importance Horizontal Bar Chart
        fig_importance = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[
                [0, "#312E81"],
                [0.5, "#6366F1"],
                [1.0, "#06B6D4"],
            ],
            labels={"importance": "Gini Importance Score", "feature": ""},
        )
        fig_importance.update_layout(coloraxis_showscale=False)
        fig_importance = apply_executive_dark_theme(
            fig_importance, "Top Drivers Influencing Customer Churn (Feature Importance)"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    with row1_c2:
        # Chart 4: Confusion Matrix Heatmap
        cm = metrics["confusion_matrix"]
        cm_labels = [
            ["True Negative (TN)", "False Positive (FP)"],
            ["False Negative (FN)", "True Positive (TP)"],
        ]

        cm_annotations = []
        for i in range(2):
            for j in range(2):
                cm_annotations.append(
                    dict(
                        x=["Retained (Pred)", "Churned (Pred)"][j],
                        y=["Retained (True)", "Churned (True)"][i],
                        text=f"<b>{cm[i, j]}</b><br><span style='font-size:10px; color:#94A3B8;'>{cm_labels[i][j]}</span>",
                        showarrow=False,
                        font=dict(
                            color="#FFFFFF", size=14, family="Plus Jakarta Sans"
                        ),
                    )
                )

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Retained (Pred)", "Churned (Pred)"],
                y=["Retained (True)", "Churned (True)"],
                colorscale=[[0, "#111827"], [0.5, "#1E3A8A"], [1.0, "#3B82F6"]],
                showscale=False,
            )
        )
        fig_cm.update_layout(annotations=cm_annotations)
        fig_cm = apply_executive_dark_theme(
            fig_cm, "Model Test Evaluation: Confusion Matrix Heatmap"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    row2_c1, row2_c2 = st.columns([1.2, 0.8])

    with row2_c1:
        # Chart 2: Monthly Charges vs. Tenure Scatter Plot
        sample_df = df_scored.sample(
            min(650, len(df_scored)), random_state=42
        ).copy()
        sample_df["Churn Status"] = sample_df["churn"].map(
            {0: "Retained", 1: "Churned"}
        )

        fig_scatter = px.scatter(
            sample_df,
            x="tenure_months",
            y="monthly_charges",
            color="Churn Status",
            size="tech_support_tickets",
            size_max=12,
            color_discrete_map={"Retained": "#10B981", "Churned": "#EF4444"},
            opacity=0.75,
            labels={
                "tenure_months": "Customer Tenure (Months)",
                "monthly_charges": "Monthly Charges ($)",
                "tech_support_tickets": "Support Tickets",
            },
            hover_data=[
                "customer_id",
                "contract_type",
                "payment_method",
                "churn_prob_pct",
            ],
        )
        fig_scatter = apply_executive_dark_theme(
            fig_scatter, "Monthly Charges vs. Tenure (Bubble Size: Support Tickets)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_c2:
        # Chart 3: Churn Rate Distribution across Contract Types
        contract_churn = (
            df_scored.groupby("contract_type")
            .agg(
                total=("churn", "count"),
                churned=("churn", "sum"),
            )
            .reset_index()
        )
        contract_churn["churn_rate_pct"] = np.round(
            (contract_churn["churned"] / contract_churn["total"]) * 100.0, 1
        )
        contract_churn["retained"] = (
            contract_churn["total"] - contract_churn["churned"]
        )

        fig_contract = go.Figure()
        fig_contract.add_trace(
            go.Bar(
                name="Retained",
                x=contract_churn["contract_type"],
                y=contract_churn["retained"],
                marker_color="#10B981",
                text=contract_churn["retained"],
                textposition="auto",
            )
        )
        fig_contract.add_trace(
            go.Bar(
                name="Churned",
                x=contract_churn["contract_type"],
                y=contract_churn["churned"],
                marker_color="#EF4444",
                text=contract_churn["churn_rate_pct"].apply(lambda v: f"{v}%"),
                textposition="auto",
            )
        )
        fig_contract.update_layout(barmode="stack")
        fig_contract = apply_executive_dark_theme(
            fig_contract, "Churn Volume & Rate by Contract Commitment Tier"
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    # SQL Analytics Section
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <h3 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin: 0;">
                SQL Engine Insights: Behavioral Cohorts & Customer Lifetime Value (CLV)
            </h3>
            <span class="badge badge-indigo">SQLite Engine Aggregations</span>
        </div>
    """, unsafe_allow_html=True)

    sql_col1, sql_col2 = st.columns([1.1, 0.9])

    with sql_col1:
        st.markdown(
            "<span style='font-size: 0.8rem; font-weight: 600; color: #CBD5E1;'>Cohort Risk Matrix (Contract × Payment Method)</span>",
            unsafe_allow_html=True,
        )
        display_contract_payment = df_contract_payment.rename(
            columns={
                "contract_type": "Contract",
                "payment_method": "Payment Method",
                "total_customers": "Total Accounts",
                "churned_count": "Churned",
                "churn_rate_pct": "Churn Rate %",
                "avg_monthly_bill": "Avg Monthly ($)",
                "total_arr": "Annual ARR ($)",
            }
        )
        st.dataframe(
            display_contract_payment.style.format({
                "Churn Rate %": "{:.1f}%",
                "Avg Monthly ($)": "${:,.2f}",
                "Annual ARR ($)": "${:,.0f}",
                "Total Accounts": "{:,}",
                "Churned": "{:,}",
            }),
            use_container_width=True,
            height=260,
        )

    with sql_col2:
        st.markdown(
            "<span style='font-size: 0.8rem; font-weight: 600; color: #CBD5E1;'>RFM Behavioral Segmentation</span>",
            unsafe_allow_html=True,
        )
        display_rfm = df_rfm.rename(
            columns={
                "rfm_segment": "Behavioral Segment",
                "count_customers": "Accounts",
                "avg_monthly_charges": "Avg Bill ($)",
                "avg_tickets": "Avg Tickets",
                "segment_churn_pct": "Churn %",
                "segment_arr": "Total ARR ($)",
            }
        )
        st.dataframe(
            display_rfm.style.format({
                "Churn %": "{:.1f}%",
                "Avg Bill ($)": "${:,.2f}",
                "Avg Tickets": "{:.1f}",
                "Total ARR ($)": "${:,.0f}",
                "Accounts": "{:,}",
            }),
            use_container_width=True,
            height=260,
        )


# ------------------------------------------------------------------------------
# TAB 2: INDIVIDUAL CUSTOMER SIMULATOR (WHAT-IF ANALYSIS)
# ------------------------------------------------------------------------------
with tab_simulator:
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    sim_left, sim_right = st.columns([1, 1.25])

    with sim_left:
        st.markdown("""
            <div class="glass-panel" style="margin-bottom: 0.8rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC;">
                        🎛️ Customer Feature Vector Controls
                    </span>
                    <span class="badge badge-indigo">Interactive Simulation</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        t2_preset = st.selectbox(
            "Simulation Preset",
            options=[
                "Custom What-If",
                "High-Risk Attrition Candidate",
                "Loyal Long-Term Champion",
                "New High-Touch Onboarding",
            ],
            index=0,
            key="t2_preset_selector",
            help="Select a persona to auto-populate features.",
        )

        if t2_preset == "High-Risk Attrition Candidate":
            p_tenure, p_contract, p_monthly, p_tickets, p_payment, p_usage = 3, "Month-to-month", 98.0, 4, "Electronic check", 180.0
        elif t2_preset == "Loyal Long-Term Champion":
            p_tenure, p_contract, p_monthly, p_tickets, p_payment, p_usage = 48, "2-Year", 55.0, 0, "Credit card (auto)", 520.0
        elif t2_preset == "New High-Touch Onboarding":
            p_tenure, p_contract, p_monthly, p_tickets, p_payment, p_usage = 4, "1-Year", 65.0, 1, "Bank transfer (auto)", 310.0
        else:
            p_tenure, p_contract, p_monthly, p_tickets, p_payment, p_usage = sb_tenure, sb_contract, sb_monthly, sb_tickets, sb_payment, sb_usage

        sim_contract = st.selectbox(
            "Contract Commitment",
            options=["Month-to-month", "1-Year", "2-Year"],
            index=["Month-to-month", "1-Year", "2-Year"].index(p_contract),
            key="t2_contract",
            help="Contract terms significantly govern churn probability.",
        )

        sim_tenure = st.slider(
            "Tenure (Months)",
            min_value=1,
            max_value=72,
            value=int(p_tenure),
            step=1,
            key="t2_tenure",
            help="Customer lifetime in months.",
        )

        sim_monthly = st.slider(
            "Monthly Charges ($)",
            min_value=20.0,
            max_value=125.0,
            value=float(p_monthly),
            step=0.5,
            key="t2_monthly",
            help="Monthly subscription billing amount.",
        )

        sim_tickets = st.slider(
            "Support Tickets Filed",
            min_value=0,
            max_value=9,
            value=int(p_tickets),
            step=1,
            key="t2_tickets",
            help="Number of friction/support incidents filed.",
        )

        sim_payment = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Credit card (auto)",
                "Bank transfer (auto)",
                "Mailed check",
            ],
            index=[
                "Electronic check",
                "Credit card (auto)",
                "Bank transfer (auto)",
                "Mailed check",
            ].index(p_payment),
            key="t2_payment",
        )

        sim_usage = st.slider(
            "Data Usage (GB)",
            min_value=15.0,
            max_value=950.0,
            value=float(p_usage),
            step=5.0,
            key="t2_usage",
            help="Data transfer volume in previous billing cycle.",
        )

        t2_calc_total = round(sim_tenure * sim_monthly, 2)
        sim_total = st.number_input(
            "Cumulative Spend ($)",
            min_value=0.0,
            max_value=15000.0,
            value=float(t2_calc_total),
            step=20.0,
            key="t2_total",
        )

    # Construct feature DataFrame from the controls
    sim_input_df = pd.DataFrame([
        {
            "tenure_months": sim_tenure,
            "monthly_charges": sim_monthly,
            "total_charges": sim_total,
            "contract_type": sim_contract,
            "tech_support_tickets": sim_tickets,
            "payment_method": sim_payment,
            "usage_gb": sim_usage,
        }
    ])

    sim_prob = pipeline.predict_proba(sim_input_df)[0, 1]
    sim_prob_pct = sim_prob * 100.0

    if sim_prob_pct >= 70.0:
        sim_risk_status = "CRITICAL CHURN RISK"
        gauge_bar_color = "#EF4444"
        badge_class = "badge-crimson"
    elif sim_prob_pct >= 40.0:
        sim_risk_status = "ELEVATED CONCERN"
        gauge_bar_color = "#F59E0B"
        badge_class = "badge-amber"
    else:
        sim_risk_status = "HEALTHY RETENTION"
        gauge_bar_color = "#10B981"
        badge_class = "badge-emerald"

    with sim_right:
        # Gauge Meter for Churn Probability
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=sim_prob_pct,
                delta={
                    "reference": global_churn_rate,
                    "increasing": {"color": "#EF4444"},
                    "decreasing": {"color": "#10B981"},
                    "suffix": "% vs Avg",
                },
                number={
                    "suffix": "%",
                    "font": {
                        "size": 42,
                        "color": "#F8FAFC",
                        "family": "Plus Jakarta Sans",
                    },
                },
                title={
                    "text": "<b>PREDICTED CHURN PROBABILITY</b><br><span style='font-size:12px; color:#94A3B8;'>Real-Time Scikit-Learn Model Inference</span>",
                    "font": {"size": 14, "color": "#CBD5E1"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#475569",
                        "tickfont": {"color": "#94A3B8"},
                    },
                    "bar": {"color": gauge_bar_color, "thickness": 0.28},
                    "bgcolor": "rgba(255, 255, 255, 0.05)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {
                            "range": [0, 40],
                            "color": "rgba(16, 185, 129, 0.12)",
                        },
                        {
                            "range": [40, 70],
                            "color": "rgba(245, 158, 11, 0.12)",
                        },
                        {
                            "range": [70, 100],
                            "color": "rgba(239, 68, 68, 0.15)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#EF4444", "width": 3},
                        "thickness": 0.85,
                        "value": 70,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94A3B8", "family": "Plus Jakarta Sans"},
            height=255,
            margin=dict(l=25, r=25, t=55, b=20),
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # Prescriptive AI Playbook recommendations
        st.markdown(f"""
            <div class="glass-panel" style="margin-top: -10px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                    <span style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC;">
                        🎯 Prescriptive Retention Playbook
                    </span>
                    <span class="badge {badge_class}">{sim_risk_status}</span>
                </div>
        """, unsafe_allow_html=True)

        recommendations = []
        if sim_contract == "Month-to-month":
            recommendations.append(
                "<b>Contract Upgrade:</b> Customer is on Month-to-Month terms. Offer a 1-Year commitment with a 15% promotional discount for 6 months."
            )
        if sim_tickets >= 3:
            recommendations.append(
                f"<b>Technical Escalation:</b> {sim_tickets} support tickets indicate platform friction. Dispatch a dedicated Senior Solutions Engineer for white-glove resolution."
            )
        if sim_monthly > 80.0:
            recommendations.append(
                "<b>Pricing Optimization:</b> Monthly bill exceeds $80. Conduct a plan rightsizing audit to remove unneeded add-ons and lower bill shock."
            )
        if sim_payment == "Electronic check":
            recommendations.append(
                "<b>Payment Friction Reduction:</b> Paying by Electronic Check. Offer a one-time $10 account credit to enroll in automated ACH or Credit Card billing."
            )
        if sim_tenure <= 6:
            recommendations.append(
                "<b>Onboarding Concierge:</b> Account is in the vulnerable 90-day onboarding window. Schedule an executive Customer Success review."
            )

        if not recommendations:
            recommendations.append(
                "<b>Healthy Account:</b> Customer displays high loyalty and stable utilization. Target for account expansion, premium add-ons, and testimonial nomination."
            )

        for rec in recommendations:
            st.markdown(f"""
                <div style="display: flex; align-items: flex-start; gap: 8px; margin-bottom: 0.45rem; font-size: 0.82rem; color: #CBD5E1;">
                    <span style="color: #6366F1; font-size: 14px;">▸</span>
                    <span>{rec}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: HIGH-RISK INTERVENTION QUEUE
# ------------------------------------------------------------------------------
with tab_queue:
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3, filter_c4 = st.columns([1, 1, 1, 1])

    with filter_c1:
        threshold = st.slider(
            "Churn Risk Threshold (%)",
            min_value=50,
            max_value=95,
            value=70,
            step=5,
            help="Filter queue to customers with predicted churn probability equal to or exceeding this threshold.",
        )

    with filter_c2:
        contract_filter = st.multiselect(
            "Contract Filter",
            options=["Month-to-month", "1-Year", "2-Year"],
            default=["Month-to-month", "1-Year", "2-Year"],
        )

    with filter_c3:
        payment_filter = st.multiselect(
            "Payment Method Filter",
            options=[
                "Electronic check",
                "Credit card (auto)",
                "Bank transfer (auto)",
                "Mailed check",
            ],
            default=[
                "Electronic check",
                "Credit card (auto)",
                "Bank transfer (auto)",
                "Mailed check",
            ],
        )

    with filter_c4:
        search_query = st.text_input(
            "Search Customer ID",
            placeholder="e.g. CUST-10042",
        ).strip().upper()

    # Filter Scored Dataset
    queue_df = df_scored[
        (df_scored["churn_prob_pct"] >= threshold)
        & (df_scored["contract_type"].isin(contract_filter))
        & (df_scored["payment_method"].isin(payment_filter))
    ].copy()

    if search_query:
        queue_df = queue_df[
            queue_df["customer_id"].str.contains(search_query, na=False)
        ]

    queue_df = queue_df.sort_values(by="churn_prob_pct", ascending=False)

    # Queue Metrics Strip
    q_arr_threat = (queue_df["monthly_charges"] * 12).sum()
    q_avg_tenure = queue_df["tenure_months"].mean() if len(queue_df) > 0 else 0
    q_avg_tickets = (
        queue_df["tech_support_tickets"].mean() if len(queue_df) > 0 else 0
    )

    st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin: 0.75rem 0 1rem 0; flex-wrap: wrap;">
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 10px; padding: 0.6rem 1rem;">
                <span style="font-size: 0.72rem; text-transform: uppercase; color: #F87171; font-weight: 700;">Target Cohort Size</span>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">{len(queue_df):,} Accounts</div>
            </div>
            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 0.6rem 1rem;">
                <span style="font-size: 0.72rem; text-transform: uppercase; color: #818CF8; font-weight: 700;">Cohort ARR Exposure</span>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">${q_arr_threat:,.0f}</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.6rem 1rem;">
                <span style="font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Avg Cohort Tenure</span>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">{q_avg_tenure:.1f} mos</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.6rem 1rem;">
                <span style="font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Avg Support Tickets</span>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">{q_avg_tickets:.1f} tickets</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Format Display Queue
    display_queue = queue_df[[
        "customer_id",
        "churn_prob_pct",
        "risk_tier",
        "contract_type",
        "monthly_charges",
        "tenure_months",
        "tech_support_tickets",
        "payment_method",
        "usage_gb",
        "churn",
    ]].rename(
        columns={
            "customer_id": "Customer ID",
            "churn_prob_pct": "Predicted Churn %",
            "risk_tier": "Risk Tier",
            "contract_type": "Contract",
            "monthly_charges": "Monthly Bill ($)",
            "tenure_months": "Tenure (Mo)",
            "tech_support_tickets": "Support Tickets",
            "payment_method": "Payment Method",
            "usage_gb": "Data Usage (GB)",
            "churn": "Historical Churned",
        }
    )

    st.dataframe(
        display_queue.style.format({
            "Predicted Churn %": "{:.1f}%",
            "Monthly Bill ($)": "${:,.2f}",
            "Data Usage (GB)": "{:.1f} GB",
        }),
        use_container_width=True,
        height=380,
    )

    # Export Action
    csv_bytes = queue_df.to_csv(index=False).encode("utf-8")
    export_filename = f"churn_intervention_queue_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.download_button(
        label="📥  Export High-Risk Retention Queue (CSV)",
        data=csv_bytes,
        file_name=export_filename,
        mime="text/csv",
        help="Export all filtered accounts directly to CSV for automated CRM / Outreach integration.",
    )

# ==============================================================================
# 9. FOOTER
# ==============================================================================

st.markdown("""
    <div style="margin-top: 3.5rem; padding-top: 1.25rem; border-top: 1px solid rgba(255, 255, 255, 0.06); text-align: center; font-size: 0.78rem; color: #64748B;">
        PredicX ML Platform • Powered by Streamlit, Scikit-Learn, Plotly & SQLite • Enterprise Ready
    </div>
""", unsafe_allow_html=True)
