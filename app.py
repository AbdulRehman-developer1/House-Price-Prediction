"""
House Price Prediction — Premium Streamlit AI Dashboard
==========================================================
Loads the trained GradientBoostingRegressor artifact (model + LabelEncoder +
StandardScaler) produced by `house-price-prediction-notebook.ipynb` and serves
an interactive, production-styled prediction interface.

Run with:
    streamlit run app.py

Expected folder layout (app.py must sit at the project root):
    House_Price_Prediction/
    ├── app.py                     <- this file
    ├── models/
    │   └── artifact.pkl           <- {'Best Model', 'Encoding', 'Scaling'}
    ├── data set/
    │   └── House_Price_Prediction_1500_Records.csv
    └── plots/
        ├── correlation heatmap.png
        ├── histogram.png
        ├── KDE plot.png
        ├── boxplot.png
        ├── pair plot.png
        ├── price distribution.png
        └── price comparison according to garage availability.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# LOGGING (internal debugging only — never shown to the end user)
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("house_price_app")

# --------------------------------------------------------------------------------------
# PATHS — resolved relative to this file so the project can be moved anywhere
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "artifact.pkl"
DATASET_PATH = BASE_DIR / "data set" / "House_Price_Prediction_1500_Records.csv"
PLOTS_DIR = BASE_DIR / "plots"

# --------------------------------------------------------------------------------------
# CONSTANTS — sourced directly from the training notebook (do not invent values)
# --------------------------------------------------------------------------------------
# Exact feature order the scaler / model were fit on (artifact['Scaling'].feature_names_in_)
FEATURE_ORDER = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Age",
    "Garage",
    "LocationScore",
    "DistanceToCity_km",
    "Condition",
]

# The notebook first maps the raw string condition to an ordinal score, then
# runs a LabelEncoder over that ordinal column before scaling.
CONDITION_MAP = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}

# Real min/max/median observed in the training dataset — used to build sane,
# validated widget ranges instead of arbitrary guesses.
FEATURE_RANGES = {
    "Area_sqft": {"min": 600, "max": 5000, "default": 2681, "step": 10, "unit": "sqft"},
    "Bedrooms": {"min": 1, "max": 6, "default": 3},
    "Bathrooms": {"min": 1, "max": 5, "default": 3},
    "Floors": {"min": 1, "max": 3, "default": 2},
    "Age": {"min": 0, "max": 50, "default": 25, "unit": "years"},
    "Garage": {"min": 0, "max": 3, "default": 1, "unit": "spaces"},
    "LocationScore": {"min": 1, "max": 10, "default": 5, "unit": "/10"},
    "DistanceToCity_km": {"min": 1.0, "max": 40.0, "default": 20.0, "step": 0.5, "unit": "km"},
}

# Metrics reported by the notebook for the final tuned GradientBoostingRegressor
# (GridSearchCV best params: lr=0.05, max_depth=2, n_estimators=200, subsample=0.8)
MODEL_METRICS = {
    "R2": 0.89,
    "RMSE": 75842.50,
    "MAE": 40504.24,
    "CV_R2_best": 0.9132,
}

MODEL_NAME = "Gradient Boosting Regressor"
CURRENCY = "$"

PLOT_FILES = {
    "Correlation Heatmap": "correlation heatmap.png",
    "Price Distribution": "price distribution.png",
    "Feature Distributions": "histogram.png",
    "Price Density (KDE)": "KDE plot.png",
    "Outlier Boxplot": "boxplot.png",
    "Price vs Garage Availability": "price comparison according to garage availability.png",
    "Pairwise Relationships": "pair plot.png",
}

st.set_page_config(
    page_title="House Price Prediction | AI Valuation Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifact():
    """Load the joblib artifact bundle {Best Model, Encoding, Scaling}."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    for key in ("Best Model", "Encoding", "Scaling"):
        if key not in bundle:
            raise KeyError(f"Artifact is missing expected key: '{key}'")
    return bundle


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATASET_PATH.exists():
        return None
    return pd.read_csv(DATASET_PATH)


def get_plot_path(filename: str) -> Path | None:
    p = PLOTS_DIR / filename
    return p if p.exists() else None


# --------------------------------------------------------------------------------------
# PREPROCESSING + PREDICTION — mirrors the notebook pipeline exactly
# --------------------------------------------------------------------------------------
def preprocess_input(raw_inputs: dict, encoder) -> pd.DataFrame:
    """Convert raw UI inputs into the exact numeric feature frame the scaler expects."""
    ordinal_condition = CONDITION_MAP[raw_inputs["Condition"]]
    try:
        encoded_condition = encoder.transform([ordinal_condition])[0]
    except ValueError as exc:
        raise ValueError(
            "Condition value could not be encoded by the trained LabelEncoder."
        ) from exc

    row = {
        "Area_sqft": raw_inputs["Area_sqft"],
        "Bedrooms": raw_inputs["Bedrooms"],
        "Bathrooms": raw_inputs["Bathrooms"],
        "Floors": raw_inputs["Floors"],
        "Age": raw_inputs["Age"],
        "Garage": raw_inputs["Garage"],
        "LocationScore": raw_inputs["LocationScore"],
        "DistanceToCity_km": raw_inputs["DistanceToCity_km"],
        "Condition": encoded_condition,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def make_prediction(raw_inputs: dict, bundle: dict) -> float:
    model = bundle["Best Model"]
    encoder = bundle["Encoding"]
    scaler = bundle["Scaling"]

    features_df = preprocess_input(raw_inputs, encoder)
    scaled = scaler.transform(features_df)
    prediction = model.predict(scaled)[0]
    return float(prediction)


def validate_inputs(raw_inputs: dict) -> list[str]:
    errors = []
    for key in ["Area_sqft", "Age", "LocationScore", "DistanceToCity_km"]:
        cfg = FEATURE_RANGES[key]
        val = raw_inputs[key]
        if val is None or val < cfg["min"] or val > cfg["max"]:
            errors.append(f"'{key}' must be between {cfg['min']} and {cfg['max']}.")
    if raw_inputs["Condition"] not in CONDITION_MAP:
        errors.append("Condition must be one of Poor, Fair, Good, Excellent.")
    return errors


# --------------------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-deep: #090C14;
            --bg-panel: #10141F;
            --glass: rgba(255, 255, 255, 0.045);
            --glass-border: rgba(255, 255, 255, 0.08);
            --violet: #8B5CF6;
            --violet-soft: #A78BFA;
            --cyan: #22D3EE;
            --indigo: #6366F1;
            --gold: #F0B840;
            --text-primary: #F3F5FA;
            --text-secondary: #9CA3B5;
            --success: #34D399;
            --danger: #FB7185;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4, .app-title { font-family: 'Space Grotesk', sans-serif; }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(139, 92, 246, 0.14), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(34, 211, 238, 0.10), transparent 40%),
                var(--bg-deep);
            color: var(--text-primary);
        }

        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
        }
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            display: flex !important;
        }
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px; }

        /* ---------- HERO ---------- */
        .hero {
            position: relative;
            padding: 2.8rem 2.6rem;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(34,211,238,0.10) 55%, rgba(16,20,31,0.4));
            border: 1px solid var(--glass-border);
            overflow: hidden;
            margin-bottom: 1.8rem;
            animation: fadeIn 0.7s ease-out;
        }
        .hero::before {
            content: "";
            position: absolute; top: -60px; right: -60px;
            width: 260px; height: 260px; border-radius: 50%;
            background: radial-gradient(circle, rgba(139,92,246,0.45), transparent 70%);
            filter: blur(10px);
            animation: float 7s ease-in-out infinite;
        }
        .hero::after {
            content: "";
            position: absolute; bottom: -80px; left: 10%;
            width: 220px; height: 220px; border-radius: 50%;
            background: radial-gradient(circle, rgba(34,211,238,0.35), transparent 70%);
            filter: blur(14px);
            animation: float 9s ease-in-out infinite reverse;
        }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(139, 92, 246, 0.16);
            border: 1px solid rgba(139, 92, 246, 0.4);
            color: var(--violet-soft);
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em;
            padding: 5px 14px; border-radius: 999px; text-transform: uppercase;
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.6rem; font-weight: 700; margin: 0 0 0.6rem 0;
            background: linear-gradient(90deg, #FFFFFF, var(--violet-soft) 60%, var(--cyan));
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero p { color: var(--text-secondary); font-size: 1.02rem; max-width: 640px; margin: 0 0 1.4rem 0; }
        .status-pill {
            position: relative; z-index: 2;
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.35);
            color: var(--success); padding: 7px 16px; border-radius: 999px;
            font-size: 0.82rem; font-weight: 600; letter-spacing: 0.03em;
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%; background: var(--success);
            box-shadow: 0 0 0 0 rgba(52,211,153, 0.6);
            animation: pulse 1.8s infinite;
        }

        /* ---------- GLASS CARD ---------- */
        .glass-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 1.6rem 1.7rem;
            backdrop-filter: blur(18px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            animation: slideUp 0.5s ease-out;
            margin-bottom: 1.2rem;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            border-color: rgba(139,92,246,0.35);
            box-shadow: 0 14px 40px rgba(139,92,246,0.12);
        }
        .section-title {
            font-size: 1.02rem; font-weight: 600; color: var(--text-primary);
            display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;
        }
        .section-sub { color: var(--text-secondary); font-size: 0.85rem; margin-top: -0.6rem; margin-bottom: 1rem; }

        /* ---------- METRIC CARDS ---------- */
        .metric-card {
            background: linear-gradient(155deg, rgba(139,92,246,0.14), rgba(255,255,255,0.02));
            border: 1px solid var(--glass-border);
            border-radius: 16px; padding: 1.1rem 1.2rem;
            transition: transform 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-2px); }
        .metric-label { color: var(--text-secondary); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.07em; }
        .metric-value { font-size: 1.55rem; font-weight: 700; margin-top: 4px; color: var(--text-primary); }
        .metric-icon { font-size: 1.3rem; opacity: 0.85; }

        /* ---------- RESULT CARD ---------- */
        .result-card {
            background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(34,211,238,0.12));
            border: 1px solid rgba(139,92,246,0.45);
            border-radius: 24px; padding: 2.2rem 2rem; text-align: center;
            animation: resultPop 0.5s cubic-bezier(0.22, 1, 0.36, 1);
            position: relative; overflow: hidden;
        }
        .result-card::before {
            content: ""; position: absolute; inset: 0;
            background: radial-gradient(circle at 50% -10%, rgba(255,255,255,0.18), transparent 55%);
        }
        .result-label { color: var(--text-secondary); font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
        .result-price {
            font-size: 3.1rem; font-weight: 700; margin: 0.4rem 0 0.8rem 0;
            background: linear-gradient(90deg, #FFFFFF, var(--cyan));
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }
        .result-tags { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-top: 0.6rem; }
        .tag-chip {
            background: rgba(255,255,255,0.07); border: 1px solid var(--glass-border);
            padding: 6px 14px; border-radius: 999px; font-size: 0.8rem; color: var(--text-secondary);
        }
        .tag-chip b { color: var(--text-primary); }

        /* ---------- BUTTON ---------- */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(120deg, var(--violet), var(--indigo) 55%, var(--cyan));
            background-size: 200% 200%;
            color: white; font-weight: 700; font-size: 1.02rem;
            border: none; border-radius: 14px; padding: 0.85rem 1.2rem;
            box-shadow: 0 10px 30px rgba(139,92,246,0.35);
            transition: all 0.3s ease;
            letter-spacing: 0.02em;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 16px 40px rgba(139,92,246,0.5);
            background-position: 100% 50%;
        }
        div.stButton > button:active { transform: translateY(0px) scale(0.99); }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0C0F19 0%, #090C14 100%);
            border-right: 1px solid var(--glass-border);
        }
        .sidebar-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 0.6rem 0 1.2rem 0; margin-bottom: 0.6rem;
            border-bottom: 1px solid var(--glass-border);
        }
        .sidebar-brand-icon {
            width: 40px; height: 40px; border-radius: 12px;
            background: linear-gradient(135deg, var(--violet), var(--cyan));
            display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
        }
        .sidebar-brand-text b { font-size: 1rem; display: block; color: var(--text-primary); }
        .sidebar-brand-text span { font-size: 0.72rem; color: var(--text-secondary); }
        .sidebar-status-row { display: flex; align-items: center; gap: 8px; font-size: 0.83rem; color: var(--text-secondary); margin: 6px 0; }
        .sidebar-status-row .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); box-shadow: 0 0 0 0 rgba(52,211,153,0.6); animation: pulse 1.8s infinite; }
        .sidebar-footer { font-size: 0.72rem; color: var(--text-secondary); margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--glass-border); }

        /* ---------- FOOTER ---------- */
        .app-footer {
            text-align: center; padding: 2.2rem 0 0.6rem 0; color: var(--text-secondary); font-size: 0.85rem;
        }
        .app-footer .heart { color: var(--danger); }
        .app-footer b { color: var(--text-primary); }

        /* ---------- MISC ---------- */
        .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 0.4rem; }
        .mini-badge {
            background: rgba(255,255,255,0.06); border: 1px solid var(--glass-border);
            padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; color: var(--text-secondary);
        }
        hr.soft-divider { border: none; border-top: 1px solid var(--glass-border); margin: 1.4rem 0; }

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes resultPop { 0% { opacity: 0; transform: scale(0.94); } 100% { opacity: 1; transform: scale(1); } }
        @keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-16px); } }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(52,211,153, 0.55); }
            70% { box-shadow: 0 0 0 9px rgba(52,211,153, 0); }
            100% { box-shadow: 0 0 0 0 rgba(52,211,153, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# RENDER FUNCTIONS
# --------------------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">🤖 AI / ML · Regression Engine</span>
            <h1>🏠 House Price Prediction</h1>
            <p>Estimate a fair market value for a property in seconds. A Gradient Boosting
            model trained on 1,500+ real transactions weighs area, location, condition and
            more to generate an instant, data-driven valuation.</p>
            <div class="status-pill"><span class="status-dot"></span> MODEL STATUS: ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_ready: bool) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">🏠</div>
                <div class="sidebar-brand-text">
                    <b>Valuation AI</b>
                    <span>House Price Prediction</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            ["🔮 Prediction", "🧠 Model Insights", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
        st.markdown("**AI MODEL**")
        if model_ready:
            st.markdown(
                """
                <div class="sidebar-status-row"><span class="dot"></span> Model Loaded</div>
                <div class="sidebar-status-row"><span class="dot"></span> Prediction Ready</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("🔴 Model unavailable — see error above.")

        st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
        st.markdown("**Model Info**")
        st.markdown(
            f"""
            <div class="badge-row">
                <span class="mini-badge">{MODEL_NAME}</span>
                <span class="mini-badge">R² {MODEL_METRICS['R2']:.2f}</span>
                <span class="mini-badge">9 features</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-footer">
                Version 1.0.0<br>
                Built by <b>Abdul Rehman</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


def render_metric_card(col, icon: str, label: str, value: str):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_section(bundle: dict):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Property Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Structural & size information</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    area = c1.number_input(
        "Area (sqft)",
        min_value=float(FEATURE_RANGES["Area_sqft"]["min"]),
        max_value=float(FEATURE_RANGES["Area_sqft"]["max"]),
        value=float(FEATURE_RANGES["Area_sqft"]["default"]),
        step=10.0,
        help="Total built-up area of the property in square feet.",
    )
    age = c2.slider(
        "Property Age (years)",
        min_value=FEATURE_RANGES["Age"]["min"],
        max_value=FEATURE_RANGES["Age"]["max"],
        value=FEATURE_RANGES["Age"]["default"],
        help="Years since construction.",
    )
    floors = c3.select_slider(
        "Floors",
        options=list(range(FEATURE_RANGES["Floors"]["min"], FEATURE_RANGES["Floors"]["max"] + 1)),
        value=FEATURE_RANGES["Floors"]["default"],
    )

    c4, c5, c6 = st.columns(3)
    bedrooms = c4.slider(
        "🛏️ Bedrooms",
        min_value=FEATURE_RANGES["Bedrooms"]["min"],
        max_value=FEATURE_RANGES["Bedrooms"]["max"],
        value=FEATURE_RANGES["Bedrooms"]["default"],
    )
    bathrooms = c5.slider(
        "🛁 Bathrooms",
        min_value=FEATURE_RANGES["Bathrooms"]["min"],
        max_value=FEATURE_RANGES["Bathrooms"]["max"],
        value=FEATURE_RANGES["Bathrooms"]["default"],
    )
    garage = c6.select_slider(
        "🚗 Garage Spaces",
        options=list(range(FEATURE_RANGES["Garage"]["min"], FEATURE_RANGES["Garage"]["max"] + 1)),
        value=FEATURE_RANGES["Garage"]["default"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📍 Location & Quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Neighbourhood and overall condition</div>', unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    location_score = c7.slider(
        "Location Score",
        min_value=FEATURE_RANGES["LocationScore"]["min"],
        max_value=FEATURE_RANGES["LocationScore"]["max"],
        value=FEATURE_RANGES["LocationScore"]["default"],
        help="1 = least desirable area, 10 = most desirable area.",
    )
    distance = c8.number_input(
        "Distance to City (km)",
        min_value=float(FEATURE_RANGES["DistanceToCity_km"]["min"]),
        max_value=float(FEATURE_RANGES["DistanceToCity_km"]["max"]),
        value=float(FEATURE_RANGES["DistanceToCity_km"]["default"]),
        step=0.5,
    )
    condition = c9.selectbox(
        "🏗️ Condition",
        options=list(CONDITION_MAP.keys()),
        index=list(CONDITION_MAP.keys()).index("Good"),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    raw_inputs = {
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Age": age,
        "Garage": garage,
        "LocationScore": location_score,
        "DistanceToCity_km": distance,
        "Condition": condition,
    }

    predict_clicked = st.button("✨ Predict Now", use_container_width=True)

    if predict_clicked:
        errors = validate_inputs(raw_inputs)
        if errors:
            st.error("Please fix the following before predicting:\n\n" + "\n".join(f"- {e}" for e in errors))
            return

        with st.spinner("🤖 Analyzing property features..."):
            try:
                prediction = make_prediction(raw_inputs, bundle)
            except Exception:
                logger.exception("Prediction failed")
                st.error("Something went wrong while generating the prediction. Please verify your inputs.")
                return

        render_result_card(prediction, raw_inputs)
        render_explainability(bundle, raw_inputs)


def render_result_card(prediction: float, raw_inputs: dict):
    formatted_price = f"{CURRENCY}{prediction:,.0f}"
    margin = MODEL_METRICS["MAE"]
    low = f"{CURRENCY}{max(prediction - margin, 0):,.0f}"
    high = f"{CURRENCY}{prediction + margin:,.0f}"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">🎯 Estimated Market Value</div>
            <div class="result-price">{formatted_price}</div>
            <div class="result-tags">
                <span class="tag-chip">Model: <b>{MODEL_NAME}</b></span>
                <span class="tag-chip">R² Score: <b>{MODEL_METRICS['R2']:.2f}</b></span>
                <span class="tag-chip">Typical range: <b>{low} – {high}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Range reflects the model's average error (MAE ≈ {CURRENCY}{margin:,.0f}) on held-out test data, "
        "not a formal confidence interval."
    )


def render_explainability(bundle: dict, raw_inputs: dict):
    model = bundle["Best Model"]
    if not hasattr(model, "feature_importances_"):
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Why this prediction?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Global feature importance learned by the model during training</div>',
        unsafe_allow_html=True,
    )

    importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER).sort_values(ascending=False)
    top = importances.head(5)
    for feat, imp in top.items():
        pct = imp * 100
        st.markdown(f"**{feat}** — {pct:.1f}% of the model's decision")
        st.progress(min(imp / importances.max(), 1.0))

    dominant = importances.index[0]
    st.caption(
        f"For this model, **{dominant}** is by far the strongest driver of predicted price — "
        "changes to it move the estimate the most."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_insights(bundle: dict):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Metrics measured on the held-out test split, as reported in the training notebook</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    render_metric_card(m1, "🎯", "R² Score", f"{MODEL_METRICS['R2']:.2f}")
    render_metric_card(m2, "📉", "RMSE", f"{CURRENCY}{MODEL_METRICS['RMSE']:,.0f}")
    render_metric_card(m3, "📐", "MAE", f"{CURRENCY}{MODEL_METRICS['MAE']:,.0f}")
    render_metric_card(m4, "🔁", "Best CV R²", f"{MODEL_METRICS['CV_R2_best']:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    model = bundle["Best Model"]
    params = model.get_params()
    relevant = {k: params[k] for k in ["n_estimators", "learning_rate", "max_depth", "subsample"] if k in params}
    cols = st.columns(len(relevant))
    for col, (k, v) in zip(cols, relevant.items()):
        render_metric_card(col, "🔧", k.replace("_", " ").title(), str(v))
    st.caption(
        "Selected via GridSearchCV (5-fold CV, scoring = R²) across LinearRegression, Ridge, Lasso, "
        "ElasticNet, RandomForest, GradientBoosting, AdaBoost, SVR and KNN."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Global Feature Importance</div>', unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER).sort_values(ascending=False)
        st.bar_chart(importances)
    else:
        st.info("This model does not expose feature importances.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_analytics():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Visualizations generated during model training</div>',
        unsafe_allow_html=True,
    )

    if not PLOTS_DIR.exists():
        st.warning("The `plots/` folder was not found next to app.py, so saved charts can't be displayed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tab_labels = list(PLOT_FILES.keys())
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            path = get_plot_path(PLOT_FILES[label])
            if path:
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"'{PLOT_FILES[label]}' was not found in the plots folder.")
    st.markdown("</div>", unsafe_allow_html=True)

    df = load_dataset()
    if df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗃️ Training Data Snapshot</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"{len(df):,} records · {df.shape[1]} columns")
        st.markdown("</div>", unsafe_allow_html=True)


def render_about():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This application serves a **Gradient Boosting Regressor** trained on 1,500 historical
        house sale records. The pipeline handles missing-value imputation (median for continuous
        columns, mode for discrete counts), maps property condition to an ordinal scale before
        label-encoding it, standardizes all nine features with a fitted `StandardScaler`, and
        finally predicts price with the tuned model — reproducing the exact steps from the
        training notebook.

        **Pipeline summary**
        1. Collect raw property details from the form
        2. Map `Condition` → ordinal (Poor=1 … Excellent=4) → `LabelEncoder`
        3. Assemble the 9-feature vector in training order
        4. Scale with the saved `StandardScaler`
        5. Predict with the saved `GradientBoostingRegressor`
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with <span class="heart">❤️</span> using Python, Streamlit &amp; Machine Learning<br>
            Developed by <b>Abdul Rehman</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    inject_css()
    render_header()

    bundle = None
    model_ready = False
    try:
        bundle = load_artifact()
        model_ready = True
    except FileNotFoundError:
        st.error(
            "⚠️ The trained model artifact is missing. Expected it at "
            f"`{MODEL_PATH.relative_to(BASE_DIR)}`. Please make sure the `models/` folder "
            "sits next to app.py."
        )
    except KeyError as exc:
        st.error(f"⚠️ The model artifact appears corrupted or incomplete: {exc}")
    except Exception as exc:
        logger.exception("Failed to load model artifact")
        st.error("⚠️ Something went wrong while loading the AI model. Please check the artifact file.")
        with st.expander("🔧 Technical details (for debugging)"):
            st.code(f"{type(exc).__name__}: {exc}")
            st.caption(
                "This is most often a scikit-learn / joblib version mismatch between the "
                "environment the model was trained in and the environment running this app. "
                "Try: `pip install -r requirements.txt` in a clean virtual environment, and "
                "confirm with `pip show scikit-learn joblib`."
            )

    page = render_sidebar(model_ready)

    if page == "🔮 Prediction":
        if model_ready:
            render_prediction_section(bundle)
        else:
            st.info("The prediction form is unavailable until the model artifact loads successfully.")
    elif page == "🧠 Model Insights":
        if model_ready:
            render_model_insights(bundle)
        else:
            st.info("Model insights are unavailable until the model artifact loads successfully.")
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "ℹ️ About":
        render_about()

    render_footer()


if __name__ == "__main__":
    main()