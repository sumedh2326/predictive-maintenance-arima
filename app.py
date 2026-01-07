
# app.py — Predictive Maintenance (Optimized for Streamlit Cloud)
# Author: Sumedh's PM PoC
# Runtime: Streamlit Community Cloud (Python 3.11 recommended)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import StringIO

# -------------------- Optional: quiet logs --------------------
st.set_option("deprecation.showfileUploaderEncoding", False)

# -------------------- Try ARIMA from statsmodels; show hint if missing --------------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATS_OK = True
    IMPORT_ERR = ""
except Exception as e:
    STATS_OK = False
    IMPORT_ERR = (
        "⚠️ 'statsmodels' (and deps: numpy, scipy, patsy) not available.\n"
        "Ensure pinned versions in requirements.txt:\n"
        "  statsmodels==0.14.1, numpy==1.26.4, scipy==1.10.1, patsy==0.5.6\n\n"
        f"Import error: {e}"
    )

# ====================== PAGE CONFIG & CONSTANTS ======================
st.set_page_config(page_title="Predictive Maintenance | Home (Streamlit)", layout="wide")

# Fixed dropdowns
SHOPS = ["Press Shop", "Annealing Shop", "Machine Shop", "Winding Shop"]
MACHINES = ["PM_5001", "PM_5002", "PM_5012", "MS_2087"]
PARAMETERS = ["Motor Current", "Bearing Vibration", "Motor Temperature"]

# Map friendly parameter -> CSV column (adjust when new signals are available)
PARAM_TO_COLUMN = {
    "Motor Current": "5001_P0421_M_MTR_AMTR_AVG",                 # present in your CSV
    "Bearing Vibration": "5001_P0223_MFLY_MTR_VIB_VRMS_100msec",  # present in your CSV
    "Motor Temperature": None                                      # not present -> disabled visually
}

# Baseline defaults (set to engineering limits if known)
PARAM_BASELINES = {
    "Motor Current": 46.9,    # near observed 95th percentile from your data
    "Bearing Vibration": 3.8, # placeholder; adjust to your spec
    "Motor Temperature": 75.0 # placeholder; adjust when temp data is available
}

# Lite performance settings
DEFAULT_WINDOW_DAYS = 7           # only model the last N days
DOWNSAMPLE_FREQ = "10T"           # 10-minute cadence
ARIMA_ORDERS_FAST = [(1, 0, 1), (2, 0, 2)]
ARIMA_ORDERS_ACCURATE = ARIMA_ORDERS_FAST + [(1, 1, 1)]
DEFAULT_HORIZON_DAYS = 1          # smaller forecast by default (Cloud-friendly)

# ====================== CSS (chips & selectors) ======================
CUSTOM_CSS = """
<style>
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
.header-left h1 { margin: 0; font-size: 1.4rem; font-weight: 700; }
.header-right { text-align: right; font-size: 0.95rem; color: #444; }

.selector-title { font-weight: 600; margin-bottom: 0.25rem; }
.stSelectbox > div > div { border: 1px solid #444; border-radius: 6px; }

.chips-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 2px 0; }
.chip {
    display: inline-flex; align-items: center; padding: 6px 10px;
    border-radius: 6px; border: 1px solid #444; color: #222; background: #fff;
    font-size: 0.9rem; cursor: default;
}
.chip-active {
    border-color: #316ed0; color: #316ed0; font-weight: 600;
}
.chip-disabled {
    color: #777; border-color: #aaa; background: #f3f3f3;
    pointer-events: none;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====================== HEADER ======================
col_h1, col_user = st.columns([3, 2])
with col_h1:
    st.markdown('<div class="header-left"><h1>Home Page</h1></div>', unsafe_allow_html=True)
with col_user:
    st.markdown('<div class="header-right">Hi, Sumedh! | User Details</div>', unsafe_allow_html=True)

st.write("")  # spacer

# ====================== FILE UPLOAD ======================
uploaded = st.file_uploader("Upload dataset (CSV): SensorDumpData_ARIMAX_ready_2min_full.csv", type=["csv"])
if not uploaded:
    st.info("Please upload the CSV to proceed.")
    st.stop()

# ---------------------- Cached loaders ----------------------
@st.cache_data(ttl="1h", max_entries=10, show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(ttl="1h", max_entries=10, show_spinner=False)
def to_datetime_sorted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    return out

df = load_csv(uploaded)
TIME_COL = "timestamp"
if TIME_COL not in df.columns:
    st.error(f"Missing required column: '{TIME_COL}'")
    st.stop()

df = to_datetime_sorted(df, TIME_COL)

# Determine available parameters in this CSV
def available_params(df_columns):
    avail, unavail = [], []
    for p in PARAMETERS:
        col = PARAM_TO_COLUMN.get(p)
        if col and col in df_columns:
            avail.append(p)
        else:
            unavail.append(p)
    return avail, unavail

avail_params, unavail_params = available_params(df.columns)

# ====================== SELECTORS ======================
sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 1])

with sel_col1:
    st.markdown('<div class="selector-title">Select Shop</div>', unsafe_allow_html=True)
    shop = st.selectbox("", SHOPS, index=0, label_visibility="collapsed")

with sel_col2:
    st.markdown('<div class="selector-title">Select Machine</div>', unsafe_allow_html=True)
    machine = st.selectbox("", MACHINES, index=0, label_visibility="collapsed")

with sel_col3:
    st.markdown('<div class="selector-title">Select Parameter</div>', unsafe_allow_html=True)

    # Disabled chips with tooltips (visual only)
    chips_html = '<div class="chips-row">'
    for p in PARAMETERS:
        if p in avail_params:
            chips_html += f'<div class="chip" title="{p} is available">{p}</div>'
        else:
            chips_html += f'<div class="chip chip-disabled" title="Data not found for {p} in this CSV">{p}</div>'
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # Actual selector shows only AVAILABLE items (to prevent invalid selection)
    if avail_params:
        parameter = st.selectbox("", avail_params, index=0, label_visibility="collapsed")
    else:
        st.warning("No known parameters found in this CSV. Please upload a file containing any of: "
                   f"{', '.join(PARAMETERS)}")
        st.stop()

# Re-render chips marking the active parameter
chips_html_active = '<div class="chips-row">'
for p in PARAMETERS:
