
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import StringIO

# Try importing statsmodels; show a helpful message if missing
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATS_OK = True
except Exception as e:
    STATS_OK = False
    IMPORT_ERR = (
        "⚠️ 'statsmodels' (and deps: numpy, scipy, patsy) not available.\n"
        "Please ensure your app's requirements.txt includes pinned versions:\n"
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
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
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
    if p == parameter:
        chips_html_active += f'<div class="chip chip-active" title="{p} is selected">{p}</div>'
    elif p in avail_params:
        chips_html_active += f'<div class="chip" title="{p} is available">{p}</div>'
    else:
        chips_html_active += f'<div class="chip chip-disabled" title="Data not found for {p} in this CSV">{p}</div>'
chips_html_active += "</div>"
st.markdown(chips_html_active, unsafe_allow_html=True)

# ====================== SIDEBAR CONTROLS (Lite) ======================
with st.sidebar:
    st.header("⚙️ Controls (Lite)")
    window_days = st.slider("Model window (days)", 3, 30, DEFAULT_WINDOW_DAYS)
    default_baseline = float(PARAM_BASELINES.get(parameter, 0.0))
    baseline = st.number_input("Baseline (threshold)", value=default_baseline, step=0.1)
    horizon_days = st.slider("Forecast horizon (days)", 1, 7, DEFAULT_HORIZON_DAYS)
    mode = st.radio("Model mode", ["Fast (recommended)", "Accurate"], index=0,
                    help="Fast: small ARIMA grid for low CPU; Accurate adds (1,1,1)")
    st.caption("Downsample: 10-minute cadence; modeling last N days only.")

# ====================== PREP SERIES (windowed, downsampled, cached) ======================
selected_col = PARAM_TO_COLUMN[parameter]

@st.cache_data(show_spinner=False)
def prep_series(df: pd.DataFrame, time_col: str, value_col: str,
                window_days: int, downsample_freq: str) -> pd.Series:
    s = pd.to_numeric(df[value_col], errors="coerce")
    valid_idx = s.dropna().index
    s = s.dropna()
    s.index = df.loc[valid_idx, time_col]

    # limit to last N days to reduce memory/CPU
    end_ts = s.index.max()
    start_ts = end_ts - pd.Timedelta(days=window_days)
    s = s.loc[s.index >= start_ts]

    # resample to lighter cadence (10 minutes)
    s = s.resample(downsample_freq).mean().interpolate("time")
    return s

series = prep_series(df, TIME_COL, selected_col, window_days, DOWNSAMPLE_FREQ)

# ====================== ARIMA (tiny grid, cached) ======================
@st.cache_resource(show_spinner=False)
def fit_best_arima(series: pd.Series, orders: list[tuple[int,int,int]]):
    if not STATS_OK:
        return None, None
    best_aic = np.inf
    best_order, best_fit = None, None
    for (p, d, q) in orders:
        try:
            fit = ARIMA(series, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
        except Exception:
            pass
    return best_order, best_fit

orders = ARIMA_ORDERS_FAST if mode.startswith("Fast") else ARIMA_ORDERS_ACCURATE
best_order, best_fit = fit_best_arima(series, orders)

# ====================== FORECAST (lite horizon) ======================
def build_forecast(best_fit, horizon_days: int, downsample_freq: str, series: pd.Series):
    if not STATS_OK or best_fit is None:
        return None
    # Steps for 10-minute cadence: 6 points/hour, 144 points/day
    steps = 144 * horizon_days
    future_index = pd.date_range(series.index[-1] + pd.Timestamp(0), periods=steps, freq=downsample_freq)
    forecast_vals = best_fit.forecast(steps=steps)
    forecast = pd.Series(forecast_vals.values, index=future_index)
    return forecast

if not STATS_OK:
    st.warning(IMPORT_ERR)
    forecast = None
elif best_fit is None:
    st.warning("ARIMA fitting failed; try Fast mode or reduce window.")
    forecast = None
else:
    forecast = build_forecast(best_fit, horizon_days, DOWNSAMPLE_FREQ, series)

# ====================== PLOT (scrollable, compact) ======================
plot_container = st.container()
with plot_container:
    fig = go.Figure()

    # Historical (solid)
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, name="Historical",
        mode="lines", line=dict(color="steelblue")
    ))

    # Forecast (dashed)
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values, name=f"Forecast ({horizon_days}d)",
            mode="lines", line=dict(color="orange", dash="dash")
        ))

    # Baseline line
    fig.add_hline(y=float(baseline), line=dict(color="#c68c53", width=2),
                  annotation_text=f"Baseline={baseline:.2f}")

    fig.update_layout(
        title=f"{shop} • {machine} • {parameter} — Historical & {horizon_days}-Day Forecast"
              + (f" (ARIMA {best_order})" if best_order else ""),
        xaxis_title="Timestamp",
        yaxis_title="Parameter Value",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        margin=dict(l=40, r=40, t=60, b=10),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use the range slider under the chart to scroll; drag/zoom for details.")

# ====================== MAINTENANCE PANEL (same page) ======================
st.write("---")
st.subheader("🛠 Maintenance Panel")

if forecast is None:
    st.info("No forecast available yet. Upload data, choose parameter, and ensure 'statsmodels' is installed.")
    first_exceed_ts = None
    max_forecast_val = float("nan")
    delta_over_baseline = float("nan")
else:
    exceeds_mask = forecast > float(baseline)
    first_exceed_ts = (str(forecast.index[exceeds_mask.argmax()]) if exceeds_mask.any() else None)
    max_forecast_val = float(np.nanmax(forecast.values))
    delta_over_baseline = max(0.0, max_forecast_val - float(baseline))

status_col, details_col = st.columns([1, 2])
with status_col:
    if forecast is None:
        st.info("ℹ️ Waiting for valid forecast.")
    elif (forecast > float(baseline)).any():
        st.error("⚠ Forecast crosses baseline — schedule maintenance.")
    else:
        st.success("✅ Forecast stays within baseline.")

with details_col:
    st.markdown("**Selected Context**")
    st.write(f"- **Shop**: {shop}")
    st.write(f"- **Machine**: {machine}")
    st.write(f"- **Parameter**: {parameter}")
    st.write(f"- **Baseline**: {baseline:.2f}")
    if first_exceed_ts:
        st.write(f"- **First exceed timestamp**: {first_exceed_ts}")
    else:
        st.write("- **First exceed timestamp**: _None (no crossing in forecast horizon)_")
    st.write(f"- **Max forecast value**: {max_forecast_val if not np.isnan(max_forecast_val) else 'n/a'}")
    st.write(f"- **Delta over baseline (max)**: {delta_over_baseline if not np.isnan(delta_over_baseline) else 'n/a'}")

# Downloads (forecast & schedule) — only when forecast exists
if forecast is not None:
    forecast_df = pd.DataFrame({"timestamp": forecast.index, f"{parameter}_forecast": forecast.values})
    st.download_button(
        "⬇️ Download Forecast CSV",
        forecast_df.to_csv(index=False).encode(),
        file_name=f"{parameter.replace(' ', '_')}_forecast.csv",
        mime="text/csv"
    )

    schedule_df = pd.DataFrame([{
        "Shop": shop,
        "Machine": machine,
        "Parameter": parameter,
        "Baseline": baseline,
        "FirstExceedTS": first_exceed_ts or "",
        "MaxForecast": max_forecast_val,
        "DeltaOverBaseline": delta_over_baseline
    }])
    st.download_button(
        "⬇️ Download Maintenance Schedule (CSV)",
        schedule_df.to_csv(index=False).encode(),
        file_name=f"maintenance_{machine}_{parameter.replace(' ','_')}.csv",
        mime="text/csv"
    )

st.caption("Lite mode: cached data & tiny ARIMA grids keep CPU/memory low on Streamlit Cloud.")
