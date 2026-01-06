
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go

st.set_page_config(page_title="AMTR_AVG Forecast (ARIMA)", layout="wide")

st.title("🔧 AMTR_AVG (Motor Current Avg) — 3-Day Forecast")
st.caption("Upload the CSV: SensorDumpData_ARIMAX_ready_2min_full.csv")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.stop()

# ---- Load & prepare ----
df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

time_col = "timestamp"
value_col = "5001_P0421_M_MTR_AMTR_AVG"

# Basic checks
required = {time_col, value_col}
if not required.issubset(set(df.columns)):
    st.error(f"❌ Required columns missing. Need: {required}")
    st.stop()

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)

series = pd.to_numeric(df[value_col], errors="coerce")
valid_idx = series.dropna().index
series = series.dropna()
series.index = df.loc[valid_idx, time_col]

freq = pd.infer_freq(series.index) or "2T"
full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
series = series.reindex(full_index).interpolate("time")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("⚙️ Controls")
    baseline = st.number_input("Baseline threshold", value=float(series.quantile(0.95)))
    forecast_days = st.slider("Forecast horizon (days)", 1, 7, 3)
    # Small AIC search
    max_pq = 2
    st.write("ARIMA grid: p,q=0..2, d∈{0,1}")

# ---- ARIMA (small grid search by AIC) ----
best_aic = np.inf
best_order = None
best_fit = None
for p in range(0, max_pq+1):
    for d in (0,1):
        for q in range(0, max_pq+1):
            try:
                fit = ARIMA(series, order=(p,d,q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order, best_fit = fit.aic, (p,d,q), fit
            except Exception:
                pass

steps = (24*60//2) * forecast_days  # 2-min cadence
future_index = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq),
                             periods=steps, freq=freq)
forecast = best_fit.forecast(steps=steps)
forecast = pd.Series(forecast.values, index=future_index)

# ---- Plotly (scrollable with range slider) ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical",
                         mode="lines", line=dict(color="steelblue")))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                         mode="lines", line=dict(color="orange", dash="dash")))
fig.add_hline(y=baseline, line=dict(color="red", dash="dot"), annotation_text=f"Baseline={baseline:.2f}")

fig.update_layout(
    title=f"AMTR_AVG — Historical & {forecast_days}-Day Forecast (Best ARIMA {best_order})",
    xaxis_title="Time", yaxis_title="AMTR_AVG",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True)),  # <-- scrollable slider
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ---- Baseline check + session store for Maintenance page ----
exceeds = float(np.nanmax(forecast.values)) > baseline
if exceeds:
    st.error("⚠️ Forecast crosses baseline — maintenance recommended.")
else:
    st.success("✅ Forecast within baseline.")

if "maintenance_items" not in st.session_state:
    st.session_state["maintenance_items"] = []
if exceeds:
    st.session_state["maintenance_items"].append({
        "Component": "Motor Current (AMTR_AVG)",
        "Best_ARIMA": str(best_order),
        "Baseline": float(baseline),
        "MaxForecast": float(np.nanmax(forecast.values)),
        "FirstExceedTS": str(forecast.index[np.argmax(forecast.values)])
    })

# Optional: download forecast
csv_bytes = pd.DataFrame({"timestamp": forecast.index, "AMTR_AVG_forecast": forecast.values}).to_csv(index=False).encode()
st.download_button("⬇️ Download Forecast CSV", csv_bytes, file_name="amtr_avg_forecast.csv", mime="text/csv")

st.caption("Tip: Use the range slider under the chart to scroll; drag/zoom for details.")
