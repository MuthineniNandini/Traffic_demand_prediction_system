# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Demand Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a69ac);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .alert-high   { background: #c0392b; border-radius: 8px;
                    padding: 12px; color: white; font-weight: 600; }
    .alert-medium { background: #e67e22; border-radius: 8px;
                    padding: 12px; color: white; font-weight: 600; }
    .alert-low    { background: #27ae60; border-radius: 8px;
                    padding: 12px; color: white; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("models/xgboost_model.json")

    lgb_model_b = lgb.Booster(model_file="models/lightgbm_model.txt")

    return encoders, scaler, xgb_model, lgb_model_b


encoders, scaler, xgb_model, lgb_booster = load_models()


# ── Prediction helper ──────────────────────────────────────────────────────────
def predict_demand(inputs: dict) -> dict:
    dt       = inputs["datetime"]
    hour     = dt.hour
    dow      = dt.weekday()

    weather_impact_map = {
        "Clear": 1.0, "Cloudy": 0.85, "Rain": 0.6,
        "Fog": 0.45, "Storm": 0.2
    }

    # Build raw feature dict
    row = {
        "geohash":              inputs["geohash"],
        "day_of_week":          dow,
        "hour":                 hour,
        "road_type":            inputs["road_type"],
        "num_lanes":            inputs["num_lanes"],
        "traffic_signals":      inputs["traffic_signals"],
        "large_vehicles":       inputs["large_vehicles"],
        "temperature":          inputs["temperature"],
        "humidity":             inputs["humidity"],
        "rainfall":             inputs["rainfall"],
        "weather_condition":    inputs["weather"],
        "nearby_landmarks":     inputs["landmarks"],
        "event_indicator":      int(inputs["event"]),
        # Engineered features
        "peak_hour_flag":       1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
        "weekend_flag":         1 if dow >= 5 else 0,
        "rush_hour_indicator":  1 if (6 <= hour <= 10) or (16 <= hour <= 20) else 0,
        "traffic_density_score": (inputs["num_lanes"] * 0.4
                                   - inputs["traffic_signals"] * 0.1
                                   - inputs["large_vehicles"] * 0.05),
        "weather_impact_score": weather_impact_map.get(inputs["weather"], 0.7),
        "hour_sin":             np.sin(2 * np.pi * hour / 24),
        "hour_cos":             np.cos(2 * np.pi * hour / 24),
        "day_sin":              np.sin(2 * np.pi * dow / 7),
        "day_cos":              np.cos(2 * np.pi * dow / 7),
        "peak_x_event":         (1 if (7<=hour<=9) or (17<=hour<=19) else 0)
                                 * int(inputs["event"]),
        "lanes_x_road":         inputs["num_lanes"],
        "rain_x_peak":          inputs["rainfall"]
                                 * (1 if (7<=hour<=9) or (17<=hour<=19) else 0),
        "night_flag":           1 if hour < 6 or hour >= 22 else 0,
    }

    df_row = pd.DataFrame([row])

    # Encode categoricals
    for col in ["road_type", "weather_condition", "geohash"]:
        le = encoders[col]
        val = df_row[col].astype(str)
        # Handle unseen labels gracefully
        df_row[col] = val.apply(
            lambda x: le.transform([x])[0]
            if x in le.classes_ else le.transform([le.classes_[0]])[0]
        )

    # Scale numerics
    num_scale = ["temperature", "humidity", "rainfall", "large_vehicles",
                 "traffic_signals", "nearby_landmarks", "num_lanes"]
    df_row[num_scale] = scaler.transform(df_row[num_scale])

    # Predict
    lgb_pred = lgb_booster.predict(df_row)[0]
    xgb_pred = xgb_model.predict(df_row)[0]
    ensemble  = 0.55 * lgb_pred + 0.45 * xgb_pred

    # Congestion level
    if ensemble > 4000:
        congestion, cong_pct = "Critical", 95
    elif ensemble > 2500:
        congestion, cong_pct = "High", 75
    elif ensemble > 1500:
        congestion, cong_pct = "Moderate", 50
    else:
        congestion, cong_pct = "Low", 20

    peak_alert = row["peak_hour_flag"] or row["rush_hour_indicator"]

    return {
        "demand":     round(ensemble),
        "lgb":        round(lgb_pred),
        "xgb":        round(xgb_pred),
        "congestion": congestion,
        "cong_pct":   cong_pct,
        "peak_alert": bool(peak_alert),
    }


def forecast_next_hours(base_inputs: dict, hours: int = 12):
    """Generate demand forecast for the next N hours."""
    forecasts = []
    base_dt   = base_inputs["datetime"]
    for i in range(hours):
        inp = base_inputs.copy()
        inp["datetime"] = base_dt.replace(
            hour=(base_dt.hour + i) % 24
        )
        res = predict_demand(inp)
        forecasts.append({
            "hour":   f"{(base_dt.hour + i) % 24:02d}:00",
            "demand": res["demand"]
        })
    return pd.DataFrame(forecasts)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Demand Prediction System")
st.markdown("*Powered by LightGBM + XGBoost Ensemble · Innovexa Catalyst ML Project*")
st.markdown("---")

# Sidebar — Inputs
with st.sidebar:
    st.header("📍 Input Parameters")

    location = st.selectbox("Location (Geohash Zone)",
        [f"gh_{i:03d}" for i in range(0, 50, 5)])

    road_type = st.selectbox("Road Type",
        ["Highway", "Expressway", "Urban", "Suburban", "Rural"])

    weather = st.selectbox("Weather Condition",
        ["Clear", "Cloudy", "Rain", "Fog", "Storm"])

    date_input = st.date_input("Date", datetime.today())
    time_input = st.time_input("Time", datetime.now().replace(second=0, microsecond=0))

    st.markdown("---")
    st.subheader("🔧 Road Parameters")
    num_lanes       = st.slider("Number of Lanes",       1, 6, 3)
    traffic_signals = st.slider("Traffic Signals Nearby", 0, 10, 3)
    large_vehicles  = st.slider("Large Vehicles Count",   0, 50, 10)

    st.markdown("---")
    st.subheader("🌡️ Weather Details")
    temperature = st.slider("Temperature (°C)", -5, 45, 25)
    humidity    = st.slider("Humidity (%)",      30, 100, 65)
    rainfall    = st.slider("Rainfall (mm)",      0, 150, 0)

    st.markdown("---")
    st.subheader("📌 Context")
    landmarks = st.slider("Nearby Landmarks", 0, 20, 5)
    event     = st.checkbox("Special Event Active?", False)

    predict_btn = st.button("🔮 PREDICT", use_container_width=True,
                             type="primary")


# Main panel
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Traffic Demand Forecast")

    if predict_btn:
        dt_combined = datetime.combine(date_input, time_input)

        inputs = {
            "datetime": dt_combined, "geohash": location,
            "road_type": road_type, "weather": weather,
            "num_lanes": num_lanes, "traffic_signals": traffic_signals,
            "large_vehicles": large_vehicles, "temperature": temperature,
            "humidity": humidity, "rainfall": rainfall,
            "landmarks": landmarks, "event": event,
        }

        with st.spinner("Running ensemble model …"):
            result = predict_demand(inputs)

        # ── Metric cards ────────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['demand']:,}</div>
                <div class="metric-label">Vehicles / Hour</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            color = {"Critical":"#c0392b","High":"#e67e22",
                     "Moderate":"#f39c12","Low":"#27ae60"}[result["congestion"]]
            st.markdown(f"""
            <div class="metric-card" style="background:linear-gradient(135deg,{color},#333)">
                <div class="metric-value">{result['congestion']}</div>
                <div class="metric-label">Congestion Level ({result['cong_pct']}%)</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            alert_txt = "⚠️ YES" if result["peak_alert"] else "✅ NO"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{alert_txt}</div>
                <div class="metric-label">Peak Traffic Alert</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Model breakdown
        with st.expander("🔍 Model Breakdown"):
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("LightGBM",        f"{result['lgb']:,} veh/hr")
            bc2.metric("XGBoost",         f"{result['xgb']:,} veh/hr")
            bc3.metric("Ensemble (Final)", f"{result['demand']:,} veh/hr")

        # Next-12-hours forecast chart
        forecast_df = forecast_next_hours(inputs, hours=12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["hour"], y=forecast_df["demand"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#2a69ac", width=3),
            marker=dict(size=8, color="#1e3c72")
        ))
        fig.add_hrect(y0=2500, y1=forecast_df["demand"].max() * 1.1,
                      fillcolor="red", opacity=0.07,
                      annotation_text="High Congestion Zone", annotation_position="top left")
        fig.update_layout(
            title="Traffic Demand — Next 12 Hours",
            xaxis_title="Hour", yaxis_title="Vehicles / Hour",
            template="plotly_dark", height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Fill in parameters on the left and click **PREDICT**")


with col2:
    st.subheader("📈 Analysis Plots")
    tabs = st.tabs(["Demand Dist.", "Hourly", "Weather", "Feature Imp."])

    img_map = {
        0: "outputs/demand_distribution.png",
        1: "outputs/hourly_traffic.png",
        2: "outputs/weather_impact.png",
        3: "outputs/feature_importance.png",
    }
    for i, tab in enumerate(tabs):
        with tab:
            path = img_map[i]
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning("Run `train.py` first to generate plots.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    Innovexa Catalyst · Traffic Demand Prediction System ·
    LightGBM + XGBoost Ensemble · Deployed on Streamlit Cloud
</div>
""", unsafe_allow_html=True)
