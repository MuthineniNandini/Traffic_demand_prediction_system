# data/generate_dataset.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 120_000

# --- Time features ---
start_date = datetime(2022, 1, 1)
timestamps = [start_date + timedelta(hours=i) for i in range(N)]
timestamps = pd.Series(timestamps)

day_of_week = timestamps.dt.dayofweek          # 0=Mon, 6=Sun
hour        = timestamps.dt.hour

# --- Location ---
geohash_pool = [f"gh_{i:03d}" for i in range(50)]
geohash      = np.random.choice(geohash_pool, N)

# --- Road features ---
road_type_pool  = ["Highway", "Urban", "Suburban", "Rural", "Expressway"]
road_type       = np.random.choice(road_type_pool, N, p=[0.2, 0.35, 0.25, 0.1, 0.1])
num_lanes       = np.random.choice([1, 2, 3, 4, 6], N, p=[0.1, 0.3, 0.3, 0.2, 0.1])
traffic_signals = np.random.randint(0, 10, N)
large_vehicles  = np.random.randint(0, 50, N)

# --- Weather ---
weather_pool    = ["Clear", "Cloudy", "Rain", "Fog", "Storm"]
weather_cond    = np.random.choice(weather_pool, N, p=[0.4, 0.25, 0.2, 0.1, 0.05])
temperature     = np.random.normal(25, 8, N).clip(-5, 45)
humidity        = np.random.uniform(30, 100, N)
rainfall        = np.where(
    np.isin(weather_cond, ["Rain", "Storm"]),
    np.random.exponential(15, N),
    np.random.exponential(0.5, N)
).clip(0, 150)

# --- Context ---
nearby_landmarks = np.random.randint(0, 20, N)
event_indicator  = np.random.choice([0, 1], N, p=[0.85, 0.15])

# --- Target: Traffic Demand (vehicles/hour) ---
def compute_demand(hour, dow, road, lanes, weather, temp, rainfall,
                   signals, large_veh, landmarks, events):
    base = 500

    # Hour pattern
    morning_rush  = np.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2) * 1500
    evening_rush  = np.exp(-0.5 * ((hour - 17.5) / 1.5) ** 2) * 1800
    night_dip     = np.where((hour >= 0) & (hour < 5), -300, 0)
    hour_effect   = morning_rush + evening_rush + night_dip

    # Day effect
    weekend_effect = np.where(dow >= 5, -400, 200)

    # Road type multiplier
    road_map    = {"Highway": 2.5, "Expressway": 2.8, "Urban": 1.8,
                   "Suburban": 1.3, "Rural": 0.6}
    road_mult   = np.vectorize(road_map.get)(road)
    road_effect = base * (road_mult - 1)

    # Lane capacity
    lane_effect = lanes * 120

    # Weather penalty
    weather_map    = {"Clear": 0, "Cloudy": -50, "Rain": -200,
                      "Fog": -300, "Storm": -600}
    weather_effect = np.vectorize(weather_map.get)(weather)
    rain_penalty   = -rainfall * 3

    # Temp comfort
    temp_effect    = -np.abs(temp - 22) * 5

    # Event boost
    event_effect   = events * 600

    # Landmarks
    landmark_effect = landmarks * 30

    # Signals slow traffic
    signal_effect   = -signals * 20

    # Large vehicles slow traffic
    lv_effect = -large_veh * 4

    demand = (base + hour_effect + weekend_effect + road_effect
              + lane_effect + weather_effect + rain_penalty
              + temp_effect + event_effect + landmark_effect
              + signal_effect + lv_effect
              + np.random.normal(0, 80, len(hour)))

    return demand.clip(50, 8000)


traffic_demand = compute_demand(
    hour.values, day_of_week.values, road_type, num_lanes,
    weather_cond, temperature, rainfall, traffic_signals,
    large_vehicles, nearby_landmarks, event_indicator
)

df = pd.DataFrame({
    "timestamp":        timestamps,
    "geohash":          geohash,
    "day_of_week":      day_of_week.values,
    "hour":             hour.values,
    "road_type":        road_type,
    "num_lanes":        num_lanes,
    "traffic_signals":  traffic_signals,
    "large_vehicles":   large_vehicles,
    "temperature":      temperature.round(1),
    "humidity":         humidity.round(1),
    "rainfall":         rainfall.round(2),
    "weather_condition": weather_cond,
    "nearby_landmarks": nearby_landmarks,
    "event_indicator":  event_indicator,
    "traffic_demand":   traffic_demand.round(0).astype(int),
})

df.to_csv("data/traffic_data.csv", index=False)
print(f"Dataset saved: {df.shape}")
print(df.head())
