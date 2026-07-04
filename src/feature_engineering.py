# src/feature_engineering.py

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Peak Hour Flag  (7–9 AM and 5–7 PM)
    df["peak_hour_flag"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )

    # 2. Weekend Flag
    df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)

    # 3. Rush Hour Indicator  (broader: 6–10 AM and 4–8 PM)
    df["rush_hour_indicator"] = df["hour"].apply(
        lambda h: 1 if (6 <= h <= 10) or (16 <= h <= 20) else 0
    )

    # 4. Traffic Density Score  (composite of lanes, signals, large vehicles)
    df["traffic_density_score"] = (
        df["num_lanes"] * 0.4
        - df["traffic_signals"] * 0.1
        - df["large_vehicles"] * 0.05
    ).round(3)

    # 5. Weather Impact Score  (penalise rain/fog/storm)
    weather_weight = {
        "Clear": 1.0, "Cloudy": 0.85, "Rain": 0.6,
        "Fog": 0.45, "Storm": 0.2
    }
    if df["weather_condition"].dtype == object:
        df["weather_impact_score"] = df["weather_condition"].map(weather_weight).fillna(0.7)
    else:
        # Already encoded — approximate from rainfall & humidity
        df["weather_impact_score"] = (
            1.0
            - (df["rainfall"].clip(0, 1) * 0.5)   # scaled rainfall 0–1
            - (df["humidity"].clip(0, 1) * 0.1)
        ).clip(0.1, 1.0)

    # 6. Hour sine/cosine (cyclic encoding)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 7. Day sine/cosine (cyclic encoding)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 8. Interaction features
    df["peak_x_event"]    = df["peak_hour_flag"] * df["event_indicator"]
    df["lanes_x_road"] = df["num_lanes"]
    df["rain_x_peak"]     = df["rainfall"] * df["peak_hour_flag"] \
                             if "rainfall" in df.columns else 0

    # 9. Night flag
    df["night_flag"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

    return df
