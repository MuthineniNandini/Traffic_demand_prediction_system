# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])

    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # --- Missing value handling ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "timestamp"]

    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # --- Outlier detection & capping (IQR method) ---
    target = "traffic_demand"
    for col in ["temperature", "humidity", "rainfall", "large_vehicles"]:
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(Q1, Q3)

    print(f"\nCleaned shape: {df.shape}")
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                      encoders: dict = None, scaler=None):
    """
    fit=True  → fit new encoders/scaler (training)
    fit=False → use provided encoders/scaler (inference)
    """
    df = df.copy()
    cat_cols = ["road_type", "weather_condition", "geohash"]
    num_scale = ["temperature", "humidity", "rainfall",
                 "large_vehicles", "traffic_signals",
                 "nearby_landmarks", "num_lanes"]

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        scaler = StandardScaler()
        df[num_scale] = scaler.fit_transform(df[num_scale])
        os.makedirs("models", exist_ok=True)
        with open("models/encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col].astype(str))
        df[num_scale] = scaler.transform(df[num_scale])

    return df, encoders, scaler
