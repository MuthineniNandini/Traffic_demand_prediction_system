# train.py  (run this first)

import pandas as pd
import os
from data.generate_dataset import *          # generates CSV if absent
from src.data_preprocessing import load_and_clean, encode_and_scale
from src.feature_engineering import engineer_features
from src.model_training import train_all
from src.ensemble import plot_all

# ── 1. Load & clean ────────────────────────────────────────────────────────────
if not os.path.exists("data/traffic_data.csv"):
    exec(open("data/generate_dataset.py").read())

df_raw = load_and_clean("data/traffic_data.csv")

# ── 2. Feature engineering (before encoding — needs string weather) ────────────
df_fe = engineer_features(df_raw)

# ── 3. Encode & scale ──────────────────────────────────────────────────────────
df_enc, encoders, scaler = encode_and_scale(df_fe, fit=True)

# Create interaction feature AFTER encoding
df_enc["lanes_x_road"] = (
    df_enc["num_lanes"] * df_enc["road_type"]
)

# Check that there are no string columns left
print(df_enc.select_dtypes(include=["object"]).columns)

# ── 4. Train models ────────────────────────────────────────────────────────────
rf, xgb_model, lgb_model, X_test, y_test, feature_cols = train_all(df_enc)

# ── 5. Visualisations ──────────────────────────────────────────────────────────
plot_all(df_raw, rf, xgb_model, lgb_model, X_test, y_test, feature_cols)

print("\n🎉  Training complete. Models saved to models/")
