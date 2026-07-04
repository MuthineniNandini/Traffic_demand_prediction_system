# src/model_training.py

import pandas as pd
import numpy as np
import pickle, os, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (r2_score, mean_squared_error,
                              mean_absolute_error)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)

# ── Metric helper ──────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  R²   : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  MAPE : {mp:.2f}%")
    return {"model": name, "r2": r2, "rmse": rmse, "mae": mae, "mape": mp}


# ── Main training pipeline ─────────────────────────────────────────────────────
def train_all(df: pd.DataFrame, target: str = "traffic_demand"):
    feature_cols = [c for c in df.columns
                    if c not in [target, "timestamp", "geohash_raw"]]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    results = []

    # ── 1. Random Forest ────────────────────────────────────────────────────────
    print("\n[1/3] Training Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)

    # ── 2. XGBoost ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training XGBoost …")
    xgb_params = {
        "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": 8, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1,
        "reg_lambda": 1.0, "random_state": 42,
        "tree_method": "hist", "n_jobs": -1,
        "eval_metric": "rmse", "early_stopping_rounds": 50,
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    results.append(evaluate("XGBoost", y_test, xgb_model.predict(X_test)))
    xgb_model.save_model("models/xgboost_model.json")

    # ── 3. LightGBM ─────────────────────────────────────────────────────────────
    print("\n[3/3] Training LightGBM …")
    lgb_params = {
        "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": -1, "num_leaves": 127,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1,
        "verbose": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    results.append(evaluate("LightGBM", y_test, lgb_model.predict(X_test)))
    lgb_model.booster_.save_model("models/lightgbm_model.txt")

    # ── 4. Ensemble (55% LGB + 45% XGB) ────────────────────────────────────────
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ens_pred  = 0.55 * lgb_pred + 0.45 * xgb_pred
    results.append(evaluate("Ensemble (55% LGB + 45% XGB)", y_test, ens_pred))

    # Save ensemble weights
    with open("models/ensemble_weights.pkl", "wb") as f:
        pickle.dump({"lgb": 0.55, "xgb": 0.45}, f)

    # Summary table
    res_df = pd.DataFrame(results)
    print("\n\n" + "="*55)
    print("  FINAL RESULTS SUMMARY")
    print("="*55)
    print(res_df.to_string(index=False))
    res_df.to_csv("models/results.csv", index=False)

    return rf, xgb_model, lgb_model, X_test, y_test, feature_cols
