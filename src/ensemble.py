# src/ensemble.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score
import os

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")


def plot_all(df_raw, rf, xgb_model, lgb_model,
             X_test, y_test, feature_cols):

    fig_list = []

    # 1. Traffic Demand Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_raw["traffic_demand"], bins=60, kde=True,
                 color="steelblue", ax=ax)
    ax.set_title("Traffic Demand Distribution", fontsize=14)
    ax.set_xlabel("Vehicles / Hour")
    plt.tight_layout()
    fig.savefig("outputs/demand_distribution.png", dpi=150)
    fig_list.append(fig)

    # 2. Hourly Traffic Analysis
    hourly = df_raw.groupby("hour")["traffic_demand"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=hourly, x="hour", y="traffic_demand",
                 marker="o", color="tomato", ax=ax)
    ax.set_title("Average Traffic Demand by Hour", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Vehicles / Hour")
    ax.set_xticks(range(24))
    plt.tight_layout()
    fig.savefig("outputs/hourly_traffic.png", dpi=150)
    fig_list.append(fig)

    # 3. Weather Impact
    fig, ax = plt.subplots(figsize=(10, 5))
    weather_avg = df_raw.groupby("weather_condition")["traffic_demand"] \
                        .mean().sort_values(ascending=False)
    sns.barplot(x=weather_avg.index, y=weather_avg.values,
                palette="coolwarm", ax=ax)
    ax.set_title("Average Traffic Demand by Weather Condition", fontsize=14)
    ax.set_xlabel("Weather")
    ax.set_ylabel("Avg Vehicles / Hour")
    plt.tight_layout()
    fig.savefig("outputs/weather_impact.png", dpi=150)
    fig_list.append(fig)

    # 4. Correlation Heatmap
    num_df = df_raw.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    fig.savefig("outputs/correlation_heatmap.png", dpi=150)
    fig_list.append(fig)

    # 5. Feature Importance (LightGBM)
    import lightgbm as lgb
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgb_model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=importance, x="importance", y="feature",
                palette="viridis", ax=ax)
    ax.set_title("Top 15 Feature Importances (LightGBM)", fontsize=14)
    plt.tight_layout()
    fig.savefig("outputs/feature_importance.png", dpi=150)
    fig_list.append(fig)

    # 6. Actual vs Predicted (Ensemble)
    lgb_pred  = lgb_model.predict(X_test)
    xgb_pred  = xgb_model.predict(X_test)
    ens_pred  = 0.55 * lgb_pred + 0.45 * xgb_pred
    sample_n  = 500
    idx       = np.random.choice(len(y_test), sample_n, replace=False)
    y_s, p_s  = np.array(y_test)[idx], ens_pred[idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(y_s, p_s, alpha=0.4, color="royalblue", s=15)
    mn, mx = min(y_s.min(), p_s.min()), max(y_s.max(), p_s.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Fit")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Actual vs Predicted  (R²={r2_score(y_test, ens_pred):.4f})")
    axes[0].legend()

    axes[1].plot(range(200), np.array(y_test)[:200], label="Actual",
                 color="steelblue", lw=1.5)
    axes[1].plot(range(200), ens_pred[:200], label="Predicted",
                 color="tomato", lw=1.5, linestyle="--")
    axes[1].set_title("Actual vs Predicted — First 200 Samples")
    axes[1].set_xlabel("Sample Index"); axes[1].set_ylabel("Traffic Demand")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig("outputs/actual_vs_predicted.png", dpi=150)
    fig_list.append(fig)

    print("\nAll plots saved to outputs/")
    return fig_list
