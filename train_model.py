import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime

# Load dataset
df = pd.read_csv("india_2000_2024_daily_weather.csv")

# Compute average temperature
df["temp"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
df["wind_speed"] = df["wind_speed_10m_max"]
df["humidity"] = df["precipitation_sum"]

FEATURES = ["temp", "humidity", "wind_speed"]


def compute_dynamic_contamination(city, month, historical_df):

    city_data = historical_df[historical_df["city"] == city]
    if city_data.empty:
        return 0.01  # default

    # Filter by month
    city_data["date"] = pd.to_datetime(city_data["date"])
    month_data = city_data[city_data["date"].dt.month == month]

    X = month_data[FEATURES].dropna().values
    # Rough estimate: proportion of extreme values (top/bottom 1% of temp)
    temp_extremes = X[:, 0]  # temperature
    high_temp = temp_extremes > temp_extremes.quantile(0.99)
    low_temp = temp_extremes < temp_extremes.quantile(0.01)
    dynamic_cont = (high_temp.sum() + low_temp.sum()) / len(temp_extremes)

    return max(dynamic_cont, 0.01)  # minimum 1%
