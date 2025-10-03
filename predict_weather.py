import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Load historical dataset
historical_df = pd.read_csv("india_2000_2024_daily_weather.csv")

# Compute features
historical_df["temp"] = (historical_df["temperature_2m_max"] + historical_df["temperature_2m_min"]) / 2
historical_df["wind_speed"] = historical_df["wind_speed_10m_max"]
historical_df["humidity"] = historical_df["precipitation_sum"]
historical_df["date"] = pd.to_datetime(historical_df["date"])

FEATURES = ["temp", "humidity", "wind_speed"]

def compute_dynamic_contamination(city, month, historical_df, min_cont=0.02):

    city_data = historical_df[historical_df["city"] == city].copy()
    if city_data.empty:
        city_data = historical_df.copy()  # fallback

    month_data = city_data[city_data["date"].dt.month == month]
    if month_data.empty:
        month_data = city_data.copy()  # fallback to city all months

    X = month_data[FEATURES].dropna().values
    if len(X) == 0:
        return min_cont

    temp_extremes = X[:, 0]
    high_temp = temp_extremes > np.quantile(temp_extremes, 0.99)
    low_temp = temp_extremes < np.quantile(temp_extremes, 0.01)
    dynamic_cont = (high_temp.sum() + low_temp.sum()) / len(temp_extremes)
    return max(dynamic_cont, min_cont)

def get_weather(city, API_KEY):

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        raise Exception(f"Error fetching weather: {data.get('message', 'Unknown error')}")

    temp = data['main']['temp']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    lat = data['coord']['lat']
    lon = data['coord']['lon']

    return temp, humidity, wind_speed, lat, lon

def predict_safety(city, API_KEY):
    """Predict weather safety for any city"""
    month = datetime.now().month

    # Dynamic contamination
    contamination = compute_dynamic_contamination(city, month, historical_df)

    # Filter historical data for city/month
    city_month_data = historical_df[(historical_df["city"] == city) &
                                    (historical_df["date"].dt.month == month)]
    if city_month_data.empty:
        city_month_data = historical_df.copy()

    # Compute mean/std for normalization
    mean_vals = city_month_data[FEATURES].mean()
    std_vals = city_month_data[FEATURES].std().replace(0, 1)  # avoid division by zero

    # Normalize historical data
    X_norm = (city_month_data[FEATURES] - mean_vals) / std_vals

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_norm.values)

    # Fetch live weather
    temp, humidity, wind_speed, lat, lon = get_weather(city, API_KEY)

    # Normalize live data
    new_data = np.array([[
        (temp - mean_vals["temp"]) / std_vals["temp"],
        (humidity - mean_vals["humidity"]) / std_vals["humidity"],
        (wind_speed - mean_vals["wind_speed"]) / std_vals["wind_speed"]
    ]])

    # Predict
   # prediction = model.predict(new_data)
   # safety_status = "normal/safe 🌤️" if prediction[0] == 0.4 else "unusual/anomalous ⚠️"

    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        #"safety_status": safety_status,
        "dynamic_contamination": contamination
    }

# ----------------- Example Usage -----------------
if __name__ == "__main__":
    API_KEY = "3871538a465755176c5de4d6627c8d12"# API Key
    city = input("Enter city name: ")
    try:
        result = predict_safety(city, API_KEY)
        print(f"Weather Safety Report for {result['city']}:")
        print(f"  Location: ({result['lat']}, {result['lon']})")
        print(f"  Temperature: {result['temperature']}°C")
        print(f"  Humidity: {result['humidity']}%")
        print(f"  Wind Speed: {result['wind_speed']} m/s")
        print(f"  Status: {result['safety_status']}")
        print(f"  Dynamic contamination used: {result['dynamic_contamination']:.4f}")
    except Exception as e:
        print("Error:", e)
