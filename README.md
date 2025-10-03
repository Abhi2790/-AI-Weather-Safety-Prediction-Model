# 🌦 AI Weather Safety Prediction Model

This project uses AI & anomaly detection to predict whether it is safe to travel to a given location based on live weather conditions and geolocation data.

---

## The model integrates:

- Isolation Forest (AI anomaly detection)

- Weather API (live weather data)

- Geopy (latitude & longitude extraction)

- MongoDB/PostgreSQL (optional, for storing results)

- Power BI dashboard (visualization of predictions)

---

## 🚀 Features

- ✅ Fetch live weather data for any city.

- ✅ AI model predicts Safe / Unsafe travel conditions.

- ✅ Uses geo-fencing (lat, lon) with real-time weather.

- ✅ Optional database integration for storing results.

- ✅ Power BI dashboard support for visualization.

---

## 🛠 Tech Stack

- Python 3.10+

- Scikit-learn (Isolation Forest)

- Geopy (location → coordinates)

- WeatherAPI / OpenWeather API (live data)

- Joblib (save & load models)

- MongoDB/PostgreSQL (optional storage)

- Power BI (dashboard visualization)

---


## 📂 Project Structure

```
AI-Weather-Safety/
│── train_model.py          # Train Isolation Forest on historical weather data
│── predict_location.py     # Full pipeline: input city → prediction
│── historical_weather.csv  # Historical dataset (training data)
│── weather_geo_model.pkl   # Saved trained model
│── README.md               # Project documentation
```

---
## Requirements
- Weather API 
- geopy
- scikit-learn
- joblib
- pandas

---

## 📊 Training the Model

- Run the training script on historical weather data:

- python train_model.py

. This generates a trained model file:

- weather_geo_model.pkl

---

## 🌐 Running Prediction

. To predict if a city is safe to travel:

- python predict_location.py


Example:
```
Enter city name: Delhi
✅ Safe to travel in Delhi. (Conditions normal)

🔑 API Key Setup

Get a free API key from:

WeatherAPI
 (Recommended)

or OpenWeatherMap

Add your API key in predict_location.py:

API_KEY = "your_api_key_here"
```

---

## 📦 Database Integration (Optional)

. You can save predictions (city, weather values, safe/unsafe result) into MongoDB/PostgreSQL for long-term analysis and      connect it with Power BI.

---

## 📊 Power BI Dashboard

- Import stored predictions into Power BI.

## Visualize:

- Safe vs Unsafe travel locations

- Weather anomalies by city

- Time trends of anomalies

---

## 🖼 Flowchart
```
   ┌─────────────┐
   │   User       │
   │ Input City   │
   └─────┬───────┘
         │
         ▼
 ┌──────────────────┐
 │ Geocoding (Lat,Lon)│
 └─────┬────────────┘
       │
       ▼
 ┌──────────────────────┐
 │ Weather API (Live Data)│
 └─────┬────────────────┘
       │
       ▼
 ┌──────────────────────┐
 │  Preprocess Features │
 └─────┬────────────────┘
       │
       ▼
 ┌────────────────────────────┐
 │ AI Model (Isolation Forest)│
 └─────┬──────────────────────┘
       │
 ┌─────▼──────────┐
 │ Safe or Unsafe │
 │   Prediction   │
 └─────┬──────────┘
       │
       ▼
 ┌───────────────────┐
 │ Store in Database │
 │ (MongoDB/Postgres)│
 └───────────────────┘
       │
       ▼
 ┌───────────────────┐
 │ Power BI Dashboard│
 └───────────────────┘
```

---

## ✅ Future Improvements

- Add forecast support (next 7 days safety prediction).

- Integrate AQI & Pollution Data.

- Build mobile app frontend (Flutter / React Native).

- Deploy as a REST API (FastAPI/Flask).

## 👨‍💻 Author

Developed by [Abhishek Kumar] 🚀
MCA (AI/ML) Student | AI & Data Science Enthusiast
