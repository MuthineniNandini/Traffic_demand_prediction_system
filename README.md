# 🚦 Traffic Demand Prediction System using Ensemble Machine Learning

## 📌 Project Overview

Traffic congestion has become one of the biggest challenges in modern cities. Accurate traffic demand prediction helps traffic authorities optimize signal timing, reduce congestion, improve route planning, and enhance overall transportation efficiency.

This project develops an **Ensemble Machine Learning-based Traffic Demand Prediction System** that predicts the number of vehicles expected on a road segment per hour using road characteristics, weather conditions, temporal information, and surrounding contextual factors.

The system combines the predictions of **LightGBM** and **XGBoost** models to improve prediction accuracy and provides an interactive **Streamlit dashboard** for real-time traffic demand estimation.

---

# 🎯 Objectives

* Predict hourly traffic demand.
* Analyze the impact of weather and road conditions on traffic flow.
* Compare multiple machine learning models.
* Improve prediction accuracy using ensemble learning.
* Deploy the prediction model through an interactive Streamlit web application.

---

# 🛠 Technologies Used

| Category           | Technologies                    |
| ------------------ | ------------------------------- |
| Language           | Python 3                        |
| Data Processing    | Pandas, NumPy                   |
| Machine Learning   | Scikit-learn, XGBoost, LightGBM |
| Data Visualization | Plotly, Matplotlib              |
| Deployment         | Streamlit                       |
| Model Storage      | Pickle                          |

---

# 📂 Project Structure

```
Traffic_Demand_Prediction/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── generate_dataset.py
│   └── traffic_data.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── ensemble.py
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost_model.json
│   ├── lightgbm_model.txt
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── ensemble_weights.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── demand_distribution.png
│   ├── feature_importance.png
│   ├── hourly_traffic.png
│   └── weather_impact.png
│


---

# 📊 Dataset

The project uses a **synthetically generated dataset** consisting of approximately **120,000 records**.

The dataset simulates realistic traffic scenarios using multiple influencing factors.

### Dataset Features

* Timestamp
* Geohash (Location)
* Day of Week
* Hour
* Road Type
* Number of Lanes
* Traffic Signals
* Large Vehicles
* Temperature
* Humidity
* Rainfall
* Weather Condition
* Nearby Landmarks
* Event Indicator

### Target Variable

* **Traffic Demand (Vehicles per Hour)**

---

# ⚙ Feature Engineering

The project creates additional features to improve model performance.

These include:

* Peak Hour Flag
* Weekend Flag
* Rush Hour Indicator
* Traffic Density Score
* Weather Impact Score
* Hour Sin/Cos Encoding
* Day Sin/Cos Encoding
* Peak × Event Interaction
* Lane × Road Interaction
* Rain × Peak Interaction
* Night Flag

---

# 🤖 Machine Learning Models

The following models are trained:

* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

The final prediction is generated using an **ensemble approach**:

```
Final Prediction =
0.55 × LightGBM Prediction +
0.45 × XGBoost Prediction
```

---

# 📈 Visualizations

The project automatically generates:

* Demand Distribution
* Hourly Traffic Analysis
* Weather Impact Analysis
* Correlation Heatmap
* Feature Importance
* Actual vs Predicted Comparison

These graphs are stored inside the **outputs/** folder.

---

# 🌐 Streamlit Application

The Streamlit dashboard allows users to:

* Select road type
* Choose weather conditions
* Set date and time
* Specify traffic signals
* Enter number of lanes
* Configure temperature, humidity and rainfall
* Select nearby landmarks
* Enable/disable special events

The application displays:

* Predicted Traffic Demand
* Congestion Level
* Peak Traffic Alert
* LightGBM Prediction
* XGBoost Prediction
* Ensemble Prediction
* 12-Hour Traffic Forecast
* Analysis Graphs

---

# 🚀 Installation

Clone the repository:

```bash
git clone <repository_url>
cd Traffic_Demand_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶ Generate Dataset

```bash
python data/generate_dataset.py
```

---

# ▶ Train Models

```bash
python train.py
```

This generates:

* Trained models
* Encoders
* Scaler
* Performance plots

---

# ▶ Run the Application

```bash
streamlit run app.py
```

---

# 📊 Results

The ensemble model provides accurate traffic demand predictions by combining the strengths of LightGBM and XGBoost.

The generated dashboard enables interactive exploration of traffic scenarios and provides useful congestion insights.

---

# 🔮 Future Enhancements

* Integration with real-time traffic APIs
* Live weather API integration
* GPS-based location prediction
* Traffic heatmap visualization
* Deep Learning (LSTM/Transformer) models
* Explainable AI using SHAP
* Cloud deployment
* Mobile application support

---

# 👩‍💻 Author

**Nandini Muthineni**

B.Tech Graduate

Machine Learning | Data Science | Python Developer

---

# 📜 License

This project has been developed for academic and learning purposes.
