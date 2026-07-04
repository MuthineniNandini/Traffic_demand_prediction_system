
# DATASET DOCUMENTATION

## Traffic Demand Prediction System

**Project Title:** Traffic Demand Prediction System Using Ensemble Machine Learning

**Dataset Version:** 1.0

**Dataset Type:** Synthetic Machine Learning Dataset

**Author:** Nandini Muthineni

**Records:** Approximately 120,000

**File Name:** `traffic_data.csv`

---

# 1. Introduction

The Traffic Demand Prediction dataset was developed specifically for this project because publicly available traffic datasets either lacked sufficient features or did not align with the project's objectives. The dataset simulates realistic traffic conditions by combining temporal, geographical, weather, road infrastructure, and contextual information.

The dataset was generated programmatically using Python (`generate_dataset.py`) to ensure consistency, scalability, and realistic traffic patterns. Statistical distributions, probability-based sampling, and mathematical functions were used to model real-world traffic behavior.

The target variable, **traffic_demand**, represents the estimated number of vehicles passing through a road segment in one hour.

---

# 2. Dataset Summary

| Property          | Value                   |
| ----------------- | ----------------------- |
| Dataset Name      | Traffic Demand Dataset  |
| File Name         | traffic_data.csv        |
| Records           | 120,000                 |
| Features          | 15                      |
| Target Variable   | traffic_demand          |
| File Format       | CSV                     |
| Generation Method | Python Script           |
| Missing Values    | None                    |
| Data Type         | Structured Tabular Data |

---

# 3. Dataset Features

| Column            | Data Type   | Description                          |
| ----------------- | ----------- | ------------------------------------ |
| timestamp         | DateTime    | Date and time of traffic observation |
| geohash           | Categorical | Geographic zone identifier           |
| day_of_week       | Integer     | Day number (0 = Monday, 6 = Sunday)  |
| hour              | Integer     | Hour of the day (0–23)               |
| road_type         | Categorical | Type of road                         |
| num_lanes         | Integer     | Number of traffic lanes              |
| traffic_signals   | Integer     | Number of nearby traffic signals     |
| large_vehicles    | Integer     | Number of buses/trucks               |
| temperature       | Float       | Ambient temperature (°C)             |
| humidity          | Float       | Relative humidity (%)                |
| rainfall          | Float       | Rainfall intensity (mm)              |
| weather_condition | Categorical | Weather category                     |
| nearby_landmarks  | Integer     | Number of nearby landmarks           |
| event_indicator   | Binary      | Special event indicator              |
| traffic_demand    | Integer     | Target variable (vehicles/hour)      |

---

# 4. Data Generation Methodology

The dataset was generated using Python by combining deterministic calculations with probabilistic distributions.

The following libraries were used:

* Pandas
* NumPy
* Random
* Datetime

Random seeds were fixed using:

```python
np.random.seed(42)
random.seed(42)
```

This ensures reproducibility, meaning the same dataset can be regenerated consistently.

---

# 5. Feature Generation

## 5.1 Timestamp

Hourly timestamps were generated starting from January 1, 2022.

```python
start_date = datetime(2022,1,1)
timestamps = start_date + timedelta(hours=i)
```

This creates realistic hourly observations over multiple years.

---

## 5.2 Geographical Zones

Fifty synthetic geographical regions were created.

Example:

```
gh_000

gh_001

gh_002

...

gh_049
```

Each record was randomly assigned to one of these zones.

---

## 5.3 Road Types

Road categories included:

* Highway
* Urban
* Suburban
* Rural
* Expressway

Each road type was assigned different probabilities to reflect real-world distributions.

---

## 5.4 Number of Lanes

Possible values:

```
1
2
3
4
6
```

Road capacity increases with lane count.

---

## 5.5 Weather Conditions

Five weather categories were simulated.

| Weather | Meaning            |
| ------- | ------------------ |
| Clear   | Normal traffic     |
| Cloudy  | Slight reduction   |
| Rain    | Moderate reduction |
| Fog     | Heavy reduction    |
| Storm   | Severe reduction   |

---

## 5.6 Temperature

Temperature follows a normal distribution.

Mean

```
25°C
```

Standard Deviation

```
8°C
```

Range

```
-5°C to 45°C
```

---

## 5.7 Humidity

Humidity values were generated uniformly.

Range

```
30% – 100%
```

---

## 5.8 Rainfall

Rainfall depends on weather.

Clear weather generates almost zero rainfall, while Rain and Storm produce significantly higher rainfall values using an exponential distribution.

---

## 5.9 Nearby Landmarks

Random integers between

```
0 – 20
```

represent shopping malls, schools, offices, hospitals, and commercial centers.

Higher landmark counts generally increase traffic demand.

---

## 5.10 Event Indicator

Binary feature

```
0 → Normal day

1 → Special Event
```

Special events increase expected traffic.

---

# 6. Target Variable Generation

The target variable **traffic_demand** was calculated using a custom mathematical model.

The model considers:

* Base traffic volume
* Morning rush hours
* Evening rush hours
* Weekend effects
* Road type
* Number of lanes
* Weather
* Rainfall
* Temperature
* Traffic signals
* Heavy vehicles
* Nearby landmarks
* Special events
* Random noise

Finally,

```
traffic_demand=Base Traffic + Hour Effect + Road Effect + Weather Effect + Event Effect + Random Noise
```

The final values are clipped between

```
50

and

8000
```

vehicles per hour.

---

# 7. Dataset Characteristics

The dataset demonstrates several realistic traffic behaviors.

### Morning Peak

High demand around

```
8–9 AM
```

---

### Evening Peak

Maximum traffic around

```
5–7 PM
```

---

### Weather Impact

Heavy rain and storms reduce traffic demand.

---

### Road Capacity

Highways and expressways accommodate significantly larger traffic volumes.

---

### Event Influence

Special events increase expected traffic.

---

### Weekend Effect

Traffic patterns differ on weekends compared to weekdays.

---

# 8. Advantages of the Dataset

* Large-scale dataset suitable for machine learning
* Balanced representation of road types
* Realistic temporal patterns
* Weather-aware traffic simulation
* Suitable for regression algorithms
* Reproducible due to fixed random seeds
* Supports feature engineering experiments

---

# 9. Limitations

Although realistic, the dataset is synthetic and has certain limitations:

* No live GPS data
* No accident information
* No road construction details
* No seasonal holiday effects
* No real sensor measurements

Future work can incorporate live traffic APIs and IoT sensor data to improve realism.

---

# 10. Usage in the Project

This dataset is used throughout the project for:

* Data preprocessing
* Feature engineering
* Model training
* Model validation
* Ensemble learning
* Traffic demand prediction
* Streamlit web application testing
* Performance evaluation

---

# 11. Conclusion

The generated Traffic Demand Dataset provides a comprehensive and realistic foundation for building machine learning models for traffic forecasting. By incorporating temporal, environmental, infrastructural, and contextual variables, the dataset enables accurate traffic demand prediction and supports the development of intelligent transportation solutions.

---

