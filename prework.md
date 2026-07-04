
# Traffic Demand Prediction System — Pre-Work

## Goal

Walk into the project review with a **fully working machine learning project**, so the session can focus on understanding the implementation, model evaluation, and application rather than fixing setup issues.

---

# 1. Install Python

Install **Python 3.11 or newer**.

Verify the installation:

```bash
python --version
```

Expected output:

```
Python 3.11.x or higher
```

---

# 2. Create a Virtual Environment

Navigate to the project folder.

```bash
cd Project3
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

---

# 3. Install Required Packages

Install all dependencies.

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly streamlit joblib pickle-mixin
```

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

# 4. Verify Project Structure

Before running the project, ensure the folder structure is as follows:

```text
Project3/
│
├── app.py
├── train.py
├── file.ipynb
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
│   ├── encoders.pkl
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.json
│   └── lightgbm_model.txt
│
└── outputs/
    ├── demand_distribution.png
    ├── feature_importance.png
    ├── hourly_traffic.png
    ├── weather_impact.png
    └── actual_vs_predicted.png
```

---

# 5. Train the Models

Run the training script.

```bash
python train.py
```

This should:

* Generate the dataset (if it does not already exist).
* Perform data preprocessing.
* Apply feature engineering.
* Encode and scale the features.
* Train the Random Forest, XGBoost, and LightGBM models.
* Save the trained models in the `models/` folder.
* Generate evaluation plots in the `outputs/` folder.

---

# 6. Launch the Streamlit Application

Run the web application.

```bash
streamlit run app.py
```

The application should open in your browser.

Verify that you can:

* Select traffic parameters.
* Click **Predict**.
* View the predicted traffic demand.
* See the congestion level.
* View the 12-hour forecast.
* Display all generated visualizations.

---

# 7. Smoke Test

Before the session, verify the following:

* ✅ Dataset loads successfully.
* ✅ No preprocessing errors occur.
* ✅ Models are trained successfully.
* ✅ Model files are created inside the `models/` folder.
* ✅ Output graphs are generated.
* ✅ Streamlit launches successfully.
* ✅ Predictions are displayed without errors.
* ✅ Feature importance chart is visible.
* ✅ Forecast graph updates correctly.

---

# What You DON'T Need

* No external APIs.
* No internet connection after all dependencies are installed.
* No database.
* No cloud services.
* No API keys.

Everything runs locally using the generated dataset and saved machine learning models.


