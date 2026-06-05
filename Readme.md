# NYC Taxi Trip Duration Prediction

## Project Overview

This project predicts the duration of New York City taxi trips using machine learning.

The objective is to estimate trip duration based on:

* Pickup location
* Dropoff location
* Pickup time
* Passenger count
* Vendor information

## Dataset

NYC Taxi Trip Duration Dataset

Features include:

* Pickup and dropoff coordinates
* Pickup datetime
* Passenger count
* Store and forward flag
* Vendor ID

Target Variable:

* Trip Duration (seconds)

---

## Feature Engineering

Custom sklearn transformers were created:

### TimeFeatureTransformer

Extracted:

* Pickup Hour
* Pickup Month
* Pickup Weekday
* Pickup Day

### TrafficFeatureTransformer

Created:

* is_rush_hour
* is_night
* is_working_hour
* is_weekend

### DistanceTransformer

Calculated:

* Haversine Distance (distance_km)

### LocationFeatureTransformer

Created:

* lat_diff
* lon_diff

### ClusterFeatureTransformer

Created:

* pickup_cluster
* dropoff_cluster

using MiniBatchKMeans.

---

## Machine Learning Pipeline

Raw Data
→ Feature Engineering
→ Preprocessing
→ Model Training

Pipeline Components:

* StandardScaler
* OneHotEncoder
* Custom Transformers

---

## Models Evaluated

* Linear Regression
* HistGradientBoosting
* XGBoost
* LightGBM

---

## Best Model

XGBoost

Best Parameters:

* n_estimators = 500
* max_depth = 6
* learning_rate = 0.1
* subsample = 0.8
* colsample_bytree = 0.8

---

## Final Results

MAE: 298.29

RMSE: 3200.41

R²: 0.033

---

## Model Deployment Assets

Saved Files:

* models/xgboost_model.pkl
* models/feature_pipeline.pkl

Prediction Script:

* predict.py

Example Output:

Predicted Trip Duration: 661.71 seconds

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* Joblib

---

## Project Structure

nyc_taxi_project/

data/

models/

* xgboost_model.pkl
* feature_pipeline.pkl

src/

* feature_engineering.py
* preprocessing.py

train.py
predict.py
README.md
