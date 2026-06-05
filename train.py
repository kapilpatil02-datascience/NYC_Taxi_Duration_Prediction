# Train the Model

import pandas as pd

import numpy as np

import time

from sklearn.model_selection import train_test_split

from src.preprocessing import feature_pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import HistGradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

from sklearn.model_selection import RandomizedSearchCV

import joblib



## Load Data

results = []

DATA_PATH = "data/raw/train.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


X = df.drop(columns = ['trip_duration'])

y = np.log1p(df['trip_duration'])


# Train_Test_Split

X_train, X_test, y_train, y_test = (train_test_split( X, y, random_state = 42, test_size = 0.2))

# Apply Pipeline

X_train_processed = (
    feature_pipeline.fit_transform(X_train)
)

feature_names = (
    feature_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

# print("\nFeature Names:")
# for i, f in enumerate(feature_names):
#     print(i, f)

# print("\nX_train_processed Shape:")
# print(X_train_processed.shape)


# print("Number of Features:", len(feature_names))

# for f in feature_names:
#     print(f)


X_test_processed = (
    feature_pipeline.transform(X_test)
)

print(
    "Training Features:",
    X_train_processed.shape
)


# =====================================================
# Model Evaluation Function
# =====================================================

def evaluate_model(model, model_name):

    # Train model
    model.fit(
        X_train_processed,
        y_train
    )

    # Prediction
    predictions = model.predict(
        X_test_processed
    )

    # Convert back from log scale
    predictions_original = np.expm1(
        predictions
    )

    y_test_original = np.expm1(
        y_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test_original,
        predictions_original
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_original,
            predictions_original
        )
    )

    r2 = r2_score(
        y_test_original,
        predictions_original
    )

    # Store Results
    results.append(
        {
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }
    )


# =====================================================
# Linear Regression
# =====================================================

evaluate_model(
    LinearRegression(),
    "Linear Regression"
)

# =====================================================
# HistGradientBoosting
# =====================================================

evaluate_model(
    HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    ),
    "HistGradientBoosting"
)

# =====================================================
# XGBRegressor
# =====================================================
evaluate_model(
    XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost"
)

# =====================================================
# LGBMRegressor
# =====================================================

evaluate_model(
    LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    ),
    "LightGBM"
)


results_df = pd.DataFrame(results)

print(results_df)



# ---------------------------------------------------------------------------------------


# ============================
# Train XGBoost for Feature Importance
# ============================


best_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

best_model.fit(
    X_train_processed,
    y_train
)

feature_names = (
    feature_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)


importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": best_model.feature_importances_
    }
)

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(
    importance_df.head(20)
)


## --------------------------------------------------------------------------------------

# ===================================
# Hyperparameter Tuning
# ===================================

# param_grid = {
#     "n_estimators": [200, 300, 500],
#     "max_depth": [4, 6, 8, 10],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "subsample": [0.7, 0.8, 1.0],
#     "colsample_bytree": [0.7, 0.8, 1.0]
# }

# xgb_model = XGBRegressor(
#     random_state=42
# )

# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_grid,
#     n_iter=10,
#     scoring="neg_mean_absolute_error",
#     cv=3,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )

# random_search.fit(
#     X_train_processed,
#     y_train
# )

# print("Best Parameters:")
# print(random_search.best_params_)


# ==========================
# Final Model Training
# ==========================

final_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    final_model,
    "models/xgboost_model.pkl"
)

joblib.dump(
    feature_pipeline,
    "models/feature_pipeline.pkl"
)

print("Model Saved Successfully!")
print("Pipeline Saved Successfully!")

loaded_pipeline = joblib.load(
    "models/feature_pipeline.pkl"
)

print(
    "Loaded Pipeline Shape:",
    loaded_pipeline.transform(
        X.head(1)
    ).shape
)



# This is exactly how production ML systems work.
# New Taxi Ride
#        ↓
# feature_pipeline.pkl
#        ↓
# 26 Features
#        ↓
# xgboost_model.pkl
#        ↓
# Predicted Duration