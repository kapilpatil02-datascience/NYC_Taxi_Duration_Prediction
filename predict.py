import pandas as pd
import joblib
import numpy as np


# ==========================
# Load Saved Pipeline
# ==========================

feature_pipeline = joblib.load(
    "models/feature_pipeline.pkl"
)
print("Pipeline loaded successfully")

# ==========================
# Load Saved Model
# ==========================

model = joblib.load(
    "models/xgboost_model.pkl"
)


# ==========================
# Example New Taxi Ride
# ==========================

new_trip = pd.DataFrame(
    {
        "id": ["test_trip"],

        "vendor_id": [1],

        "pickup_datetime": [
            "2016-03-15 08:30:00"
        ],

        "dropoff_datetime": [
            "2016-03-15 08:45:00"
        ],

        "passenger_count": [2],

        "pickup_longitude": [-73.982154],
        "pickup_latitude": [40.767937],

        "dropoff_longitude": [-73.964630],
        "dropoff_latitude": [40.765602],

        "store_and_fwd_flag": ["N"]
    }
)

# ==========================
# Feature Engineering
# ==========================

new_trip_processed = (
    feature_pipeline.transform(
        new_trip
    )
)

print(
    "Processed Shape:",
    new_trip_processed.shape
)
print('new_trip_processed')


print("Processed Shape:", new_trip_processed.shape)

feature_names = (
    feature_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

print("Feature Count:", len(feature_names))

for i, f in enumerate(feature_names):
    print(i, f)

# ==========================
# Prediction
# ==========================

prediction_log = (
    model.predict(
        new_trip_processed
    )
)

prediction_seconds = (
    np.expm1(prediction_log)
)

print(
    f"Predicted Trip Duration: "
    f"{prediction_seconds[0]:.2f} seconds"
)

