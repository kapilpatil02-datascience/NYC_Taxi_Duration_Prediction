import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==========================================

# Page Config

# ==========================================

st.set_page_config(
page_title="NYC Taxi Intelligence",
page_icon="🚕",
layout="wide"
)

# ==========================================

# Custom CSS

# ==========================================

st.markdown(
""" <style>
.main {
    background-color: #0F172A;
}

.hero {
    text-align:center;
    padding:20px;
    color:white;
}

.title {
    font-size:50px;
    font-weight:bold;
    color:#FFC300;
}

.subtitle{
    font-size:22px;
    color:white;
    margin-top:10px;
}

.skyline {
    font-size:70px;
    text-align:center;
}

.moving-car {
    animation: moveCar 8s linear infinite;
    font-size:40px;
    position: relative;
}

@keyframes moveCar {
    from { left:-40%; }
    to { left:100%; }
}

</style>
""",
unsafe_allow_html=True
)

# ==========================================

# Hero Section

# ==========================================

st.markdown(
""" <div class='hero'> <div class='skyline'>
🏙️ 🏢 🏙️ </div>
    <div class='title'>
    NYC Taxi Duration Intelligence
    </div>

</div>
""",
unsafe_allow_html=True
)

st.markdown(
""" <div class='moving-car'>
🚕 🚕 🚕 </div>
""",
unsafe_allow_html=True
)

st.divider()

# ==========================================

# User Input Form

# ==========================================

st.subheader("Trip Details")

col1, col2 = st.columns(2)

with col1:
    pickup_latitude = st.number_input(
        "Pickup Latitude",
        value=40.767937
    )

    pickup_longitude = st.number_input(
        "Pickup Longitude",
        value=-73.982154
    )

    passenger_count = st.number_input(
        "Passenger Count",
        value=2
    )

with col2:
    dropoff_latitude = st.number_input(
        "Dropoff Latitude",
        value=40.765602
    )

    dropoff_longitude = st.number_input(
        "Dropoff Longitude",
        value=-73.964630
    )

    pickup_datetime = st.datetime_input(
        "Pickup Time",
        datetime.now()
    )

predict_btn = st.button(
    "🚕 Predict Trip Duration"
)

# Load saved preprocessing pipeline and model
try:
    feature_pipeline = joblib.load("models/feature_pipeline.pkl")
    model = joblib.load("models/xgboost_model.pkl")
except FileNotFoundError:
    st.error("Model files not found. Run train.py first to create models/feature_pipeline.pkl and models/xgboost_model.pkl.")
    feature_pipeline = None
    model = None

if predict_btn and feature_pipeline is not None and model is not None:
    input_data = pd.DataFrame(
        {
            "vendor_id": [1],
            "pickup_datetime": [pickup_datetime],
            "passenger_count": [passenger_count],
            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "store_and_fwd_flag": ["N"]
        }
    )

    try:
        processed = feature_pipeline.transform(input_data)
        prediction_log = model.predict(processed)
        prediction_seconds = np.expm1(prediction_log)[0]
        prediction_minutes = prediction_seconds / 60.0

        st.success(
            f"Predicted Trip Duration: {prediction_seconds:.2f} seconds ({prediction_minutes:.1f} minutes)"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
