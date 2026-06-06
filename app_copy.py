import streamlit as st
import pandas as pd
import numpy as np
import joblib

from datetime import datetime

import folium
from streamlit_folium import st_folium

from geopy.distance import geodesic

st.write("HELLO")
print("folium working")

# =====================================================

# PAGE CONFIG

# =====================================================

st.set_page_config(
page_title="NYC Taxi Intelligence",
page_icon="🚕",
layout="wide"
)

# =====================================================

# SESSION STATE

# =====================================================

if "pickup_point" not in st.session_state:
    st.session_state.pickup_point = None

if "dropoff_point" not in st.session_state:
    st.session_state.dropoff_point = None

# =====================================================

# LOAD MODEL

# =====================================================

try:

    model = joblib.load(
        "models/xgboost_model.pkl"
    )

    feature_pipeline = joblib.load(
        "models/feature_pipeline.pkl"
    )
    st.write("Models Loaded Successfully")

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()

# =====================================================

# HEADER

# =====================================================

st.title(
"🚕 NYC Taxi Intelligence"
)

st.caption(
"Click once for Pickup • Click again for Dropoff"
)

# =====================================================

# USER INPUTS

# =====================================================

colA, colB, colC, colD = st.columns(4)

with colA:

    passenger_count = st.slider(
    "Passengers",
        1,
        6,
        2
    )


with colB:

    pickup_datetime = st.datetime_input(
        "Pickup Time",
        datetime.now()
    )

with colC:

    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    if st.button("🚕 Predict"):
        st.session_state.predict_clicked = True


with colD:


    reset_btn = st.button(
        "🔄 Reset"
    )

# =====================================================

# RESET

# =====================================================

if reset_btn:

    st.session_state.pickup_point = None
    st.session_state.dropoff_point = None
    st.session_state.predict_clicked = False

# =====================================================

# MAP

# =====================================================

m = folium.Map(
location=[40.7580, -73.9855],
zoom_start=11
)

# Pickup Marker

if st.session_state.pickup_point:


    folium.Marker(
        st.session_state.pickup_point,
        popup="Pickup",
        icon=folium.Icon(
            color="green"
        )
    ).add_to(m)

# Dropoff Marker

if st.session_state.dropoff_point:

    folium.Marker(
        st.session_state.dropoff_point,
        popup="Dropoff",
        icon=folium.Icon(
            color="red"
        )
    ).add_to(m)

# Route

if (
    st.session_state.pickup_point
    and
    st.session_state.dropoff_point
    ):

    folium.PolyLine(
        [
            st.session_state.pickup_point,
            st.session_state.dropoff_point
        ],
        color="yellow",
        weight=6
    ).add_to(m)

# Display Map

map_data = st_folium(
m,
height=600,
use_container_width=True
)

# =====================================================

# CLICK HANDLING

# =====================================================

if map_data and map_data.get("last_clicked"):

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    if st.session_state.pickup_point is None:

        st.session_state.pickup_point = [
            lat,
            lon
        ]

    elif st.session_state.dropoff_point is None:

        st.session_state.dropoff_point = [
            lat,
            lon
        ]
#```

# =====================================================

# STATUS

# =====================================================

col1, col2 = st.columns(2)

with col1:

    if st.session_state.pickup_point:

        st.success(
            "📍 Pickup Selected"
        )

with col2:

    if st.session_state.dropoff_point:

        st.success(
            "🏁 Dropoff Selected"
        )

# =====================================================

# PREDICTION

# =====================================================

if (
    st.session_state.predict_clicked
    and
    st.session_state.pickup_point
    and
    st.session_state.dropoff_point
):
    st.write("Predict Button Clicked")

    try:

        pickup_lat = st.session_state.pickup_point[0]
        pickup_lon = st.session_state.pickup_point[1]

        dropoff_lat = st.session_state.dropoff_point[0]
        dropoff_lon = st.session_state.dropoff_point[1]

        input_data = pd.DataFrame(
            {
                "vendor_id": [1],
                "pickup_datetime": [pickup_datetime],
                "passenger_count": [passenger_count],
                "pickup_longitude": [pickup_lon],
                "pickup_latitude": [pickup_lat],
                "dropoff_longitude": [dropoff_lon],
                "dropoff_latitude": [dropoff_lat],
                "store_and_fwd_flag": ["N"]
            }
        )

        processed = feature_pipeline.transform(
            input_data
        )

        prediction_log = model.predict(
            processed
        )

        prediction_seconds = np.expm1(
            prediction_log
        )[0]

        prediction_minutes = (
            prediction_seconds / 60
        )

        distance_km = geodesic(
            (
                pickup_lat,
                pickup_lon
            ),
            (
                dropoff_lat,
                dropoff_lon
            )
        ).km

        st.divider()

        st.subheader(
            "📊 Trip Insights"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "⏰ Duration",
                f"{prediction_minutes:.1f} min"
            )

        with c2:

            st.metric(
                "📍 Distance",
                f"{distance_km:.2f} km"
            )

        with c3:

            rush_hour = (
                7 <= pickup_datetime.hour <= 10
                or
                16 <= pickup_datetime.hour <= 19
            )

            st.metric(
                "🚦 Traffic",
                "High" if rush_hour else "Normal"
            )
            processed = feature_pipeline.transform(
                input_data
            )
            st.write("Input Data")
            st.dataframe(input_data)
            processed = feature_pipeline.transform(
                input_data
            )
            st.write(
                "Processed Shape:",
                processed.shape
            )

    except Exception as e:

        st.error(
            f"Prediction Failed: {type(e).__name__}: {e}"
        )

        import traceback

        st.code(
            traceback.format_exc()
        )

st.write("Pickup Hour:", pickup_datetime.hour)