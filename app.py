import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic



NYC_LOCATIONS = {
    "Times Square": (40.7580, -73.9855),
    "Central Park": (40.7829, -73.9654),
    "Empire State Building": (40.7484, -73.9857),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Wall Street": (40.7064, -74.0094),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Penn Station": (40.7506, -73.9935),
    "Madison Square Garden": (40.7505, -73.9934),
    "Upper East Side": (40.7736, -73.9566),
    "Upper West Side": (40.7870, -73.9754),
    "SoHo": (40.7233, -74.0030),
    "Chinatown": (40.7158, -73.9970),
    "Battery Park": (40.7033, -74.0170),
    "Yankee Stadium": (40.8296, -73.9262)
}


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NYC Taxi Intelligence Dashboard",
    page_icon="🚕",
    layout="wide"
)

# =====================================================
# LOAD MODEL + PIPELINE
# =====================================================

try:
    model = joblib.load("models/xgboost_model.pkl")
    feature_pipeline = joblib.load("models/feature_pipeline.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =====================================================

# CUSTOM CSS

# =====================================================

st.markdown(
    """ <style>
.stApp {
    background-color: #0F172A;
}

.hero-title {
    font-family:'Google Sans', sans-serif !important;
    text-align:center;
    color:#FFFFFF;
    font-size:48px;
    font-weight:bold;
}

.hero-subtitle {
    font-family: 'Google Sans', sans-serif;
    text-align:center;
    color:yellow;
    font-size:20px;
    margin-bottom:20px;
}

.moving-car {

    position: relative;

    font-size: 40px;

    animation: taxiMove 5s linear infinite;

    white-space: nowrap;

    width: fit-content;
}

@keyframes taxiMove {

    100% {
        left: -30%;
    }

    0% {
        left: 100%;
    }
}

</style>
""",
    unsafe_allow_html=True
)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
    """ <div class="hero-title">
NYC Taxi Intelligence Dashboard </div>
<div class="hero-subtitle">
    Predict Taxi Trip Duration Using Machine Learning
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="moving-car">
        🚕  🚕  🚕 'Madamji Otp??'
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("🚕 Trip Planner")

    pickup_location = st.selectbox(
        "📍 Pickup Location",
        NYC_LOCATIONS
)

    dropoff_location = st.selectbox(
        "🏁 Dropoff Location",
        NYC_LOCATIONS
)

    passenger_count = st.slider(
        "Passenger Count",
        min_value=1,
        max_value=6,
        value=2
    )

    pickup_datetime = st.datetime_input(
        "Pickup Time",
        datetime.now()
    )

    predict_btn = st.button(
        "🚕 Predict Duration"
    )


# =====================================================
# GEOCODING
# =====================================================

@st.cache_data
def get_location(place):

    geolocator = Nominatim(
        user_agent="nyc_taxi_project"
    )

    return geolocator.geocode(
        place + ", New York"
    )

geolocator = Nominatim(user_agent="nyc_taxi_project_v1")

pickup = None
dropoff = None

try:
    pickup = get_location(
    pickup_location
)

    dropoff = get_location(
    dropoff_location
)
    
except Exception as e:
    st.error(f"Geocoding Error: {e}")

# Temporarily add:

st.write("Pickup Input:", pickup_location)
st.write("Dropoff Input:", dropoff_location)



# =====================================================
# MAP
# =====================================================

if pickup and dropoff:
    st.subheader("🗺️ Map")

    m = folium.Map(
        location=[pickup.latitude, pickup.longitude],
        zoom_start=11
    )

    folium.Marker(
        [pickup.latitude, pickup.longitude],
        popup="Pickup"
    ).add_to(m)

    folium.Marker(
        [dropoff.latitude, dropoff.longitude],
        popup="Dropoff"
    ).add_to(m)

    folium.PolyLine(
        [
            [pickup.latitude, pickup.longitude],
            [dropoff.latitude, dropoff.longitude]
        ],
        color="yellow",
        weight=6
    ).add_to(m)

    st_folium(
        m,
        width=None,
        height=500
    )
else:
    st.warning(
        "Unable to locate one of the addresses."
    )

# =====================================================
# PREDICTION
# =====================================================

if predict_btn and pickup and dropoff:
    try:
        input_data = pd.DataFrame(
            {
                "vendor_id": [1],
                "pickup_datetime": [pickup_datetime],
                "passenger_count": [passenger_count],
                "pickup_longitude": [pickup.longitude],
                "pickup_latitude": [pickup.latitude],
                "dropoff_longitude": [dropoff.longitude],
                "dropoff_latitude": [dropoff.latitude],
                "store_and_fwd_flag": ["N"]
            }
        )

        processed_data = feature_pipeline.transform(input_data)
        prediction_log = model.predict(processed_data)
        prediction_seconds = np.expm1(prediction_log)[0]
        prediction_minutes = prediction_seconds / 60

        st.divider()
        st.subheader("📊 Trip Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "⏰ Duration",
                f"{prediction_minutes:.1f} min"
            )

        with col2:

            distance_km = geodesic(
                (
                    pickup.latitude,
                    pickup.longitude
                ),
                (
                    dropoff.latitude,
                    dropoff.longitude
                )
            ).km

            st.metric(
                "📍 Distance",
                f"{distance_km:.2f} km"
            )

        with col3:
            rush_hour = (
                7 <= pickup_datetime.hour <= 10
                or 16 <= pickup_datetime.hour <= 19
            )
            st.metric(
                "🚦 Traffic",
                "High" if rush_hour else "Normal"
            )

        st.success(
            f"Estimated Trip Duration: {prediction_seconds:.2f} seconds"
        )
    except Exception as e:
        st.error(
            f"Prediction failed: {e}"
        )
