# Prepocessing (Pipeline)

from sklearn.pipeline import Pipeline

from src.feature_engineering import (TimeFeatureTransformer, DistanceTransformer, TrafficFeatureTransformer, LocationFeatureTransformer, ClusterFeatureTransformer)

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (StandardScaler, OneHotEncoder)



numerical_columns = [
    'vendor_id',
    'passenger_count',

    'pickup_longitude',
    'pickup_latitude',

    'dropoff_longitude',
    'dropoff_latitude',

    'pickup_hour',
    'pickup_month',
    'pickup_weekday',

    'distance_km',

    'is_weekend',
    'is_rush_hour',
    'is_night',
    'is_working_hour',

    'lat_diff',
    'lon_diff',
    'pickup_cluster',
    'dropoff_cluster'
]

categorical_columns = [
    'store_and_fwd_flag',
    'pickup_day'
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num", 
            StandardScaler(),
            numerical_columns
        ),
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore'

            ),
            categorical_columns
        )
    ],
    remainder='drop'
)


feature_pipeline = Pipeline(
    [
        (
            "time_features",
            TimeFeatureTransformer()
        ),

        (
            "traffic_features",
            TrafficFeatureTransformer()
        ),

        (
            "distance_features",
            DistanceTransformer()
        ),

        (
            "location_features",
            LocationFeatureTransformer()
        ),
        (
            "cluster_features",
            ClusterFeatureTransformer()
        ),

        (
            "preprocessing",
            preprocessor
        )
    ]
)