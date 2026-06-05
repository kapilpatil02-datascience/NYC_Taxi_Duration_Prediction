# Feature Engineering


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans

# TimeFeatureTransformer

class TimeFeatureTransformer(BaseEstimator,TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X['pickup_datetime'] = pd.to_datetime(
            X['pickup_datetime']
        )

        X['pickup_hour'] = (
            X['pickup_datetime'].dt.hour
        )

        X['pickup_day'] = (
            X['pickup_datetime'].dt.day_name()
        )

        X['pickup_month'] = (
            X['pickup_datetime'].dt.month
        )

        X['pickup_weekday'] = (
            X['pickup_datetime'].dt.weekday
        )

        X['is_weekend'] = (
            X['pickup_weekday']
            .isin([5, 6])
            .astype(int)
        )

        X['is_rush_hour'] = (
            X['pickup_hour']
            .isin([7, 8, 9, 16, 17, 18, 19])
            .astype(int)
        )

        X['is_night'] = (
            X['pickup_hour']
            .isin([0, 1, 2, 3, 4, 5])
            .astype(int)
        )

        X['is_working_hour'] = (
            X['pickup_hour']
            .between(9, 18)
            .astype(int)
        )

        return X
    

# DistanceTransformer

class DistanceTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        X = X.copy()

        pickup_lat = np.radians(X['pickup_latitude'])

        pickup_lon = np.radians(X['pickup_longitude'])

        dropoff_lat = np.radians(X['dropoff_latitude'])

        dropoff_lon = np.radians(X['dropoff_longitude'])

        dlat = dropoff_lat - pickup_lat

        dlon = dropoff_lon - pickup_lon


        a = (np.sin(dlat /2) **2 
             +
             np.cos(pickup_lat) 
             *
             np.cos(dropoff_lat)
             *
             (np.sin(dlon /2) **2 )
        )

        c = 2 * np.arcsin(np.sqrt(a))

        earth_radius_km = 6371

        X['distance_km'] = (
            earth_radius_km * c
        )

        return X
        
# RushHourTransformer & LocationFeatureTransformer

class TrafficFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        X = X.copy()

        X["is_rush_hour"] = (
            X["pickup_hour"]
            .isin([7, 8, 9, 16, 17, 18, 19])
            .astype(int)
        )

        X["is_night"] = (
            X["pickup_hour"]
            .isin([0, 1, 2, 3, 4, 5])
            .astype(int)
        )

        X["is_working_hour"] = (
            X["pickup_hour"]
            .between(9, 18)
            .astype(int)
        )

        return X
    

# LocationFeatureTransformer

class LocationFeatureTransformer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self


    def transform(self, X):

        X = X.copy()

        X["lat_diff"] = (
            X["dropoff_latitude"]
            - X["pickup_latitude"]
        )

        X["lon_diff"] = (
            X["dropoff_longitude"]
            - X["pickup_longitude"]
        )

        return X
        

class ClusterFeatureTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        n_clusters=20
    ):
        self.n_clusters = n_clusters

    def fit(
        self,
        X,
        y=None
    ):

        pickup_coords = X[
            [
                "pickup_latitude",
                "pickup_longitude"
            ]
        ]

        self.pickup_cluster_model = (
            MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                batch_size=10000
            )
        )

        self.pickup_cluster_model.fit(
            pickup_coords
        )

    
        dropoff_coords = X[
            [
                "dropoff_latitude",
                "dropoff_longitude"
            ]
        ]

        self.dropoff_cluster_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=10000
        )

        self.dropoff_cluster_model.fit(
            dropoff_coords
        )

        return self


    def transform(
        self,
        X
    ):

        X = X.copy()

        X["pickup_cluster"] = (
            self.pickup_cluster_model.predict(
                X[
                    [
                        "pickup_latitude",
                        "pickup_longitude"
                    ]
                ]
            )
        )

        X["dropoff_cluster"] = (
        self.dropoff_cluster_model.predict(
            X[
                [
                    "dropoff_latitude",
                    "dropoff_longitude"
                ]
            ]
        )
    )

        return X