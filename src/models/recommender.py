"""
Recommendation engine utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import (
    KNN_ALGORITHM,
    KNN_METRIC,
    KNN_NEIGHBORS,
)


def build_knn_model(
    latent_features: np.ndarray,
    metric: str = KNN_METRIC,
    algorithm: str = KNN_ALGORITHM,
) -> NearestNeighbors:
    """
    Train a KNN model on latent song embeddings.

    Args:
        latent_features: Latent vectors from the encoder.
        metric: Distance metric.
        algorithm: KNN search algorithm.

    Returns:
        Trained NearestNeighbors model.
    """

    knn = NearestNeighbors(
        metric=metric,
        algorithm=algorithm,
    )

    knn.fit(latent_features)

    return knn


def recommend_tracks(
    track_index: int,
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
    knn: NearestNeighbors,
    n_neighbors: int = KNN_NEIGHBORS,
    n_recommendations: int = 9,
) -> pd.DataFrame:
    """
    Recommend tracks similar to a selected track.

    Parameters
    ----------
    track_index:
        Index of the selected track.

    dataframe:
        Processed dataframe.

    latent_features:
        Latent vectors.

    knn:
        Trained KNN model.

    Returns
    -------
    DataFrame containing recommended tracks.
    """

    if track_index < 0 or track_index >= len(dataframe):
        raise ValueError("Invalid track index.")

    track_vector = latent_features[track_index].reshape(1, -1)

    _, indices = knn.kneighbors(
        track_vector,
        n_neighbors=n_neighbors,
    )

    indices = indices.flatten()[1:]

    recommendations = dataframe.iloc[indices]

    recommendations = recommendations.drop_duplicates(
        subset=[
            "track_name",
            "artists",
        ]
    )

    recommendations = recommendations.head(
        n_recommendations
    )

    columns = [
        "track_name",
        "artists",
        "track_genre",
        "track_id",
    ]

    return recommendations[columns]


def recommend_by_name(
    track_name: str,
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
    knn: NearestNeighbors,
    n_neighbors: int = KNN_NEIGHBORS,
    n_recommendations: int = 9,
) -> pd.DataFrame:
    """
    Recommend songs using the track name.
    """

    matches = dataframe[
        dataframe["track_name"].str.lower()
        == track_name.lower()
    ]

    if matches.empty:
        raise ValueError(
            f"Track '{track_name}' not found."
        )

    track_index = matches.index[0]

    return recommend_tracks(
        track_index=track_index,
        dataframe=dataframe,
        latent_features=latent_features,
        knn=knn,
        n_neighbors=n_neighbors,
        n_recommendations=n_recommendations,
    )


def get_track_details(
    dataframe: pd.DataFrame,
    track_index: int,
) -> pd.Series:
    """
    Return metadata for a selected track.
    """

    if track_index < 0 or track_index >= len(dataframe):
        raise ValueError("Invalid track index.")

    return dataframe.iloc[track_index]