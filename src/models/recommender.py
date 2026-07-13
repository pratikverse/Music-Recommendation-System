"""
Recommendation engine for TuneMatch.
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
    Train the KNN recommender.
    """

    knn = NearestNeighbors(
        metric=metric,
        algorithm=algorithm,
    )

    knn.fit(latent_features)

    return knn


def rank_candidates(
    candidates: pd.DataFrame,
    similarities: np.ndarray,
) -> pd.DataFrame:
    """
    Rank recommendation candidates.
    """

    ranked = candidates.copy()

    ranked["similarity"] = similarities

    if "popularity" in ranked.columns:

        ranked["popularity_score"] = (
            ranked["popularity"] / 100
        )

    else:

        ranked["popularity_score"] = 0

    ranked["ranking_score"] = (
        0.85 * ranked["similarity"]
        +
        0.15 * ranked["popularity_score"]
    )

    ranked = ranked.sort_values(
        "ranking_score",
        ascending=False,
    )

    return ranked


def diversify_artists(
    recommendations: pd.DataFrame,
    max_per_artist: int = 2,
) -> pd.DataFrame:
    """
    Prevent one artist from dominating recommendations.
    """

    final = []

    artist_counter = {}

    for _, row in recommendations.iterrows():

        artist = row["artists"]

        count = artist_counter.get(
            artist,
            0,
        )

        if count >= max_per_artist:
            continue

        artist_counter[artist] = count + 1

        final.append(row)

    return pd.DataFrame(final)


def recommend_tracks(
    track_index: int,
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
    knn: NearestNeighbors,
    n_neighbors: int = 50,
    n_recommendations: int = 10,
) -> pd.DataFrame:
    """
    Recommend tracks.
    """

    if track_index >= len(dataframe):
        raise ValueError("Invalid track index.")

    query = latent_features[
        track_index
    ].reshape(1, -1)

    distances, indices = knn.kneighbors(
        query,
        n_neighbors=n_neighbors,
    )

    distances = distances.flatten()[1:]

    indices = indices.flatten()[1:]

    candidates = dataframe.iloc[
        indices
    ].copy()

    similarities = 1 - distances

    candidates = rank_candidates(
        candidates,
        similarities,
    )

    candidates = candidates.drop_duplicates(
        subset=[
            "track_name",
            "artists",
        ]
    )

    candidates = diversify_artists(
        candidates,
        max_per_artist=2,
    )

    return candidates.head(
        n_recommendations
    )


def get_track_details(
    dataframe: pd.DataFrame,
    track_index: int,
) -> pd.Series:
    """
    Return metadata for one track.
    """

    return dataframe.iloc[
        track_index
    ]