"""
Recommendation engine for TuneMatch.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import KNN_ALGORITHM, KNN_METRIC
from src.models.genre import infer_genre_families


AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

AUDIO_FEATURE_WEIGHTS = {
    "danceability": 1.10,
    "energy": 1.25,
    "loudness": 0.60,
    "speechiness": 0.70,
    "acousticness": 1.00,
    "instrumentalness": 0.70,
    "liveness": 0.65,
    "valence": 1.15,
    "tempo": 1.05,
}

FEATURE_NORMALIZERS = {
    "tempo": 120.0,
    "loudness": 30.0,
}

INTENT_WEIGHT_PROFILES = {
    "Balanced": {
        "latent_similarity": 0.40,
        "audio_similarity": 0.28,
        "genre_score": 0.20,
        "popularity_score": 0.05,
        "source_support_score": 0.07,
    },
    "Same vibe": {
        "latent_similarity": 0.50,
        "audio_similarity": 0.30,
        "genre_score": 0.10,
        "popularity_score": 0.03,
        "source_support_score": 0.07,
    },
    "Same genre": {
        "latent_similarity": 0.28,
        "audio_similarity": 0.20,
        "genre_score": 0.38,
        "popularity_score": 0.05,
        "source_support_score": 0.09,
    },
    "Discovery": {
        "latent_similarity": 0.34,
        "audio_similarity": 0.24,
        "genre_score": 0.16,
        "popularity_score": 0.02,
        "source_support_score": 0.24,
    },
    "More popular": {
        "latent_similarity": 0.28,
        "audio_similarity": 0.18,
        "genre_score": 0.16,
        "popularity_score": 0.28,
        "source_support_score": 0.10,
    },
    "More energetic": {
        "latent_similarity": 0.32,
        "audio_similarity": 0.30,
        "genre_score": 0.14,
        "popularity_score": 0.04,
        "source_support_score": 0.20,
    },
}

INTENT_AUDIO_FEATURE_ADJUSTMENTS = {
    "Balanced": {},
    "Same vibe": {"energy": 1.1, "valence": 1.1},
    "Same genre": {},
    "Discovery": {"acousticness": 0.9, "speechiness": 0.9},
    "More popular": {},
    "More energetic": {"energy": 1.4, "tempo": 1.3, "danceability": 1.2},
}


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


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_weight_profile(intent: str) -> dict[str, float]:
    """
    Resolve the active hybrid weight profile.
    """

    return INTENT_WEIGHT_PROFILES.get(
        intent,
        INTENT_WEIGHT_PROFILES["Balanced"],
    ).copy()


def get_audio_feature_weights(intent: str) -> dict[str, float]:
    """
    Resolve the active audio-feature weights for an intent.
    """

    weights = AUDIO_FEATURE_WEIGHTS.copy()
    for feature, multiplier in INTENT_AUDIO_FEATURE_ADJUSTMENTS.get(
        intent,
        {},
    ).items():
        weights[feature] = weights.get(feature, 1.0) * multiplier
    return weights


def _normalize_feature_similarity(
    feature: str,
    selected_values: np.ndarray,
    candidate_values: np.ndarray,
) -> np.ndarray:
    """
    Vectorized 0-1 similarity for one audio feature.
    """

    differences = np.abs(
        selected_values - candidate_values
    )
    normalizer = FEATURE_NORMALIZERS.get(feature, 1.0)
    normalized_difference = np.minimum(
        differences / normalizer,
        1.0,
    )
    return 1.0 - normalized_difference


def compute_audio_similarity_scores(
    selected_song: pd.Series,
    candidates: pd.DataFrame,
    intent: str = "Balanced",
) -> np.ndarray:
    """
    Compare songs directly using weighted raw audio features.
    """

    feature_weights = get_audio_feature_weights(intent)
    weighted_similarity = np.zeros(len(candidates))
    total_weight = 0.0

    for feature in AUDIO_FEATURES:
        if feature not in candidates.columns or feature not in selected_song:
            continue

        selected_value = _safe_float(selected_song[feature])
        candidate_values = (
            candidates[feature]
            .fillna(selected_value)
            .to_numpy(dtype=float)
        )

        similarities = _normalize_feature_similarity(
            feature,
            np.full(len(candidates), selected_value, dtype=float),
            candidate_values,
        )

        weight = feature_weights.get(feature, 1.0)
        weighted_similarity += similarities * weight
        total_weight += weight

    if total_weight == 0:
        return np.zeros(len(candidates))

    return weighted_similarity / total_weight


def compute_audio_similarity(
    selected_song: pd.Series,
    candidate_song: pd.Series,
    intent: str = "Balanced",
) -> float:
    """
    Single-row audio similarity helper.
    """

    candidate_df = pd.DataFrame([candidate_song])
    return float(
        compute_audio_similarity_scores(
            selected_song,
            candidate_df,
            intent=intent,
        )[0]
    )


def compute_genre_score(
    selected_song: pd.Series,
    candidate_song: pd.Series,
) -> float:
    """
    Reward exact genre matches and broader genre-family matches.
    """

    selected_genre = str(
        selected_song.get("track_genre", "")
    ).strip().casefold()
    candidate_genre = str(
        candidate_song.get("track_genre", "")
    ).strip().casefold()

    if not selected_genre or not candidate_genre:
        return 0.0

    if selected_genre == candidate_genre:
        return 1.0

    selected_families = infer_genre_families(selected_genre)
    candidate_families = infer_genre_families(candidate_genre)

    if (
        selected_families
        and candidate_families
        and selected_families & candidate_families
    ):
        return 0.75

    selected_tokens = {
        token
        for token in selected_genre.replace("-", " ").split()
        if token
    }
    candidate_tokens = {
        token
        for token in candidate_genre.replace("-", " ").split()
        if token
    }

    if selected_tokens & candidate_tokens:
        return 0.45

    return 0.0


def _build_latent_candidate_pool(
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
    knn: NearestNeighbors,
    track_index: int,
    n_neighbors: int,
) -> pd.DataFrame:
    query = latent_features[track_index].reshape(1, -1)
    distances, indices = knn.kneighbors(
        query,
        n_neighbors=n_neighbors,
    )

    distances = distances.flatten()[1:]
    indices = indices.flatten()[1:]

    candidates = dataframe.iloc[indices].copy()
    candidates["source_latent"] = 1
    candidates["latent_similarity"] = 1 - distances
    candidates["similarity"] = candidates["latent_similarity"]
    return candidates


def _build_genre_candidate_pool(
    dataframe: pd.DataFrame,
    selected_song: pd.Series,
    limit: int = 150,
) -> pd.DataFrame:
    selected_genre = str(
        selected_song.get("track_genre", "")
    )
    selected_families = infer_genre_families(selected_genre)

    def matches(row_genre: str) -> bool:
        if row_genre == selected_genre:
            return True
        row_families = infer_genre_families(row_genre)
        return bool(selected_families & row_families)

    genre_df = dataframe[
        dataframe["track_genre"].apply(matches)
    ].copy()

    if genre_df.empty:
        return genre_df

    genre_df["source_genre"] = 1
    genre_df = genre_df.sort_values(
        "popularity",
        ascending=False,
    ).head(limit)
    return genre_df


def _build_popularity_candidate_pool(
    dataframe: pd.DataFrame,
    limit: int = 120,
) -> pd.DataFrame:
    popularity_df = dataframe.copy()
    popularity_df["source_popularity"] = 1

    if "popularity" in popularity_df.columns:
        popularity_df = popularity_df.sort_values(
            "popularity",
            ascending=False,
        )

    return popularity_df.head(limit)


def _build_audio_candidate_pool(
    dataframe: pd.DataFrame,
    selected_song: pd.Series,
    selected_index: int,
    intent: str,
    limit: int = 150,
) -> pd.DataFrame:
    audio_df = dataframe.copy()
    audio_df["audio_similarity"] = compute_audio_similarity_scores(
        selected_song,
        audio_df,
        intent=intent,
    )
    audio_df = audio_df.drop(index=selected_index, errors="ignore")
    audio_df["source_audio"] = 1
    audio_df = audio_df.sort_values(
        "audio_similarity",
        ascending=False,
    ).head(limit)
    return audio_df


def build_candidate_pool(
    track_index: int,
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
    knn: NearestNeighbors,
    intent: str = "Balanced",
    n_neighbors: int = 80,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Blend candidates from latent, genre, popularity, and audio sources.
    """

    selected_song = dataframe.iloc[track_index]

    latent_df = _build_latent_candidate_pool(
        dataframe,
        latent_features,
        knn,
        track_index,
        n_neighbors=n_neighbors,
    )
    genre_df = _build_genre_candidate_pool(
        dataframe,
        selected_song,
    )
    popularity_df = _build_popularity_candidate_pool(
        dataframe,
    )
    audio_df = _build_audio_candidate_pool(
        dataframe,
        selected_song,
        track_index,
        intent=intent,
    )

    candidate_pool = pd.concat(
        [
            latent_df,
            genre_df,
            popularity_df,
            audio_df,
        ],
        axis=0,
        sort=False,
    )

    candidate_pool["row_index"] = candidate_pool.index
    candidate_pool = candidate_pool[
        candidate_pool["row_index"] != track_index
    ]

    source_columns = [
        "source_latent",
        "source_genre",
        "source_popularity",
        "source_audio",
    ]
    for column in source_columns:
        if column not in candidate_pool.columns:
            candidate_pool[column] = 0

    candidate_pool["source_support_score"] = (
        candidate_pool[source_columns]
        .fillna(0)
        .sum(axis=1)
        / len(source_columns)
    )

    grouped = (
        candidate_pool.groupby("row_index", sort=False)
        .agg(
            {
                "track_name": "first",
                "artists": "first",
                "track_genre": "first",
                "popularity": "first",
                "track_id": "first",
                "danceability": "first",
                "energy": "first",
                "loudness": "first",
                "speechiness": "first",
                "acousticness": "first",
                "instrumentalness": "first",
                "liveness": "first",
                "valence": "first",
                "tempo": "first",
                "latent_similarity": "max",
                "audio_similarity": "max",
                "source_support_score": "max",
                "source_latent": "max",
                "source_genre": "max",
                "source_popularity": "max",
                "source_audio": "max",
            }
        )
        .reset_index(drop=False)
    )

    return selected_song, grouped


def rank_candidates(
    selected_song: pd.Series,
    candidates: pd.DataFrame,
    intent: str = "Balanced",
) -> pd.DataFrame:
    """
    Rank recommendation candidates using a tuned hybrid score.
    """

    ranked = candidates.copy()
    weights = get_weight_profile(intent)

    if "latent_similarity" not in ranked.columns:
        ranked["latent_similarity"] = 0.0
    if "audio_similarity" not in ranked.columns:
        ranked["audio_similarity"] = compute_audio_similarity_scores(
            selected_song,
            ranked,
            intent=intent,
        )

    ranked["similarity"] = ranked["latent_similarity"].fillna(0.0)

    ranked["genre_score"] = ranked.apply(
        lambda row: compute_genre_score(
            selected_song,
            row,
        ),
        axis=1,
    )

    if "popularity" in ranked.columns:
        ranked["popularity_score"] = (
            ranked["popularity"].fillna(0) / 100.0
        )
    else:
        ranked["popularity_score"] = 0.0

    if intent == "More energetic":
        ranked["intent_bonus_score"] = ranked["energy"].fillna(0.0)
    elif intent == "Discovery":
        ranked["intent_bonus_score"] = 1.0 - ranked[
            "popularity_score"
        ]
    else:
        ranked["intent_bonus_score"] = 0.0

    ranked["ranking_score"] = (
        weights["latent_similarity"] * ranked["latent_similarity"]
        + weights["audio_similarity"] * ranked["audio_similarity"]
        + weights["genre_score"] * ranked["genre_score"]
        + weights["popularity_score"] * ranked["popularity_score"]
        + weights["source_support_score"] * ranked["source_support_score"]
    )

    ranked["ranking_score"] = (
        ranked["ranking_score"]
        + 0.03 * ranked["intent_bonus_score"]
    )

    ranked = ranked.sort_values(
        ["ranking_score", "source_support_score"],
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
    artist_counter: dict[str, int] = {}

    for _, row in recommendations.iterrows():
        artist = str(row.get("artists", ""))
        count = artist_counter.get(artist, 0)

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
    n_neighbors: int = 80,
    n_recommendations: int = 10,
    intent: str = "Balanced",
) -> pd.DataFrame:
    """
    Recommend tracks using multi-source hybrid scoring.
    """

    if track_index >= len(dataframe):
        raise ValueError("Invalid track index.")

    selected_song, candidates = build_candidate_pool(
        track_index=track_index,
        dataframe=dataframe,
        latent_features=latent_features,
        knn=knn,
        intent=intent,
        n_neighbors=n_neighbors,
    )

    candidates = rank_candidates(
        selected_song,
        candidates,
        intent=intent,
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

    return candidates.head(n_recommendations)


def tune_weight_profiles(
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
    knn: NearestNeighbors,
    sample_track_indices: list[int],
) -> pd.DataFrame:
    """
    Lightweight offline comparison across intent profiles.

    This does not learn from user feedback, but it provides a practical
    evaluation foundation for comparing ranking profiles instead of guessing.
    """

    rows = []

    for intent, track_index in product(
        INTENT_WEIGHT_PROFILES.keys(),
        sample_track_indices,
    ):
        recommendations = recommend_tracks(
            track_index=track_index,
            dataframe=dataframe,
            latent_features=latent_features,
            knn=knn,
            n_recommendations=10,
            intent=intent,
        )

        if recommendations.empty:
            continue

        rows.append(
            {
                "intent": intent,
                "track_index": track_index,
                "avg_latent_similarity": recommendations[
                    "latent_similarity"
                ].mean(),
                "avg_audio_similarity": recommendations[
                    "audio_similarity"
                ].mean(),
                "avg_genre_score": recommendations[
                    "genre_score"
                ].mean(),
                "avg_popularity_score": recommendations[
                    "popularity_score"
                ].mean(),
                "artist_diversity": recommendations[
                    "artists"
                ].nunique()
                / max(len(recommendations), 1),
                "avg_source_support": recommendations[
                    "source_support_score"
                ].mean(),
                "avg_final_score": recommendations[
                    "ranking_score"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def get_track_details(
    dataframe: pd.DataFrame,
    track_index: int,
) -> pd.Series:
    """
    Return metadata for one track.
    """

    return dataframe.iloc[track_index]
