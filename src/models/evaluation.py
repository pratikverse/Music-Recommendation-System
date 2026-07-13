"""
Offline evaluation helpers for TuneMatch recommenders.
"""

from __future__ import annotations

import pandas as pd

from src.models.recommender import (
    INTENT_WEIGHT_PROFILES,
    tune_weight_profiles,
)


def summarize_profile_metrics(
    evaluation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate profile metrics across sampled tracks.
    """

    if evaluation_frame.empty:
        return evaluation_frame

    summary = (
        evaluation_frame.groupby("intent", as_index=False)
        .agg(
            {
                "avg_latent_similarity": "mean",
                "avg_audio_similarity": "mean",
                "avg_genre_score": "mean",
                "avg_popularity_score": "mean",
                "artist_diversity": "mean",
                "avg_source_support": "mean",
                "avg_final_score": "mean",
            }
        )
        .sort_values("avg_final_score", ascending=False)
    )
    return summary.reset_index(drop=True)


def choose_best_profile(
    evaluation_summary: pd.DataFrame,
) -> str:
    """
    Return the highest-scoring profile from an evaluation summary.
    """

    if evaluation_summary.empty:
        return "Balanced"

    return str(
        evaluation_summary.iloc[0]["intent"]
    )


def evaluate_recommendation_profiles(
    dataframe,
    latent_features,
    knn,
    sample_track_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Compare ranking profiles across sampled tracks and return a recommendation.
    """

    raw_results = tune_weight_profiles(
        dataframe=dataframe,
        latent_features=latent_features,
        knn=knn,
        sample_track_indices=sample_track_indices,
    )
    summary = summarize_profile_metrics(raw_results)
    best_profile = choose_best_profile(summary)
    return raw_results, summary, best_profile


def list_available_profiles() -> list[str]:
    """
    Return configured ranking profiles.
    """

    return list(INTENT_WEIGHT_PROFILES.keys())
