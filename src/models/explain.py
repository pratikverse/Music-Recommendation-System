"""
Explainable recommendation utilities.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


EXPLAINABLE_FEATURES = [
    "danceability",
    "energy",
    "tempo",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
]

FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "tempo": "Tempo",
    "valence": "Mood",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumental feel",
    "speechiness": "Vocal style",
    "liveness": "Live feel",
    "loudness": "Loudness",
}


def _is_available(song: pd.Series, feature: str) -> bool:
    return feature in song and pd.notna(song[feature])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_feature_matches(
    selected_song: pd.Series,
    recommended_song: pd.Series,
    threshold: float = 0.15,
) -> list[dict[str, float | str]]:
    """
    Return feature-level closeness signals used in the explanation UI.
    """

    feature_matches: list[dict[str, float | str]] = []

    for feature in EXPLAINABLE_FEATURES:
        if not (
            _is_available(selected_song, feature)
            and _is_available(recommended_song, feature)
        ):
            continue

        selected_value = _safe_float(selected_song[feature])
        recommended_value = _safe_float(recommended_song[feature])
        difference = abs(
            selected_value - recommended_value
        )

        if difference <= threshold:
            closeness = max(
                0.0,
                1.0 - (difference / threshold),
            )
            feature_matches.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "difference": difference,
                    "closeness": closeness,
                    "selected_value": selected_value,
                    "recommended_value": recommended_value,
                }
            )

    feature_matches.sort(
        key=lambda item: item["closeness"],
        reverse=True,
    )

    return feature_matches


def explain_recommendation(
    selected_song: pd.Series,
    recommended_song: pd.Series,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """
    Build a richer explanation object for one recommendation.
    """

    similarity = _safe_float(
        recommended_song.get("similarity")
    )
    ranking_score = _safe_float(
        recommended_song.get("ranking_score")
    )
    popularity_score = _safe_float(
        recommended_song.get("popularity_score")
    )
    popularity = recommended_song.get(
        "popularity",
        "N/A",
    )

    feature_matches = get_feature_matches(
        selected_song,
        recommended_song,
        threshold=threshold,
    )

    top_reasons: list[str] = []

    if feature_matches:
        for match in feature_matches[:3]:
            top_reasons.append(
                f'{match["label"]} is very close'
            )

    if (
        selected_song.get("track_genre")
        == recommended_song.get("track_genre")
    ):
        top_reasons.append("Same genre profile")

    if (
        selected_song.get("artists")
        == recommended_song.get("artists")
    ):
        top_reasons.append("Same artist")

    if not top_reasons:
        top_reasons.append(
            "Strong nearest-neighbor match in latent space"
        )

    explanation_summary = (
        f'This song was recommended because it has a '
        f'{similarity * 100:.2f}% latent-space similarity match'
    )

    if feature_matches:
        top_labels = ", ".join(
            match["label"].lower()
            for match in feature_matches[:3]
        )
        explanation_summary += (
            f" and closely matches on {top_labels}."
        )
    else:
        explanation_summary += "."

    return {
        "similarity_percent": similarity * 100,
        "ranking_score_percent": ranking_score * 100,
        "popularity_score_percent": popularity_score * 100,
        "popularity": popularity,
        "summary": explanation_summary,
        "top_reasons": top_reasons,
        "feature_matches": feature_matches,
        "same_genre": (
            selected_song.get("track_genre")
            == recommended_song.get("track_genre")
        ),
        "same_artist": (
            selected_song.get("artists")
            == recommended_song.get("artists")
        ),
    }
