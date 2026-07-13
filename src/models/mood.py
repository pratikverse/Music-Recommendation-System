"""
Mood profiling and mood-based recommendation utilities.
"""

from __future__ import annotations

import pandas as pd


MOOD_ORDER = [
    "Workout",
    "Study",
    "Sleep",
    "Party",
    "Happy",
    "Sad",
]

MOOD_RULES = {
    "Workout": {
        "energy": 0.80,
        "tempo": 125.0,
        "valence": 0.55,
    },
    "Study": {
        "acousticness": 0.45,
        "instrumentalness": 0.20,
        "energy_max": 0.55,
        "speechiness_max": 0.12,
        "danceability_max": 0.65,
    },
    "Sleep": {
        "acousticness": 0.55,
        "energy_max": 0.35,
        "tempo_max": 95.0,
        "valence_max": 0.55,
        "loudness_max": -6.0,
    },
    "Party": {
        "danceability": 0.72,
        "energy": 0.72,
        "valence": 0.60,
        "tempo": 115.0,
    },
    "Happy": {
        "valence": 0.68,
        "energy": 0.55,
        "danceability": 0.55,
    },
    "Sad": {
        "valence_max": 0.38,
        "energy_max": 0.50,
        "acousticness": 0.30,
    },
}


def _safe_float(row: pd.Series, feature: str) -> float:
    value = row.get(feature, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def score_track_moods(track: pd.Series) -> dict[str, float]:
    """
    Score a track against each mood profile using simple heuristics.
    """

    energy = _safe_float(track, "energy")
    tempo = _safe_float(track, "tempo")
    valence = _safe_float(track, "valence")
    danceability = _safe_float(track, "danceability")
    acousticness = _safe_float(track, "acousticness")
    instrumentalness = _safe_float(track, "instrumentalness")
    speechiness = _safe_float(track, "speechiness")
    loudness = _safe_float(track, "loudness")

    scores = {
        "Workout": (
            0.35 * energy
            + 0.20 * min(tempo / 160.0, 1.0)
            + 0.20 * valence
            + 0.15 * danceability
            + 0.10 * (1.0 - acousticness)
        ),
        "Study": (
            0.30 * acousticness
            + 0.25 * instrumentalness
            + 0.20 * (1.0 - min(energy, 1.0))
            + 0.15 * (1.0 - min(speechiness * 3.0, 1.0))
            + 0.10 * (1.0 - abs(danceability - 0.45))
        ),
        "Sleep": (
            0.30 * acousticness
            + 0.25 * (1.0 - min(energy, 1.0))
            + 0.20 * (1.0 - min(tempo / 140.0, 1.0))
            + 0.15 * (1.0 - min(valence, 1.0))
            + 0.10 * min(max((-loudness - 3.0) / 15.0, 0.0), 1.0)
        ),
        "Party": (
            0.30 * danceability
            + 0.30 * energy
            + 0.20 * valence
            + 0.15 * min(tempo / 150.0, 1.0)
            + 0.05 * (1.0 - acousticness)
        ),
        "Happy": (
            0.45 * valence
            + 0.25 * energy
            + 0.20 * danceability
            + 0.10 * min(tempo / 140.0, 1.0)
        ),
        "Sad": (
            0.40 * (1.0 - valence)
            + 0.25 * acousticness
            + 0.20 * (1.0 - energy)
            + 0.15 * (1.0 - min(tempo / 140.0, 1.0))
        ),
    }

    return {
        mood: round(max(0.0, min(score, 1.0)), 4)
        for mood, score in scores.items()
    }


def pick_primary_mood(track: pd.Series) -> str:
    """
    Choose the best-fit mood label for a track.
    """

    scores = score_track_moods(track)
    return max(
        MOOD_ORDER,
        key=lambda mood: (scores[mood], -MOOD_ORDER.index(mood)),
    )


def assign_moods(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add mood labels and scores to the track catalog.
    """

    enriched = dataframe.copy()
    mood_scores = enriched.apply(
        score_track_moods,
        axis=1,
    )
    mood_score_df = pd.DataFrame(
        mood_scores.tolist(),
        index=enriched.index,
    )

    for mood in MOOD_ORDER:
        enriched[f"{mood.lower()}_score"] = mood_score_df[mood]

    enriched["mood"] = mood_score_df.idxmax(axis=1)
    return enriched


def explain_mood_fit(track: pd.Series, mood: str) -> list[str]:
    """
    Generate short human-readable reasons for the assigned mood.
    """

    reasons: list[str] = []
    energy = _safe_float(track, "energy")
    tempo = _safe_float(track, "tempo")
    valence = _safe_float(track, "valence")
    danceability = _safe_float(track, "danceability")
    acousticness = _safe_float(track, "acousticness")
    instrumentalness = _safe_float(track, "instrumentalness")

    if mood == "Workout":
        if energy >= 0.8:
            reasons.append("High energy supports intense activity")
        if tempo >= 125:
            reasons.append("Fast tempo keeps momentum up")
        if valence >= 0.55:
            reasons.append("Positive mood makes it feel motivating")
    elif mood == "Study":
        if acousticness >= 0.45:
            reasons.append("Acoustic texture feels less distracting")
        if instrumentalness >= 0.20:
            reasons.append("Instrumental character helps focus")
        if energy <= 0.55:
            reasons.append("Moderate energy suits concentration")
    elif mood == "Sleep":
        if energy <= 0.35:
            reasons.append("Low energy feels calm and restful")
        if tempo <= 95:
            reasons.append("Slow tempo supports winding down")
        if acousticness >= 0.55:
            reasons.append("Soft acoustic tone fits a sleep mood")
    elif mood == "Party":
        if danceability >= 0.72:
            reasons.append("High danceability makes it party-friendly")
        if energy >= 0.72:
            reasons.append("Strong energy lifts the room")
        if valence >= 0.60:
            reasons.append("Upbeat mood adds celebration")
    elif mood == "Happy":
        if valence >= 0.68:
            reasons.append("High valence gives it a cheerful feel")
        if energy >= 0.55:
            reasons.append("Lively energy keeps the mood bright")
        if danceability >= 0.55:
            reasons.append("Danceable rhythm makes it feel playful")
    elif mood == "Sad":
        if valence <= 0.38:
            reasons.append("Low valence gives it an emotional tone")
        if acousticness >= 0.30:
            reasons.append("Acoustic texture makes it feel intimate")
        if energy <= 0.50:
            reasons.append("Lower energy reinforces the reflective mood")

    if not reasons:
        reasons.append("Its audio profile best matches this mood")

    return reasons


def recommend_by_mood(
    dataframe: pd.DataFrame,
    mood: str,
    n_recommendations: int = 12,
) -> pd.DataFrame:
    """
    Return the best matching tracks for a selected mood.
    """

    score_column = f"{mood.lower()}_score"
    if score_column not in dataframe.columns:
        raise ValueError(f"Unknown mood score column: {score_column}")

    mood_df = dataframe[
        dataframe["mood"] == mood
    ].copy()

    if mood_df.empty:
        mood_df = dataframe.copy()

    if "popularity" in mood_df.columns:
        mood_df["popularity_score"] = (
            mood_df["popularity"].fillna(0) / 100.0
        )
    else:
        mood_df["popularity_score"] = 0.0

    mood_df["mood_match_score"] = (
        0.80 * mood_df[score_column]
        + 0.20 * mood_df["popularity_score"]
    )

    mood_df = mood_df.sort_values(
        ["mood_match_score", score_column],
        ascending=False,
    )
    mood_df = mood_df.drop_duplicates(
        subset=["track_name", "artists"]
    )
    return mood_df.head(n_recommendations)
