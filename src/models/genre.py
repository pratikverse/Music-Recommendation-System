"""
Genre explorer utilities.
"""

from __future__ import annotations

import pandas as pd


GENRE_EXPLORER_OPTIONS = [
    "Pop",
    "Rock",
    "EDM",
    "Jazz",
    "Metal",
]

GENRE_KEYWORDS = {
    "Pop": [
        "pop",
        "indie-pop",
        "power-pop",
        "synth-pop",
        "k-pop",
        "j-pop",
    ],
    "Rock": [
        "rock",
        "alt-rock",
        "alternative",
        "hard-rock",
        "punk-rock",
        "grunge",
    ],
    "EDM": [
        "edm",
        "electro",
        "electronic",
        "dance",
        "house",
        "techno",
        "trance",
        "dubstep",
        "drum-and-bass",
    ],
    "Jazz": [
        "jazz",
        "smooth-jazz",
        "jazz-fusion",
        "blues",
        "soul-jazz",
    ],
    "Metal": [
        "metal",
        "heavy-metal",
        "black-metal",
        "death-metal",
        "metalcore",
        "hardcore",
    ],
}


def normalize_genre_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def matches_genre_family(track_genre: str, selected_genre: str) -> bool:
    """
    Map dataset genres into a user-facing explorer family.
    """

    normalized = normalize_genre_text(track_genre)
    keywords = GENRE_KEYWORDS.get(selected_genre, [])
    return any(
        keyword.casefold() in normalized
        for keyword in keywords
    )


def recommend_by_genre(
    dataframe: pd.DataFrame,
    selected_genre: str,
    n_recommendations: int = 12,
) -> pd.DataFrame:
    """
    Return a browseable set of tracks for a selected genre family.
    """

    genre_df = dataframe[
        dataframe["track_genre"].apply(
            lambda genre: matches_genre_family(
                genre,
                selected_genre,
            )
        )
    ].copy()

    if genre_df.empty:
        return genre_df

    if "popularity" in genre_df.columns:
        genre_df["genre_explorer_score"] = (
            genre_df["popularity"].fillna(0) / 100.0
        )
    else:
        genre_df["genre_explorer_score"] = 0.0

    genre_df = genre_df.sort_values(
        ["genre_explorer_score", "track_name"],
        ascending=[False, True],
    )
    genre_df = genre_df.drop_duplicates(
        subset=["track_name", "artists"]
    )
    return genre_df.head(n_recommendations)


def generate_genre_playlist(
    dataframe: pd.DataFrame,
    selected_genre: str,
    playlist_size: int = 20,
) -> pd.DataFrame:
    """
    Generate a fixed-length playlist for a selected genre family.
    """

    playlist = recommend_by_genre(
        dataframe=dataframe,
        selected_genre=selected_genre,
        n_recommendations=max(playlist_size * 3, playlist_size),
    )

    if playlist.empty:
        return playlist

    # Keep the playlist varied by avoiding artist repetition where possible.
    artist_seen: dict[str, int] = {}
    selected_rows = []

    for _, row in playlist.iterrows():
        artist = str(row.get("artists", "")).strip()
        artist_count = artist_seen.get(artist, 0)

        if artist and artist_count >= 2:
            continue

        artist_seen[artist] = artist_count + 1
        selected_rows.append(row)

        if len(selected_rows) >= playlist_size:
            break

    if len(selected_rows) < playlist_size:
        used_keys = {
            (row["track_name"], row["artists"])
            for row in selected_rows
        }
        for _, row in playlist.iterrows():
            row_key = (
                row["track_name"],
                row["artists"],
            )
            if row_key in used_keys:
                continue
            selected_rows.append(row)
            used_keys.add(row_key)
            if len(selected_rows) >= playlist_size:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)
