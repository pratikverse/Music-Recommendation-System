"""
Data loading and preprocessing utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DATASET_NAME


NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "track_genre_encoded",
]


def remove_outliers_iqr(
    dataframe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Remove outliers using the IQR rule.
    """

    df = dataframe.copy()

    for feature in features:

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[
            (df[feature] >= lower)
            & (df[feature] <= upper)
        ]

    return df.reset_index(drop=True)


def load_and_preprocess_data(
    dataset_name: str = DATASET_NAME,
):
    """
    Loads and preprocesses the Spotify dataset.

    Returns
    -------
    dataframe
    scaled_features
    scaler
    label_encoder
    numeric_features
    """

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = load_dataset(dataset_name)

    df = pd.DataFrame(dataset["train"])

    print(f"Loaded {len(df):,} songs.")

    label_encoder = LabelEncoder()

    df["track_genre_encoded"] = label_encoder.fit_transform(
        df["track_genre"]
    )

    df = df.dropna(
        subset=NUMERIC_FEATURES
    ).reset_index(drop=True)

    print(f"After removing missing values : {len(df):,}")

    df = remove_outliers_iqr(
        df,
        NUMERIC_FEATURES[:-1],
    )

    print(f"After removing outliers       : {len(df):,}")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        df[NUMERIC_FEATURES].to_numpy()
    )

    print("Feature scaling complete.")

    return (
        df,
        X_scaled,
        scaler,
        label_encoder,
        NUMERIC_FEATURES,
    )