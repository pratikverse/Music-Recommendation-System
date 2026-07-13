"""
Utilities for saving and loading model artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model

from src.config import (
    AUTOENCODER_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    KNN_MODEL_PATH,
    LATENT_FEATURES_PATH,
    DATAFRAME_PATH,
)


# ==========================================================
# Save Functions
# ==========================================================

def save_artifacts(
    autoencoder: Model,
    encoder: Model,
    scaler,
    label_encoder,
    knn,
    dataframe: pd.DataFrame,
    latent_features: np.ndarray,
) -> None:
    """
    Save every artifact required for inference.
    """

    print("\nSaving trained artifacts...")

    autoencoder.save(AUTOENCODER_PATH)

    encoder.save(ENCODER_PATH)

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH,
    )

    joblib.dump(
        knn,
        KNN_MODEL_PATH,
    )

    dataframe.to_pickle(
        DATAFRAME_PATH,
    )

    np.save(
        LATENT_FEATURES_PATH,
        latent_features,
    )

    print("Artifacts saved successfully.")


# ==========================================================
# Individual Loaders
# ==========================================================

def _ensure_artifact_exists(path: Path) -> Path:
    """
    Raise a clear error when a required artifact is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. "
            "Make sure the trained files in the models/ directory are committed and deployed."
        )
    return path

def load_autoencoder() -> Model:
    """
    Load the trained autoencoder.
    """
    return load_model(_ensure_artifact_exists(AUTOENCODER_PATH))


def load_encoder() -> Model:
    """
    Load the trained encoder.
    """
    return load_model(_ensure_artifact_exists(ENCODER_PATH))


def load_dataframe() -> pd.DataFrame:
    """
    Load processed dataframe.
    """
    return pd.read_pickle(_ensure_artifact_exists(DATAFRAME_PATH))


def load_latent_features() -> np.ndarray:
    """
    Load latent embeddings.
    """
    return np.load(_ensure_artifact_exists(LATENT_FEATURES_PATH))


def load_knn():
    """
    Load trained KNN model.
    """
    return joblib.load(_ensure_artifact_exists(KNN_MODEL_PATH))


def load_scaler():
    """
    Load fitted StandardScaler.
    """
    return joblib.load(_ensure_artifact_exists(SCALER_PATH))


def load_label_encoder():
    """
    Load fitted LabelEncoder.
    """
    return joblib.load(_ensure_artifact_exists(LABEL_ENCODER_PATH))


# ==========================================================
# Convenience Loader
# ==========================================================

def load_artifacts() -> dict:
    """
    Load every artifact needed by the application.
    """

    return {
        "autoencoder": load_autoencoder(),
        "encoder": load_encoder(),
        "scaler": load_scaler(),
        "label_encoder": load_label_encoder(),
        "knn": load_knn(),
        "dataframe": load_dataframe(),
        "latent_features": load_latent_features(),
    }
