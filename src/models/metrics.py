"""
Model evaluation utilities.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

from tensorflow.keras.models import Model


def evaluate_autoencoder(
    autoencoder: Model,
    X_scaled: np.ndarray,
) -> dict:
    """
    Evaluate the trained autoencoder.

    Parameters
    ----------
    autoencoder:
        Trained autoencoder model.

    X_scaled:
        Scaled feature matrix.

    Returns
    -------
    Dictionary containing evaluation metrics.
    """

    reconstructed = autoencoder.predict(
        X_scaled,
        verbose=0,
    )

    metrics = {
        "mse": mean_squared_error(
            X_scaled,
            reconstructed,
        ),
        "mae": mean_absolute_error(
            X_scaled,
            reconstructed,
        ),
        "r2": r2_score(
            X_scaled,
            reconstructed,
        ),
        "explained_variance": explained_variance_score(
            X_scaled,
            reconstructed,
        ),
    }

    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print(f"MSE                 : {metrics['mse']:.6f}")
    print(f"MAE                 : {metrics['mae']:.6f}")
    print(f"R² Score            : {metrics['r2']:.6f}")
    print(
        f"Explained Variance  : "
        f"{metrics['explained_variance']:.6f}"
    )

    return metrics