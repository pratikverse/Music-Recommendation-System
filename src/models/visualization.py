"""
Visualization utilities for TuneMatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA


def calculate_pca(
    latent_features: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """
    Perform PCA on latent embeddings.
    """

    if len(latent_features) == 0:
        raise ValueError("Latent feature array is empty.")

    pca = PCA(n_components=n_components)

    return pca.fit_transform(latent_features)


def calculate_correlation(
    dataframe: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    """
    Compute feature correlation matrix.
    """

    features = [
        feature
        for feature in numeric_features
        if feature in dataframe.columns
    ]

    if not features:
        raise ValueError(
            "No numeric features available."
        )

    return dataframe[features].corr()


def plot_feature_heatmap(
    dataframe: pd.DataFrame,
    numeric_features: list[str],
) -> go.Figure:
    """
    Create an interactive correlation heatmap.
    """

    correlation = calculate_correlation(
        dataframe,
        numeric_features,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(correlation.values, 2),
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Feature Correlation Heatmap",
        width=800,
        height=800,
        xaxis_tickangle=-45,
    )

    return fig


def plot_tracks_by_genre(
    latent_features: np.ndarray,
    dataframe: pd.DataFrame,
) -> go.Figure:
    """
    Visualize latent embeddings using PCA.
    """

    projection = calculate_pca(
        latent_features,
    )

    plot_df = pd.DataFrame(
        projection,
        columns=[
            "PC1",
            "PC2",
            "PC3",
        ],
    )

    plot_df["Genre"] = dataframe["track_genre"]
    plot_df["Track"] = dataframe["track_name"]
    plot_df["Artist"] = dataframe["artists"]

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Genre",
        hover_data=[
            "Track",
            "Artist",
        ],
        title="Latent Space Visualization",
        labels={
            "PC1": "Principal Component 1",
            "PC2": "Principal Component 2",
            "PC3": "Principal Component 3",
        },
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        width=1200,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.85,
        ),
    )

    return fig


def plot_training_history(history) -> None:
    """
    Plot training history using matplotlib.

    Optional utility if you prefer to keep all
    visualizations inside this module.
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title("Training History")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()