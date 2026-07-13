"""
Training pipeline for TuneMatch.

Run:

    python -m src.training.train
"""

from __future__ import annotations

from src.config import (
    ENCODING_DIM,
    EPOCHS,
    BATCH_SIZE,
)

from src.models.preprocessing import (
    load_and_preprocess_data,
)

from src.models.autoencoder import (
    build_autoencoder,
    train_autoencoder,
)

from src.models.metrics import (
    evaluate_autoencoder,
)

from src.models.recommender import (
    build_knn_model,
)

from src.models.visualization import (
    plot_feature_heatmap,
    plot_tracks_by_genre,
)

from src.models.artifacts import (
    save_artifacts,
)


def main() -> None:
    """
    Complete TuneMatch training pipeline.
    """

    print("=" * 70)
    print("TuneMatch Training Pipeline")
    print("=" * 70)

    # ==========================================================
    # Load Dataset
    # ==========================================================

    (
        dataframe,
        X_scaled,
        scaler,
        label_encoder,
        numeric_features,
    ) = load_and_preprocess_data()

    # ==========================================================
    # Visualize Dataset
    # ==========================================================

    heatmap = plot_feature_heatmap(
        dataframe,
        numeric_features,
    )

    heatmap.show()

    # ==========================================================
    # Build Model
    # ==========================================================

    autoencoder, encoder = build_autoencoder(
        input_dim=X_scaled.shape[1],
        encoding_dim=ENCODING_DIM,
    )

    # ==========================================================
    # Train
    # ==========================================================

    history = train_autoencoder(
        autoencoder=autoencoder,
        X_scaled=X_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # ==========================================================
    # Evaluate
    # ==========================================================

    metrics = evaluate_autoencoder(
        autoencoder,
        X_scaled,
    )

    print("\nTraining Metrics")

    for metric, value in metrics.items():

        print(
            f"{metric:20}: {value:.6f}"
        )

    # ==========================================================
    # Encode Songs
    # ==========================================================

    print("\nGenerating latent vectors...")

    latent_features = encoder.predict(
        X_scaled,
        verbose=0,
    )

    # ==========================================================
    # Train KNN
    # ==========================================================

    print("Training recommendation engine...")

    knn = build_knn_model(
        latent_features,
    )

    # ==========================================================
    # Save Artifacts
    # ==========================================================

    save_artifacts(
        autoencoder=autoencoder,
        encoder=encoder,
        scaler=scaler,
        label_encoder=label_encoder,
        knn=knn,
        dataframe=dataframe,
        latent_features=latent_features,
    )

    # ==========================================================
    # Visualize Latent Space
    # ==========================================================

    latent_plot = plot_tracks_by_genre(
        latent_features,
        dataframe,
    )

    latent_plot.show()

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()