"""
Autoencoder architecture and training utilities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from src.config import (
    DROPOUT_RATE,
    L1_REG,
    L2_REG,
    LEARNING_RATE,
    VALIDATION_SPLIT,
)


def build_autoencoder(
    input_dim: int,
    encoding_dim: int,
) -> tuple[Model, Model]:
    """
    Build the Autoencoder and Encoder models.
    """

    inputs = Input(shape=(input_dim,), name="input")

    # ============================================================
    # Encoder
    # ============================================================

    x = Dense(
        64,
        activation="relu",
        kernel_regularizer=l1_l2(
            l1=L1_REG,
            l2=L2_REG,
        ),
    )(inputs)

    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Dense(
        32,
        activation="relu",
        kernel_regularizer=l1_l2(
            l1=L1_REG,
            l2=L2_REG,
        ),
    )(x)

    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    encoded = Dense(
        encoding_dim,
        activation="relu",
        name="latent_vector",
    )(x)

    # ============================================================
    # Decoder
    # ============================================================

    x = Dense(
        32,
        activation="relu",
    )(encoded)

    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Dense(
        64,
        activation="relu",
    )(x)

    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(
        input_dim,
        activation="linear",
    )(x)

    autoencoder = Model(
        inputs=inputs,
        outputs=outputs,
        name="Autoencoder",
    )

    encoder = Model(
        inputs=inputs,
        outputs=encoded,
        name="Encoder",
    )

    autoencoder.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="mse",
    )

    return autoencoder, encoder


def train_autoencoder(
    autoencoder: Model,
    X_scaled: np.ndarray,
    epochs: int,
    batch_size: int,
):
    """
    Train the autoencoder.
    """

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history)

    return history


def plot_training_history(history) -> None:
    """
    Plot training and validation loss.
    """

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