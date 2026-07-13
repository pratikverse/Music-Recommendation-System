"""
Global configuration for TuneMatch.
"""

from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "maharshipandya/spotify-tracks-dataset"

# =============================================================================
# Training Hyperparameters
# =============================================================================

ENCODING_DIM = 8

EPOCHS = 50

BATCH_SIZE = 128

VALIDATION_SPLIT = 0.20

LEARNING_RATE = 1e-3

L1_REG = 1e-5

L2_REG = 1e-4

DROPOUT_RATE = 0.20

# =============================================================================
# KNN
# =============================================================================

KNN_METRIC = "cosine"

KNN_ALGORITHM = "brute"

KNN_NEIGHBORS = 20

# =============================================================================
# Artifact Paths
# =============================================================================

AUTOENCODER_PATH = MODELS_DIR / "autoencoder.keras"

ENCODER_PATH = MODELS_DIR / "encoder.keras"

SCALER_PATH = MODELS_DIR / "scaler.pkl"

LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

KNN_MODEL_PATH = MODELS_DIR / "knn_model.pkl"

LATENT_FEATURES_PATH = MODELS_DIR / "latent_features.npy"

DATAFRAME_PATH = MODELS_DIR / "df_processed.pkl"