import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l1_l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import logging
import tensorflow as tf # Import tensorflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset_name="maharshipandya/spotify-tracks-dataset"):
    """Loads the Spotify tracks dataset."""
    try:
        ds = load_dataset(dataset_name)
        df = pd.DataFrame(ds["train"])
        logging.info(f"Dataset '{dataset_name}' loaded successfully. Initial shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def preprocess_data(df, initial_numeric_features):
    """
    Preprocesses the data: encodes genres, handles missing values, removes outliers,
    and scales numeric features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        initial_numeric_features (list): List of numeric feature names present initially,
                                         excluding 'track_genre_encoded'.

    Returns:
        tuple: (NumPy array, StandardScaler, pd.DataFrame) - Scaled features, the fitted scaler,
               and the cleaned/processed DataFrame.
    """
    logging.info("Starting data preprocessing...")

    # 1. Encode 'track_genre' first
    if 'track_genre' not in df.columns:
        logging.error("'track_genre' column not found in DataFrame. Cannot proceed with encoding.")
        raise KeyError("'track_genre' column is required for encoding.")

    le = LabelEncoder()
    # Convert to string to handle potential mixed types before encoding
    df['track_genre_encoded'] = le.fit_transform(df['track_genre'].astype(str)) 
    logging.info("Track genre encoded to 'track_genre_encoded'.")

    # 2. Define the complete list of numeric features, including the encoded genre
    final_numeric_features = list(initial_numeric_features)
    if 'track_genre_encoded' not in final_numeric_features:
        final_numeric_features.append('track_genre_encoded')

    # 3. Handle missing values
    initial_rows = df.shape[0]
    df_cleaned = df.dropna(subset=final_numeric_features).reset_index(drop=True)
    rows_after_dropna = df_cleaned.shape[0]
    if rows_after_dropna < initial_rows:
        logging.warning(f"Removed {initial_rows - rows_after_dropna} rows due to NaNs in specified numeric features.")
    logging.info(f"DataFrame shape after dropna: {df_cleaned.shape}")

    # 4. Handle outliers using IQR (excluding the encoded genre, as it's categorical)
    features_for_outlier_removal = [f for f in initial_numeric_features if f in df_cleaned.columns] # Ensure feature exists

    rows_before_outlier_removal = df_cleaned.shape[0]
    for feature in features_for_outlier_removal:
        Q1 = df_cleaned[feature].quantile(0.25)
        Q3 = df_cleaned[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[~((df_cleaned[feature] < lower_bound) | (df_cleaned[feature] > upper_bound))]
    rows_after_outlier_removal = df_cleaned.shape[0]
    if rows_after_outlier_removal < rows_before_outlier_removal:
        logging.warning(f"Removed {rows_before_outlier_removal - rows_after_outlier_removal} rows due to outliers.")
    logging.info(f"DataFrame shape after outlier removal: {df_cleaned.shape}")

    # 5. Check if the DataFrame is empty after preprocessing
    if df_cleaned.empty:
        logging.error("DataFrame became empty after preprocessing. Cannot proceed.")
        raise ValueError("Processed DataFrame is empty.")

    # 6. Scale numeric features
    # Ensure all final_numeric_features columns actually exist in df_cleaned
    missing_for_scaling = [f for f in final_numeric_features if f not in df_cleaned.columns]
    if missing_for_scaling:
        logging.error(f"Columns missing for scaling: {missing_for_scaling}. This indicates data loss or error earlier.")
        raise KeyError(f"Missing columns for final scaling: {missing_for_scaling}")

    X = df_cleaned[final_numeric_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info(f"Features scaled. X_scaled shape: {X_scaled.shape}")

    logging.info("Data preprocessing completed.")
    return X_scaled, scaler, df_cleaned # NOW RETURNING THE CLEANED DATAFRAME


def create_autoencoder(input_dim, architecture, l1_reg=1e-5, l2_reg=1e-4, dropout_rate=0.2):
    """
    Creates the autoencoder model.

    Args:
        input_dim (int): Dimensionality of the input data.
        architecture (list): List defining the autoencoder architecture.
                              Each element is a tuple (units, activation).
        l1_reg (float): L1 regularization strength.
        l2_reg (float): L2 regularization strength.
        dropout_rate (float): Dropout rate.

    Returns:
        tuple: (autoencoder model, encoder model)
    """
    logging.info(f"Building autoencoder with input_dim={input_dim}, architecture={architecture}")
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Encoder
    for i, (units, activation) in enumerate(architecture[:-1]):  # Exclude the last layer (latent space)
        x = Dense(units, activation=activation, kernel_regularizer=l1_l2(l1_reg, l2_reg), name=f'encoder_dense_{i}')(x)
        x = BatchNormalization(name=f'encoder_bn_{i}')(x)
        x = Dropout(dropout_rate, name=f'encoder_dropout_{i}')(x)
        logging.debug(f"Encoder layer {i}: {units} units, {activation} activation")

    # Latent space
    latent_dim, latent_activation = architecture[-1]
    encoded = Dense(latent_dim, activation=latent_activation, kernel_regularizer=l1_l2(l1_reg, l2_reg), name='latent_space')(x)
    encoder = Model(input_layer, encoded, name='encoder_model')  # Define encoder model
    logging.debug(f"Latent space: {latent_dim} units, {latent_activation} activation")

    # Decoder
    x = encoded
    for i, (units, activation) in enumerate(reversed(architecture[:-1])):
        x = Dense(units, activation=activation)(x)
        x = BatchNormalization(name=f'decoder_bn_{i}')(x)
        x = Dropout(dropout_rate, name=f'decoder_dropout_{i}')(x)
        logging.debug(f"Decoder layer {i}: {units} units, {activation} activation")

    decoded = Dense(input_dim, activation='linear', name='output_layer')(x)  # Output layer
    autoencoder = Model(input_layer, decoded, name='autoencoder_model')

    autoencoder.compile(optimizer='adam', loss='mse')
    logging.info("Autoencoder model created and compiled.")
    autoencoder.summary(print_fn=lambda x: logging.info(x)) # Log model summary
    encoder.summary(print_fn=lambda x: logging.info(x)) # Log encoder summary
    return autoencoder, encoder

def train_model(autoencoder, X_scaled, epochs, batch_size, validation_split, patience=10, reduce_lr_patience=5):
    """Trains the autoencoder model."""
    logging.info("Starting autoencoder training...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=reduce_lr_patience, min_lr=1e-6, verbose=1)
    ]

    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1 # Keep verbose to see training progress
    )

    logging.info("Autoencoder training completed.")
    return history

def evaluate_model(autoencoder, X_scaled, output_file="model_metrics.txt"):
    """Evaluates the autoencoder model and saves the metrics."""
    logging.info("Evaluating autoencoder model...")
    predictions = autoencoder.predict(X_scaled, verbose=0) # Suppress output during prediction
    mse = mean_squared_error(X_scaled, predictions)
    r2 = r2_score(X_scaled, predictions)

    logging.info(f"Model Evaluation Metrics: MSE = {mse:.4f}, R² = {r2:.4f}")

    with open(output_file, "w") as f:
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
    logging.info(f"Model metrics saved to '{output_file}'")

def plot_training_history(history, output_file="training_history.png"):
    """Plots the training history (loss vs. epochs)."""
    logging.info(f"Saving training history plot to '{output_file}'")
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Training history plot saved to '{output_file}'")

def create_knn_model(latent_features, n_neighbors=5, metric='cosine', output_file="knn_model.pkl"):
    """Creates and saves the KNN model."""
    logging.info(f"Creating KNN model with n_neighbors={n_neighbors}, metric='{metric}'")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(latent_features)
    joblib.dump(knn, output_file)
    logging.info(f"KNN model saved to '{output_file}'")
    return knn

def save_artifacts(df_processed, X_scaled, latent_features, df_file="df_processed.pkl", x_file="X_scaled.npy", latent_file="latent_features.npy"):
    """Saves data artifacts."""
    try:
        df_processed.to_pickle(df_file) # Save the processed DataFrame
        np.save(x_file, X_scaled)
        np.save(latent_file, latent_features)
        logging.info("Data artifacts saved successfully.")
    except Exception as e:
        logging.error(f"Error saving data artifacts: {e}")
        raise

def main(epochs=50, batch_size=128, validation_split=0.2, latent_dim=8):
    """Main function to orchestrate the autoencoder training and KNN model creation."""

    logging.info("Starting autoencoder training and KNN model creation process.")

    df_initial = load_data() # Load the raw data

    # Define only the initial numeric features. 'track_genre_encoded' will be added in preprocess_data.
    initial_numeric_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    # Pass a copy of df_initial to preprocess_data
    # Capture the returned df_cleaned
    try:
        X_scaled, scaler, df_cleaned = preprocess_data(df_initial.copy(), initial_numeric_features)
    except Exception as e:
        logging.error(f"Critical error during data preprocessing: {e}")
        return # Exit if preprocessing fails

    input_dim = X_scaled.shape[1]
    logging.info(f"Input dimension for autoencoder: {input_dim}")

    # Flexible architecture definition for the autoencoder
    architecture = [(64, 'relu'), (32, 'relu'), (latent_dim, 'relu')]

    autoencoder, encoder = create_autoencoder(input_dim, architecture)
    
    # Train the autoencoder
    history = train_model(autoencoder, X_scaled, epochs, batch_size, validation_split)
    plot_training_history(history)
    evaluate_model(autoencoder, X_scaled)

    # Get latent features from the trained encoder
    latent_features = encoder.predict(X_scaled, verbose=0)
    logging.info(f"Latent features extracted. Shape: {latent_features.shape}")

    # Create and save the KNN model
    create_knn_model(latent_features, n_neighbors=20) # Using a default of 20 neighbors for KNN

    # Save important artifacts (processed df, scaled X, latent features)
    # Pass the df_cleaned DataFrame to save_artifacts
    save_artifacts(df_cleaned, X_scaled, latent_features)

    logging.info("Autoencoder training and KNN model creation completed successfully.")

if __name__ == "__main__":
    # You can adjust these parameters for experimentation
    main(epochs=50, batch_size=128, validation_split=0.2, latent_dim=8)
