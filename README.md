# ALGORHYTHM

## OVERVIEW

This project delivers an intelligent music recommendation system that curates personalized track suggestions based on user preference. It employs a deep learning autoencoder to learn rich, low-dimensional representations of music features, which are then leveraged by a K-Nearest Neighbors (KNN) algorithm to identify sonically similar tracks, even across diverse genres. The entire system is deployed as an intuitive Streamlit web application, offering a seamless and interactive user experience.

## FEATURES

* **TRACK RECOMMENDATIONS:** Users can select a track from the Spotify dataset acquired from Huggingface, and the system recommends nine similar tracks.
* **GENRE VISUALISATION:** A 3D visualization of tracks in the dataset, colored by genre, provides insights into the relationships between different music genres.
* **WEB APPLICATIOn:** The Streamlit application provides an intuitive interface for exploring recommendations.

## TECHNICAL DETAILS

1.  **PREPROCESSING**
    * The Spotify tracks dataset is loaded and preprocessed.
    * Numeric features relevant to track similarity (e.g., danceability, energy, loudness) are selected.
    * Data is scaled using StandardScaler.
    * Categorical genre data is encoded.

2.  **FEATURE EXTRACTION**
    * An autoencoder neural network is trained to reduce the dimensionality of the track features into a lower-dimensional "latent space."  This helps the model to capture the essential characteristics of each song.
    * The encoder part of the autoencoder is used to generate the latent representations for all tracks.

3.  **SIMILARITY CALCULATION**
    * A KNN model is trained on the latent representations of the tracks. The cosine distance metric is used to measure similarity between tracks in the latent space.

4.  **RECOMMENDATION GENERATION**
    * When a user selects a track, the KNN model finds the k-nearest neighbors (most similar tracks) in the latent space.
    * The system returns the top 9 most similar tracks as recommendations.

5.  **WEB APP**
    * Streamlit is used to create an interactive web application.
    * Users can select a song, view its audio preview, and see recommendations.
    * The 3D genre visualization is displayed using Plotly.
