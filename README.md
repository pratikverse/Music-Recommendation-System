# 🎧 AlogRhythm

## Overview

This project is a music recommendation system that suggests tracks similar from different genres to a user-selected song. It leverages machine learning techniques, specifically an autoencoder for feature extraction and k-nearest neighbors (KNN) for finding similar tracks. The system is implemented as a Streamlit web application, providing an interactive and user-friendly experience.

## Features

* **Track Recommendations:** Users can select a track from the Spotify dataset acquired from Huggingface, and the system recommends nine similar tracks.
* **Genre Visualization:** A 3D visualization of tracks in the dataset, colored by genre, provides insights into the relationships between different music genres.
* **Web Application:** The Streamlit application provides an intuitive interface for exploring recommendations.

## Technical Details

1.  **Data Preprocessing**
    * The Spotify tracks dataset is loaded and preprocessed.
    * Numeric features relevant to track similarity (e.g., danceability, energy, loudness) are selected.
    * Data is scaled using StandardScaler.
    * Categorical genre data is encoded.

2.  **Feature Extraction**
    * An autoencoder neural network is trained to reduce the dimensionality of the track features into a lower-dimensional "latent space."  This helps the model to capture the essential characteristics of each song.
    * The encoder part of the autoencoder is used to generate the latent representations for all tracks.

3.  **Similarity Calculation**
    * A KNN model is trained on the latent representations of the tracks.  The cosine distance metric is used to measure similarity between tracks in the latent space.

4.  **Recommendation Generation**
    * When a user selects a track, the KNN model finds the k-nearest neighbors (most similar tracks) in the latent space.
    * The system returns the top 9 most similar tracks as recommendations.

5.  **Web Application**
    * Streamlit is used to create an interactive web application.
    * Users can select a song, view its audio preview, and see recommendations.
    * The 3D genre visualization is displayed using Plotly.

## Code Structure

* `model.ipynb`:  (Initial) Jupyter Notebook containing the data loading, preprocessing, model training, and evaluation code.
* `autoencoder.py`:  Python script containing the code for building and training the autoencoder model.
* `app.py`:  Python script for the Streamlit web application.
* `requirements.txt`:  List of Python dependencies.
* `data/`: Contains the original and processed data.
* `models/`:  Stores the trained models.
* `reports/`:  Contains any generated reports or visualizations.

##   Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/pratikverse/Spotify-Music-Recommender.git
    cd Spotify-Music-Recommender
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv music_env
    source music_env/bin/activate  # On Linux/macOS
    music_env\Scripts\activate.bat  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Interact with the application:**

    * Open the provided URL in your browser.
    * Select a track from the dropdown menu.
    * View the audio preview of the selected track.
    * See the top 9 recommendations.
    * Explore the 3D genre visualization.

##   Dependencies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow/Keras
* Streamlit
* Plotly
* Joblib
* Datasets

##   Data Source

The project uses the Spotify tracks dataset, which contains a variety of information about music tracks, including audio features and genre.

