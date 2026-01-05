# 🎵 Music Recommendation System

An intelligent **music recommendation system** that suggests sonically similar tracks using deep learning and nearest-neighbor search — deployed as an interactive **Streamlit web app**.

---

## ✨ Overview
It learns meaningful representations of music using a **deep autoencoder**, then identifies similar tracks via **K-Nearest Neighbors (KNN)** in a latent feature space. This approach enables accurate recommendations even across diverse genres, presented through a clean and interactive web interface.

---

## 🚀 Features
- 🎧 **Personalized Recommendations** – Select a track and receive **9 similar songs**
- 🌈 **3D Genre Visualization** – Explore genre relationships in latent space
- 🖥️ **Interactive Web App** – Seamless exploration with Streamlit

---

## ⚙️ How It Works

### 1. Preprocessing
- Spotify dataset sourced from **HuggingFace**
- Selection of numeric audio features (danceability, energy, loudness, etc.)
- Feature scaling using **StandardScaler**
- Genre encoding for visualization

### 2. Feature Extraction
- A **deep autoencoder** compresses high-dimensional audio features
- Encoder outputs a compact **latent representation** for each track

### 3. Similarity Modeling
- **KNN** trained on latent vectors
- **Cosine similarity** used to measure track closeness

### 4. Recommendation Engine
- Given a selected song, KNN retrieves nearest neighbors
- Top **9 most similar tracks** returned as recommendations

### 5. Web Application
- Built using **Streamlit**
- Song selection with audio preview
- Interactive **3D genre plot** using Plotly

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Streamlit  
- Plotly  

---

## 🎯 Use Cases
- Music discovery platforms  
- Personalized recommendation systems  
- Audio feature analysis & visualization  

---
