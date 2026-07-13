import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from src.models.search import (
    build_search_choices,
    fuzzy_search,
)

from src.models.artifacts import load_artifacts
from src.models.recommender import (
    recommend_tracks,
    get_track_details,
)
from src.models.visualization import (
    plot_feature_heatmap,
    plot_tracks_by_genre,
)

from src.models.preprocessing import (
    NUMERIC_FEATURES,
)

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="TuneMatch",
    page_icon="🎵",
    layout="wide",
)

st.title("🎧 TuneMatch AI")

st.caption(
    "AI-powered Music Recommendation Engine using Deep Autoencoders"
)

# -------------------------------------------------------
# Load Artifacts
# -------------------------------------------------------

@st.cache_resource
def load_resources():

    return load_artifacts()


resources = load_resources()

df = resources["dataframe"]

latent_features = resources["latent_features"]

knn = resources["knn"]

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------

tab1, tab2 = st.tabs(
    [
        "🎵 Recommendations",
        "📊 Visualization",
    ]
)
col1, col2, col3 = st.columns(3)

col1.metric(
    "Songs",
    f"{len(df):,}",
)

col2.metric(
    "Artists",
    df["artists"].nunique(),
)

col3.metric(
    "Genres",
    df["track_genre"].nunique(),
)

st.divider()

# -------------------------------------------------------
# Spotify Embed
# -------------------------------------------------------

def spotify_embed(track_id: str):

    if not track_id:
        return

    components.html(
        f"""
        <iframe
            style="border-radius:12px"
            src="https://open.spotify.com/embed/track/{track_id}"
            width="100%"
            height="152"
            frameBorder="0"
            allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy">
        </iframe>
        """,
        height=170,
    )


# -------------------------------------------------------
# Card CSS
# -------------------------------------------------------

st.markdown(
    """
<style>

.dark-card{

background:#1a1a1a;

padding:15px;

border-radius:12px;

margin-bottom:20px;

border:1px solid #333;

}

</style>

""",
    unsafe_allow_html=True,
)

# ==========================================================
# Recommendations Tab
# ==========================================================

with tab1:

    st.header("🎵 Music Recommendations")

    if df.empty:
        st.error("No songs available.")
        st.stop()

    search_choices = build_search_choices(df)

query = st.text_input(
    "🔍 Search Song or Artist"
)

if query:

    matches = fuzzy_search(
        query,
        search_choices,
    )

    if matches:

        options = {
            choice: index
            for choice, score, index in matches
        }

        selected_name = st.selectbox(
            "Matching Songs",
            list(options.keys()),
        )

        selected_track = options[selected_name]

    else:

        st.warning(
            "No matching songs found."
        )

        st.stop()

else:

    selected_track = st.selectbox(
        "Browse Songs",
        range(len(search_choices)),
        format_func=lambda x: search_choices[x],
    )

    if st.button(
        "Get Recommendations",
        use_container_width=True,
    ):

        selected = get_track_details(
            df,
            selected_track,
        )

        st.subheader("Currently Playing")

        col1, col2 = st.columns([2, 3])

        with col1:

            if (
                "track_id" in selected
                and selected["track_id"]
            ):

                spotify_embed(
                    selected["track_id"]
                )

        with col2:

           st.markdown(
    f"""
# 🎵 {selected["track_name"]}

### 👤 {selected["artists"]}

🎼 **Genre:** {selected["track_genre"]}

🔥 **Popularity:** {selected["popularity"] if "popularity" in selected else "N/A"}
"""
)

        st.divider()

        st.subheader("Recommended Songs")

        recommendations = recommend_tracks(
    track_index=selected_track,
    dataframe=df,
    latent_features=latent_features,
    knn=knn,
)

cols = st.columns(2)

for index, row in recommendations.iterrows():

    with cols[index % 2]:

        similarity = row["similarity"] * 100

        ranking = row["ranking_score"] * 100

        popularity = (
            row["popularity"]
            if "popularity" in row
            else "N/A"
        )

        st.markdown(
            f"""
<div class="dark-card">

## 🎵 {row['track_name']}

**👤 Artist**

{row['artists']}

**🎼 Genre**

{row['track_genre']}

---

**🎯 Similarity**

{similarity:.2f}%

**⭐ Ranking Score**

{ranking:.2f}

**🔥 Popularity**

{popularity}

</div>
""",
            unsafe_allow_html=True,
        )

        if pd.notna(row["track_id"]):

            spotify_embed(
                row["track_id"]
            )
# ==========================================================
# Visualization Tab
# ==========================================================

with tab2:

    st.header("📊 Dataset Visualizations")

    st.markdown(
        """
Explore the learned latent space and the relationships
between the audio features used by the recommendation model.
"""
    )

    # ------------------------------------------------------
    # Latent Space
    # ------------------------------------------------------

    st.subheader("3D Latent Space")

    with st.spinner("Generating visualization..."):

        fig = plot_tracks_by_genre(
            latent_features,
            df,
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # ------------------------------------------------------
    # Correlation Heatmap
    # ------------------------------------------------------

    st.subheader("Feature Correlation")

    heatmap = plot_feature_heatmap(
        df,
        NUMERIC_FEATURES,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    st.divider()

    # ------------------------------------------------------
    # Dataset Statistics
    # ------------------------------------------------------

    st.subheader("Dataset Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Songs",
            f"{len(df):,}",
        )

    with col2:

        st.metric(
            "Genres",
            df["track_genre"].nunique(),
        )

    with col3:

        st.metric(
            "Artists",
            df["artists"].nunique(),
        )

    st.divider()

    st.subheader("Genre Distribution")

    genre_counts = (
        df["track_genre"]
        .value_counts()
        .reset_index()
    )

    genre_counts.columns = [
        "Genre",
        "Songs",
    ]

    st.bar_chart(
        genre_counts.set_index("Genre")
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "TuneMatch • Deep Autoencoder + KNN • Built with TensorFlow, Scikit-learn and Streamlit"
)