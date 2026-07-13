import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.models.artifacts import load_artifacts
from src.models.explain import explain_recommendation
from src.models.genre import (
    GENRE_EXPLORER_OPTIONS,
    generate_genre_playlist,
    recommend_by_genre,
)
from src.models.mood import (
    MOOD_ORDER,
    assign_moods,
    explain_mood_fit,
    recommend_by_mood,
)
from src.models.preprocessing import NUMERIC_FEATURES
from src.models.recommender import (
    INTENT_WEIGHT_PROFILES,
    get_track_details,
    recommend_tracks,
)
from src.models.search import build_search_index, intelligent_search
from src.models.visualization import plot_feature_heatmap, plot_tracks_by_genre


st.set_page_config(
    page_title="TuneMatch",
    page_icon="🎵",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(249, 115, 22, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 24%),
            linear-gradient(180deg, #07111F 0%, #091827 100%);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15, 33, 56, 0.95), rgba(8, 20, 36, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.20);
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 0.5rem 1rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(15, 33, 56, 0.92);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #F97316 0%, #FB923C 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.8rem 1rem;
        font-weight: 700;
        box-shadow: 0 14px 28px rgba(249, 115, 22, 0.28);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #EA580C 0%, #F97316 100%);
        color: white;
    }

    .tm-hero {
        padding: 1.8rem 1.8rem 1.6rem 1.8rem;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(249, 115, 22, 0.22), rgba(14, 165, 233, 0.14)),
            rgba(8, 20, 36, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 22px 48px rgba(0, 0, 0, 0.22);
        margin-bottom: 1rem;
    }

    .tm-kicker {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(249, 115, 22, 0.16);
        color: #FDBA74;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .tm-title {
        font-size: 3rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
        color: #F8FAFC;
    }

    .tm-subtitle {
        margin-top: 0.9rem;
        max-width: 760px;
        font-size: 1.02rem;
        line-height: 1.7;
        color: #CBD5E1;
    }

    .tm-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }

    .tm-pill {
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #E2E8F0;
        font-size: 0.9rem;
    }

    .tm-section-copy {
        color: #A8B3C7;
        margin-bottom: 0.8rem;
    }

    .tm-card {
        background: linear-gradient(180deg, rgba(15, 33, 56, 0.96), rgba(8, 20, 36, 0.96));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.55rem 1rem;
        box-shadow: 0 18px 34px rgba(0, 0, 0, 0.18);
        margin-bottom: 0.8rem;
    }

    .tm-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }

    .tm-card-meta {
        color: #A8B3C7;
        margin-bottom: 0.2rem;
    }

    .tm-card-badge {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.12);
        color: #7DD3FC;
        font-size: 0.82rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tm-hero">
        <div class="tm-kicker">Music Intelligence</div>
        <h1 class="tm-title">TuneMatch AI</h1>
        <div class="tm-subtitle">
            Discover songs through hybrid recommendation signals that blend deep latent similarity,
            audio features, genre intelligence, popularity cues, mood profiling, and playlist generation.
        </div>
        <div class="tm-pill-row">
            <div class="tm-pill">Hybrid Recommender</div>
            <div class="tm-pill">Mood Discovery</div>
            <div class="tm-pill">Genre Explorer</div>
            <div class="tm-pill">Explainable AI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    return load_artifacts()


@st.cache_data
def build_catalog_search_v2(dataframe: pd.DataFrame):
    return build_search_index(dataframe)


@st.cache_data
def build_mood_catalog(dataframe: pd.DataFrame):
    return assign_moods(dataframe)


def spotify_embed(track_id: str) -> None:
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


def render_track_summary(track: pd.Series) -> None:
    popularity = (
        track["popularity"]
        if "popularity" in track
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="tm-card">
            <div class="tm-card-title">{track["track_name"]}</div>
            <div class="tm-card-meta">Artist: {track["artists"]}</div>
            <div class="tm-card-meta">Genre: {track["track_genre"]}</div>
            <div class="tm-card-badge">Popularity {popularity}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(
    recommendation: pd.Series,
    selected_track: pd.Series,
) -> None:
    explanation = explain_recommendation(
        selected_track,
        recommendation,
    )
    similarity = explanation["similarity_percent"]
    latent_similarity = explanation["latent_similarity_percent"]
    audio_similarity = explanation["audio_similarity_percent"]
    genre_score = explanation["genre_score_percent"]
    ranking = explanation["ranking_score_percent"]
    popularity_score = explanation["popularity_score_percent"]
    source_support = explanation["source_support_percent"]
    popularity = explanation["popularity"]

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="tm-card">
                <div class="tm-card-title">{recommendation["track_name"]}</div>
                <div class="tm-card-meta">{recommendation["artists"]}</div>
                <div class="tm-card-meta">Genre: {recommendation["track_genre"]}</div>
                <div class="tm-card-badge">Popularity {popularity}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(recommendation["similarity"]))
        st.caption(
            f"{similarity:.2f}% similarity | {ranking:.2f} ranking score | popularity {popularity}"
        )

        st.write("Why this song was chosen")
        st.write(explanation["summary"])
        st.write(
            f"Hybrid breakdown: latent {latent_similarity:.2f}% | audio {audio_similarity:.2f}% | genre {genre_score:.2f}% | popularity {popularity_score:.2f}% | source support {source_support:.2f}%."
        )

        if explanation["same_genre"]:
            st.write("Genre alignment: exact genre match.")
        elif explanation["same_genre_family"]:
            st.write("Genre alignment: matched through a broader genre family.")

        active_sources = []
        if explanation["source_latent"]:
            active_sources.append("latent")
        if explanation["source_audio"]:
            active_sources.append("audio")
        if explanation["source_genre"]:
            active_sources.append("genre")
        if explanation["source_popularity"]:
            active_sources.append("popularity")
        if active_sources:
            st.write(
                f'Retrieval sources: {", ".join(active_sources)}.'
            )

        if explanation["top_reasons"]:
            st.write("Main reasons:")
            for reason in explanation["top_reasons"]:
                st.write(f"- {reason}")

        with st.expander("Score details"):
            score_breakdown = pd.DataFrame(
                {
                    "Component": [
                        "Latent similarity",
                        "Audio similarity",
                        "Genre score",
                        "Popularity score",
                        "Source support",
                    ],
                    "Percent": [
                        latent_similarity,
                        audio_similarity,
                        genre_score,
                        popularity_score,
                        source_support,
                    ],
                }
            )
            st.bar_chart(
                score_breakdown.set_index("Component")
            )

        if explanation["feature_matches"]:
            st.write("Feature-level similarity:")
            for match in explanation["feature_matches"][:5]:
                st.write(
                    f'- {match["label"]}: '
                    f'{match["closeness"] * 100:.1f}% close '
                    f'(difference {match["difference"]:.3f})'
                )

        track_id = recommendation.get("track_id")
        if pd.notna(track_id):
            spotify_embed(track_id)


resources = load_resources()
df = resources["dataframe"]
latent_features = resources["latent_features"]
knn = resources["knn"]
df_with_moods = build_mood_catalog(df)
search_index = build_catalog_search_v2(df)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Recommendations",
        "Mood Recommendations",
        "Genre Explorer",
        "Visualization",
    ]
)

col1, col2, col3 = st.columns(3)
col1.metric("Songs", f"{len(df):,}")
col2.metric("Artists", df["artists"].nunique())
col3.metric("Genres", df["track_genre"].nunique())

st.divider()

with tab1:
    st.header("Music Recommendations")
    st.markdown(
        '<div class="tm-section-copy">Search by title or artist, choose a recommendation style, and inspect the hybrid score breakdown behind each suggestion.</div>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error("No songs available.")
        st.stop()

    search_query = st.text_input(
        "Search by song title or artist",
        placeholder="Try: blinding lights, weeknd, shape of you, calm down...",
        help="Supports partial matches, artist search, autocomplete-style suggestions, and typo correction.",
    )

    recommendation_intent = st.selectbox(
        "Recommendation style",
        list(INTENT_WEIGHT_PROFILES.keys()),
        help="Adjusts the hybrid ranking profile for same-vibe, same-genre, discovery, popularity, or energy-focused recommendations.",
    )

    if search_query.strip():
        matches = intelligent_search(
            search_query,
            search_index,
            limit=12,
        )

        if not matches:
            st.warning(
                "No close matches found. Try a different song title or artist."
            )
            st.stop()

        st.caption("Autocomplete suggestions")

        selected_option = st.selectbox(
            "Matching tracks",
            matches,
            format_func=lambda match: (
                f'{match["label"]}  |  match {match["score"]:.0f}%'
            ),
        )
        selected_track_index = selected_option["index"]
    else:
        browse_options = [
            {
                "index": entry.index,
                "label": entry.label,
            }
            for entry in search_index
        ]
        selected_option = st.selectbox(
            "Browse songs",
            browse_options,
            format_func=lambda option: option["label"],
        )
        selected_track_index = selected_option["index"]

    if st.button(
        "Get Recommendations",
        use_container_width=True,
    ):
        selected_track = get_track_details(
            df,
            selected_track_index,
        )

        st.subheader("Currently Playing")
        now_playing_col, details_col = st.columns([2, 3])

        with now_playing_col:
            track_id = selected_track.get("track_id")
            if pd.notna(track_id):
                spotify_embed(track_id)

        with details_col:
            render_track_summary(selected_track)

        st.divider()
        st.subheader("Recommended Songs")

        recommendations = recommend_tracks(
            track_index=selected_track_index,
            dataframe=df,
            latent_features=latent_features,
            knn=knn,
            intent=recommendation_intent,
        )

        left_col, right_col = st.columns(2)

        for offset, (_, row) in enumerate(
            recommendations.iterrows()
        ):
            with (left_col if offset % 2 == 0 else right_col):
                render_recommendation_card(
                    row,
                    selected_track,
                )

with tab2:
    st.header("Mood Recommendations")
    st.markdown(
        '<div class="tm-section-copy">Pick a listening mood and TuneMatch will surface tracks whose audio signatures best match that context.</div>',
        unsafe_allow_html=True,
    )

    selected_mood = st.selectbox(
        "Choose a mood",
        MOOD_ORDER,
    )

    mood_recommendations = recommend_by_mood(
        df_with_moods,
        selected_mood,
        n_recommendations=12,
    )

    st.caption(
        "Generated from energy, tempo, valence, danceability, acousticness, and related audio features."
    )

    mood_left_col, mood_right_col = st.columns(2)

    for offset, (_, row) in enumerate(
        mood_recommendations.iterrows()
    ):
        with (
            mood_left_col
            if offset % 2 == 0
            else mood_right_col
        ):
            mood_score = float(
                row[f"{selected_mood.lower()}_score"]
            ) * 100
            mood_match_score = float(
                row["mood_match_score"]
            ) * 100
            reasons = explain_mood_fit(
                row,
                selected_mood,
            )

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="tm-card">
                        <div class="tm-card-title">{row["track_name"]}</div>
                        <div class="tm-card-meta">{row["artists"]}</div>
                        <div class="tm-card-meta">Genre: {row["track_genre"]}</div>
                        <div class="tm-card-badge">Primary mood {row["mood"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(
                    float(row[f"{selected_mood.lower()}_score"])
                )
                st.caption(
                    f"{mood_score:.2f}% mood fit | {mood_match_score:.2f}% final mood score"
                )
                st.write(
                    f"This song was selected for {selected_mood.lower()} because its audio profile strongly matches that mood."
                )

                for reason in reasons:
                    st.write(f"- {reason}")

                track_id = row.get("track_id")
                if pd.notna(track_id):
                    spotify_embed(track_id)

with tab3:
    st.header("Genre Explorer")
    st.markdown(
        '<div class="tm-section-copy">Browse curated genre families and generate ready-to-share playlists without typing a search query.</div>',
        unsafe_allow_html=True,
    )

    selected_genre = st.selectbox(
        "Choose a genre",
        GENRE_EXPLORER_OPTIONS,
    )

    generated_playlist = generate_genre_playlist(
        df,
        selected_genre,
        playlist_size=20,
    )

    genre_recommendations = recommend_by_genre(
        df,
        selected_genre,
        n_recommendations=12,
    )

    st.subheader(f"{selected_genre} Playlist Generator")

    if generated_playlist.empty:
        st.warning(
            f"No playlist could be generated for the {selected_genre} explorer group."
        )
    else:
        st.caption(
            f"Generated a {len(generated_playlist)}-song playlist for {selected_genre}."
        )

        playlist_table = generated_playlist[
            ["track_name", "artists", "track_genre"]
        ].copy()
        playlist_table.columns = [
            "Track",
            "Artist",
            "Genre",
        ]
        st.dataframe(
            playlist_table,
            use_container_width=True,
            hide_index=True,
        )

    if genre_recommendations.empty:
        st.warning(
            f"No tracks were found for the {selected_genre} explorer group."
        )
    else:
        st.caption(
            f"Showing browseable {selected_genre.lower()} picks without using search."
        )

        genre_left_col, genre_right_col = st.columns(2)

        for offset, (_, row) in enumerate(
            genre_recommendations.iterrows()
        ):
            with (
                genre_left_col
                if offset % 2 == 0
                else genre_right_col
            ):
                popularity = (
                    row["popularity"]
                    if "popularity" in row
                    else "N/A"
                )

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="tm-card">
                            <div class="tm-card-title">{row["track_name"]}</div>
                            <div class="tm-card-meta">{row["artists"]}</div>
                            <div class="tm-card-meta">Genre: {row["track_genre"]}</div>
                            <div class="tm-card-badge">Popularity {popularity}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"This track appears in the {selected_genre.lower()} explorer because its dataset genre maps into that family."
                    )

                    track_id = row.get("track_id")
                    if pd.notna(track_id):
                        spotify_embed(track_id)

with tab4:
    st.header("Dataset Visualizations")
    st.markdown(
        '<div class="tm-section-copy">Explore the learned latent space and the feature relationships that power the recommendation engine.</div>',
        unsafe_allow_html=True,
    )

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
    st.subheader("Dataset Statistics")

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("Songs", f"{len(df):,}")
    with stat_col2:
        st.metric("Genres", df["track_genre"].nunique())
    with stat_col3:
        st.metric("Artists", df["artists"].nunique())

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

st.divider()
st.caption(
    "TuneMatch | Hybrid music recommendation, mood discovery, genre exploration, and explainable AI in one polished Streamlit experience."
)
