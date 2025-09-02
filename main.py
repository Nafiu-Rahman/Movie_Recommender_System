import pickle
import streamlit as st
import requests
from huggingface_hub import hf_hub_download
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
h1 { text-align: center; color: #FF4B4B; }
.movie-title { font-size: 16px; font-weight: bold; text-align: center; min-height: 3rem; color: #FFFFFF; }
.stButton > button { width: 100%; border-radius: 50px; font-size: 18px; font-weight: bold; margin: 0.5em 0; background-color: #FF4B4B; color: white; }
.stButton > button:hover { background-color: #FFFFFF; color: #FF4B4B; border: 2px solid #FF4B4B; }
</style>
""", unsafe_allow_html=True)

# --- API AND RECOMMENDATION FUNCTIONS ---
@st.cache_data
def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return "https://via.placeholder.com/500x750.png?text=API+Error"

def recommend(movie):
    """Recommends 5 movies based on similarity."""
    try:
        index = st.session_state.movies[st.session_state.movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(st.session_state.similarity[index])), reverse=True, key=lambda x: x[1])
        
        recommended_movie_names = []
        recommended_movie_posters = []
        
        for i in distances[1:6]:
            movie_id = st.session_state.movies.iloc[i[0]].movie_id
            recommended_movie_posters.append(fetch_poster(movie_id))
            recommended_movie_names.append(st.session_state.movies.iloc[i[0]].title)
            
        return recommended_movie_names, recommended_movie_posters
    except IndexError:
        st.error("Movie not found in the dataset. Please select another one.")
        return [], []

# --- LOAD DATA FROM HUGGING FACE HUB ---
@st.cache_resource
def load_model_files():
    """Load model files from Hugging Face Hub with caching."""
    try:
        # Set environment variable to use /tmp for cache
        os.environ['HF_HOME'] = '/tmp/huggingface'
        os.makedirs('/tmp/huggingface', exist_ok=True)
        
        # Download files using Hugging Face Hub
        movie_list_path = hf_hub_download(
            repo_id="N4F1U/Movie_Recommender_tmdb",
            filename="movie_list.pkl",
            cache_dir="/tmp/huggingface"
        )
        
        similarity_path = hf_hub_download(
            repo_id="N4F1U/Movie_Recommender_tmdb",
            filename="similarity.pkl",
            cache_dir="/tmp/huggingface"
        )
        
        # Load the pickle files
        with open(movie_list_path, 'rb') as f:
            movies_data = pickle.load(f)
        
        with open(similarity_path, 'rb') as f:
            similarity_data = pickle.load(f)
        
        return movies_data, similarity_data
        
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.error(f"Error type: {type(e).__name__}")
        return None, None

# --- INITIALIZE SESSION STATE ---
if 'movies' not in st.session_state or 'similarity' not in st.session_state:
    movies, similarity = load_model_files()
    if movies is not None and similarity is not None:
        st.session_state.movies = movies
        st.session_state.similarity = similarity
        st.session_state.movie_list = movies['title'].values
    else:
        st.error("Failed to load model data. Please check your repository and try again.")
        st.stop()

# --- APP LAYOUT ---
st.title('Movie Recommender System 🍿')

# Center the selection box and button using columns
_, col_centered, _ = st.columns([1, 2, 1])
with col_centered:
    selected_movie = st.selectbox(
        "Type or select a movie to get recommendations",
        st.session_state.movie_list
    )

    if st.button('Show Recommendation'):
        with st.spinner('Finding similar movies for you...'):
            recommended_names, recommended_posters = recommend(selected_movie)
        
        if recommended_names:
            st.success("Here are your top 5 recommendations!")
            cols = st.columns(5, gap="medium")
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f'<p class="movie-title">{recommended_names[i]}</p>', unsafe_allow_html=True)
                    st.image(recommended_posters[i], use_container_width='always')
