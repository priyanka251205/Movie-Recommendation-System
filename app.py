import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- OMDb API ---------------------
OMDB_API_KEY = "d388665"  # 🔑 Replace with your real OMDb key

def fetch_poster_omdb(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("Poster") and data["Poster"] != "N/A":
        return data["Poster"]
    return "https://via.placeholder.com/140x200?text=No+Image"

# ------------------- Load Data ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("BollywoodMovieDetail.csv")
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('')
    df['genre'] = df['genre'].replace('-', '').replace('nan', '')
    df['tags'] = df['genre'] + ' ' + df['actors'] + ' ' + df['directors'] + ' ' + df['writers']
    df['tags'] = df['tags'].str.lower()
    return df

# ------------------- Similarity Matrix ---------------------
@st.cache_data
def generate_similarity_matrix(df):
    cv = CountVectorizer(stop_words='english')
    vectors = cv.fit_transform(df['tags'])
    similarity = cosine_similarity(vectors)
    return similarity

# ------------------- Recommend ---------------------
def recommend(title, df, similarity, top_n=5):
    title = title.lower()
    if title not in df['title'].str.lower().values:
        return [], []

    idx = df[df['title'].str.lower() == title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommended_titles = []
    poster_urls = []
    for i in sorted_scores:
        movie = df.iloc[i[0]]['title']
        recommended_titles.append(movie)
        poster_urls.append(fetch_poster_omdb(movie))

    return recommended_titles, poster_urls

# ------------------- Streamlit UI ---------------------
st.set_page_config(page_title="🎬 Bollywood Movie Recommender", layout="wide")
st.title("🍿 Bollywood Movie Recommendation System")

df = load_data()
similarity = generate_similarity_matrix(df)

movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("🎥 Select a movie to get recommendations:", movie_list)

if st.button("🔍 Recommend"):
    names, posters = recommend(selected_movie, df, similarity)
    if names:
        st.success(f"Top 5 movies similar to **{selected_movie}**:")
        cols = st.columns(5)
        for i in range(len(names)):
            with cols[i]:
                st.image(posters[i], width=140)
                st.caption(names[i])
    else:
        st.error("Movie not found or no recommendations available.")

# ------------------- Optional EDA ---------------------
with st.expander("📊 Show Movie Insights"):
    st.subheader("🎞 Movies by Release Year")
    if 'releaseYear' in df.columns:
        year_count = df['releaseYear'].value_counts().sort_index()
        st.bar_chart(year_count)
