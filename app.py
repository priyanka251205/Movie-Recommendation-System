# app.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Load & Clean Data ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("BollywoodMovieDetail.csv")

    # Fill missing values in text columns only
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('')

    # Replace '-' and 'nan' in genre column globally
    df['genre'] = df['genre'].replace('-', '').replace('nan', '')

    # Create 'tags' from multiple features
    df['tags'] = df['genre'] + ' ' + df['actors'] + ' ' + df['directors'] + ' ' + df['writers']
    df['tags'] = df['tags'].str.lower()

    return df

# ------------------- Generate Similarity ---------------------
@st.cache_data
def generate_similarity_matrix(df):
    cv = CountVectorizer(stop_words='english')
    vectors = cv.fit_transform(df['tags'])
    similarity = cosine_similarity(vectors)
    return similarity

# ------------------- Recommendation Function ---------------------
def recommend(title, df, similarity, top_n=5):
    title = title.lower()
    if title not in df['title'].str.lower().values:
        return []
    
    idx = df[df['title'].str.lower() == title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df.iloc[i[0]]['title'] for i in sorted_scores]

# ------------------- Streamlit UI ---------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")
st.title("🎥 Bollywood Movie Recommendation System")

df = load_data()
similarity = generate_similarity_matrix(df)

# Dropdown to select a movie
movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)

# Recommend button
if st.button("Recommend"):
    results = recommend(selected_movie, df, similarity)
    if results:
        st.success(f"Top 5 movies similar to **{selected_movie}**:")
        for movie in results:
            st.write(f"👉 {movie}")
    else:
        st.error("Movie not found or no recommendations available.")

# ------------------- Optional EDA (Release Year Only) ---------------------
with st.expander("📊 Show Movie Insights"):
    st.subheader("🎞 Movies by Release Year")
    if 'releaseYear' in df.columns:
        year_count = df['releaseYear'].value_counts().sort_index()
        st.bar_chart(year_count)
