import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


###### helper functions. Use them when needed #######
def get_title_from_index(index):
    return df[df.imdb_title_id == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["imdb_title_id"].values[0]


##################################################
st.title("Film Recommendor System")
st.write(
    "This system predicts is a user willlike a certain movie on the basis of its ('genre', 'director', 'writer', 'actors','description', 'language')")

df = pd.read_csv("Movies.csv")
# print df.columns
##Step 2: Select Features

st.subheader('RAW DATA')
with st.echo('below'):
    df = pd.read_csv("Movies.csv")
    df = df.head(10000)
    data = df.loc[:, ['imdb_title_id', 'title', 'genre', 'director', 'writer', 'actors', 'description', 'language']]
    features = ['genre', 'director', 'writer', 'actors', 'description', 'language']
    for feature in features:
        df[feature] = df[feature].fillna('')

    st.write(data)


with st.echo('below'):
    def combine_features(row):
        try:
            return row['genre'] + " " + row["director"] + " " + row["writer"] + " " + row["actors"] + " " + row[
                "description"] + " " + row["language"]
        except:
            print("Error:", row)


    df["combined_features"] = df.apply(combine_features, axis=1)
    FEATURES = df["combined_features"]
    st.subheader('Extracted Features / Vocabulary for Text Classification')
    st.write(FEATURES)

cv = CountVectorizer()
count_matrix = cv.fit_transform(FEATURES)

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = st.sidebar.text_input("Name a Movie You Like..!", value='', max_chars=None, key=None, type='default')
if st.sidebar.button("Predict Reseults"):
    ## Step 6: Get index of this movie from its title
    movie_index = get_index_from_title(movie_user_likes)

    similar_movies = list(enumerate(cosine_sim[movie_index]))

    ## Step 7: Get a list of similar movies in descending order of similarity score
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    st.subheader("Scores")
    st.write(sorted_similar_movies[:50])
    ## Step 8: Print titles of first 50 movies
    i = 0
    st.subheader("Best 50 Matches are :")
    results = [''] * 51

    for element in sorted_similar_movies:
        results[i] = get_title_from_index(element[0])
        i = i + 1
        if i > 50:
            break
    st.write(results)
