import pandas as pd
import random
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
import json

app = Flask(__name__)
selected_movies_set = set()
recommended_movies_by_genre = {}


def tmdb_request(endpoint, params=None, language='es-ES'):
    base_url = 'https://api.themoviedb.org/3/'
    api_key = '2af2da82d49b988704b95e0a53661965'  # Reemplaza con tu propia API key
    if params is None:
        params = {}
    params['api_key'] = api_key
    params['language'] = language
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_genre_names(genre_ids):
    genre_names = []
    genre_info = tmdb_request('genre/movie/list')
    if genre_info:
        for genre_id in genre_ids:
            for genre in genre_info['genres']:
                if genre['id'] == genre_id:
                    genre_names.append(genre['name'])
                    break
    return genre_names

def get_random_backdrop():
    tmdb_endpoint = 'discover/movie'
    params = {
        'primary_release_date.gte': '2015-01-01',
        'primary_release_date.lte': '2023-12-31',
        'with_backdrop': 'true',
        'sort_by': 'revenue.desc'
    }
    data = tmdb_request(tmdb_endpoint, params)
    if data and data['results']:
        random_movie = random.choice(data['results'])
        return random_movie['backdrop_path']
    return None

def get_tmdb_movies():
    tmdb_endpoint = 'discover/movie'
    params = {
        'page': 1,
        'primary_release_date.gte': '2010-01-01',
        'sort_by': 'revenue.desc'
    }
    return tmdb_request(tmdb_endpoint, params)

tmdb_data = get_tmdb_movies()
if tmdb_data and 'results' in tmdb_data:
    movies = pd.DataFrame(tmdb_data['results'])
    movies['features'] = movies['title'] + ' ' + movies['overview']
    movies['genre_names'] = movies['genre_ids'].apply(get_genre_names)
else:
    movies = pd.read_csv('data/movies.csv').head(100)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/')
def index():
    random_movies = movies.sample(n=20)
    hero_movie = random_movies.sample(n=1).iloc[0]
    movie_details = tmdb_request(f'movie/{hero_movie["id"]}', language='es-ES')
    hero_movie_synopsis = movie_details['overview'] if movie_details else ''
    hero_movie_director = movie_details.get('director', 'Director Desconocido')
    hero_movie_backdrop_url = f"https://image.tmdb.org/t/p/w1280{hero_movie['backdrop_path']}" if hero_movie['backdrop_path'] else None
    hero_movie_poster_path = f"https://image.tmdb.org/t/p/w500{hero_movie['poster_path']}"
    
    return render_template('index.html', 
                           random_movies=random_movies, 
                           hero_movie_title=hero_movie['title'], 
                           hero_movie_synopsis=hero_movie_synopsis, 
                           hero_movie_director=hero_movie_director,
                           hero_movie_poster_path=hero_movie_poster_path,
                           hero_movie_backdrop_url=hero_movie_backdrop_url)

def get_recommendations(movie_title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

def search_movies_by_genre(genre_ids):
    tmdb_endpoint = 'discover/movie'
    params = {
        'with_genres': ','.join(map(str, genre_ids)),
        'sort_by': 'popularity.desc',
        'page': 1
    }
    return tmdb_request(tmdb_endpoint, params)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    selected_movies = request.form.getlist('selected_movies')
    selected_movie = selected_movies[0]

    selected_movie_info = tmdb_request('search/movie', {'query': selected_movie})
    unique_movies = set()
    recommendations_list = []
    genre_recommendations = {}
    
    genre_info = tmdb_request('genre/movie/list')

    if selected_movie_info and 'results' in selected_movie_info:
        selected_movie_data = selected_movie_info['results'][0]
        selected_movie_genre_ids = selected_movie_data['genre_ids'][:3]
        selected_movie_poster_path = selected_movie_data['poster_path']
        selected_movie_genres = get_genre_names(selected_movie_genre_ids)

        for genre_id in selected_movie_genre_ids:
            genre_movies = search_movies_by_genre([genre_id])
            if genre_movies and 'results' in genre_movies:
                genre_name = next((genre['name'] for genre in genre_info['genres'] if genre['id'] == genre_id), None)
                if genre_name:
                    genre_recommendations[genre_name] = []

                    # Filtra las películas recomendadas para que no sean las mismas que las seleccionadas
                    recommended_movies = [movie for movie in genre_movies['results'] if movie['title'] != selected_movie]

                    # Filtra las películas recomendadas para que no se repitan en otras categorías
                    for movie in recommended_movies:
                        if movie['title'] not in unique_movies and len(genre_recommendations[genre_name]) < 10:
                            recommendations_list.append(movie)
                            unique_movies.add(movie['title'])
                            genre_recommendations[genre_name].append(movie)

                    # Agrega las películas seleccionadas al conjunto de seguimiento
                    selected_movies_set.add(selected_movie)

    return render_template('recommendations.html', 
                           recommended_movies=recommendations_list, 
                           genre_recommendations=genre_recommendations,
                           selected_movie_genres=selected_movie_genres,
                           selected_movie_poster_path=f"https://image.tmdb.org/t/p/w500{selected_movie_poster_path}")

if __name__ == '__main__':
    app.run(debug=True)
