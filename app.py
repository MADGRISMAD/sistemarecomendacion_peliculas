import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request
import requests  # Importa la biblioteca requests

# Define tu clave de API de TMDb
TMDB_API_KEY = "15d2ea6d0dc1d476efbca3eba2b9bbfb"

# Carga los datos
movies = pd.read_csv('data/movies.csv').head(100)
ratings = pd.read_csv('data/ratings.csv')

# Combina título y géneros en una sola columna de características
movies['features'] = movies['title'] + ' ' + movies['genres']

# Fusiona las calificaciones con las películas
movies_with_ratings = movies.merge(ratings, on='movieId')

# Vectorización de características usando TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['features'])

# Cálculo de similitud del coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Selecciona 20 películas aleatorias
random_movies = movies.sample(n=20)
random_movies = random_movies.merge(ratings.groupby('movieId')['rating'].mean().reset_index(), on='movieId', how='left')

# Simula la selección del usuario de 5 películas
user_selection = random_movies.sample(n=5)['title']

def get_movie_info(movie_title, api_key=TMDB_API_KEY):
    # Realiza una solicitud a la API de TMDb para obtener información de la película
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': api_key,
        'query': movie_title
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        movie_data = response.json()
        if 'results' in movie_data and len(movie_data['results']) > 0:
            return movie_data['results'][0]
    return None

app = Flask(__name__)

# Obtén las 20 películas aleatorias cuando se inicia la aplicación
random_movies = movies.sample(n=20)

@app.route('/')
def index():
    return render_template('index.html', random_movies=random_movies)

@app.route('/recommendations', methods=['POST'])
def get_user_recommendations():
    selected_movies = request.form.getlist('selected_movies')
    
    # Genera recomendaciones basadas en las películas seleccionadas
    recommendations = []
    for movie_title in selected_movies:
        movie_info = get_movie_info(movie_title)
        if movie_info:
            recommendations.extend(get_recommendations(movie_title))
    
    # Elimina duplicados y las películas ya seleccionadas
    recommendations = list(set(recommendations) - set(selected_movies))
    
    # Obtiene las 10 películas recomendadas finales
    recommended_movies = recommendations[:10]

    return render_template('recommendations.html', recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
