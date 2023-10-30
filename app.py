import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request

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

def get_recommendations(movie_title, cosine_sim=cosine_sim, movies_data=movies_with_ratings):
    idx = movies_data[movies_data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Obtén las calificaciones promedio para las películas recomendadas
    recommended_movies = movies_data.iloc[movie_indices]
    recommended_movies = recommended_movies[recommended_movies['rating'] >= 3.5]

    return recommended_movies[['title', 'rating']]

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
        recommendations.extend(get_recommendations(movie_title))
    
    # Elimina duplicados y las películas ya seleccionadas
    recommendations = list(set(recommendations) - set(selected_movies))
    
    # Obtiene las 10 películas recomendadas finales
    recommended_movies = recommendations[:10]

    return render_template('recommendations.html', recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
