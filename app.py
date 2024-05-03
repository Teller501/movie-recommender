from flask import Flask, request, jsonify
import pandas as pd
import pickle
from surprise import KNNWithMeans, Dataset, Reader
from flask import current_app, g

app = Flask(__name__)

def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def get_data():
    if 'data' not in g:
        g.data = pd.read_csv('ratings.csv')
    return g.data

@app.route('/predict', methods=['POST'])
def predict():
    algo = load_model()
    data = request.get_json()
    user_id = data['userId']
    movie_id = data['movieId']
    prediction = algo.predict(user_id, movie_id)
    return jsonify({'prediction': prediction.est})

@app.route('/recommend', methods=['POST'])
def recommend():
    algo = load_model()
    data = request.get_json()
    user_id = data['userId']
    n = data.get('n', 10)  # Default to 10 if not provided

    # Load ratings data safely
    ratings = get_data()

    # Get recommendations
    recommendations = get_top_n_recommendations(algo, ratings, user_id, n)

    # Prepare response
    response = [{
        'movieId': int(movie[0]),  # Ensure movieId is an int
        'predictedRating': float(movie[1])  # Ensure predictedRating is a float
    } for movie in recommendations]

    return jsonify(response)

@app.route('/update_dataset', methods=['POST'])
def update_dataset():
    new_user_data = request.get_json()
    new_user_id = new_user_data['userId']
    new_user_ratings = new_user_data['ratings']
    ratings = get_data()
    updated_ratings = add_new_user_ratings(ratings, new_user_id, new_user_ratings)
    g.data = updated_ratings  # update global ratings data

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(updated_ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = load_model()
    algo.fit(trainset)
    save_model(algo)

    return jsonify({'message': 'Dataset and model updated successfully'})

def add_new_user_ratings(ratings, new_user_id, movie_ratings):
    new_ratings = pd.DataFrame({
        'userId': [new_user_id] * len(movie_ratings),
        'movieId': [mr[0] for mr in movie_ratings],
        'rating': [mr[1] for mr in movie_ratings]
    })
    return pd.concat([ratings, new_ratings], ignore_index=True)

def get_top_n_recommendations(algo, ratings, user_id, n=10):
    movies = pd.read_csv('movies.csv')
    all_movies = movies['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    unrated_movies = [movie for movie in all_movies if movie not in rated_movies]

    predictions = []
    for movie_id in unrated_movies:
        predicted_rating = algo.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = predictions[:n]
    return top_n_movies


if __name__ == '__main__':
    app.run(debug=True)
