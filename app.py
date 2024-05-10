from flask import Flask, request, jsonify
from fastai.tabular.all import *
from fastai.collab import *
import pandas as pd
import torch.nn as nn
from pymongo import MongoClient
from dotenv import load_dotenv
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

load_dotenv()

app = Flask(__name__)

# Load your trained model
path = untar_data(URLs.ML_100k)
learn = load_learner('movie-recommender.pkl')

mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri)
db = client['cinematch']
ratings_collection = db['ratings']


def get_recommendations(user_ratings, user_id):
    user_ratings_dicts = []
    for movie_id, rating in user_ratings:
        user_ratings_dicts.append({"user": user_id, "movie": movie_id, "rating": rating})

    # Add or update the user's ratings in MongoDB
    ratings_collection.update_one({"user": user_id}, {"$set": {"ratings": user_ratings_dicts}}, upsert=True)

    # Load the user's ratings from MongoDB
    user_data = ratings_collection.find_one({"user": user_id})
    if user_data:
        ratings = pd.DataFrame(user_data["ratings"])
    else:
        ratings = pd.DataFrame(columns=['user', 'movie', 'rating'])

    # Add the new user's ratings to the existing dataset
    ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names=['user', 'movie', 'rating', 'timestamp'])
    # Convert user_ratings_dicts to a DataFrame
    user_ratings_df = pd.DataFrame(user_ratings_dicts)

    # Append the new user's ratings to the existing dataset
    new_ratings = pd.concat([ratings, user_ratings_df], ignore_index=True)

    # Cross-tabulate the data and find the new user's row
    crosstab = pd.crosstab(new_ratings['user'], new_ratings['movie'], values=new_ratings['rating'], aggfunc='sum').fillna(0)
    other_users = crosstab.values[:-1]
    new_user = crosstab.values[-1].reshape(1, -1)

    # Find the top 5 most similar users using cosine similarity
    similarities = nn.CosineSimilarity()(tensor(other_users), tensor(new_user))
    top5 = similarities.topk(5)

    # Calculate the new user's vector and bias
    user_vectors = learn.u_weight.weight[1+top5.indices, :]
    new_user_vector = user_vectors.mean(dim=0, keepdim=True)
    user_biases = learn.u_bias.weight[1+top5.indices, :]
    new_user_bias = user_biases.mean()

    # Predict the new user's ratings for all movies
    pred_ratings = torch.matmul(new_user_vector, learn.i_weight.weight.T) + learn.i_bias.weight.T + new_user_bias

    # Get the top 5 movie recommendations
    top5_ratings = pred_ratings.topk(5)
    recommendations = [(learn.classes['title'][index], index) for index in top5_ratings.indices.tolist()[0]]

    # Format recommendations as requested
    formatted_recommendations = [{"id": id, "title": title} for title, id in recommendations]

    return formatted_recommendations

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid request data'}), 400

    user_id = data.get('user_id')
    user_ratings = data.get('user_ratings')

    if user_id is None or user_ratings is None:
        return jsonify({'error': 'Missing user_id or user_ratings in the request'}), 400

    if not isinstance(user_ratings, list):
        return jsonify({'error': 'user_ratings must be a list of [movie_id, rating] pairs'}), 400

    # Get the list of recommended movies
    recommendations = get_recommendations(user_ratings, user_id)

    # Return the recommendations as JSON
    return jsonify({'data': recommendations})


if __name__ == '__main__':
    app.run(debug=True)
