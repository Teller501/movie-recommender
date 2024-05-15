from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import ssl
import torch
from fastai.collab import *
from fastai.tabular.all import *
import torch.nn.functional as F

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
learn = load_learner('movie-recommender.pkl')

# Define the schema for user input
class UserRating(BaseModel):
    tmdb_id: int
    rating: float

class UserRatings(BaseModel):
    user_id: int
    user_ratings: list[UserRating]

# Dummy ratings DataFrame (assuming it's the same format as in your training)
ssl._create_default_https_context = ssl._create_unverified_context

path = untar_data(URLs.ML_100k)
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None, names=['user','movie','rating','timestamp'])
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1', usecols=(0,1), names=('movie','title'), header=None)
ratings = ratings.merge(movies)

# Load the links.csv file to get mappings
links = pd.read_csv('links.csv')
tmdb_to_internal = dict(zip(links['tmdbId'], links['movieId']))
internal_to_tmdb = dict(zip(links['movieId'], links['tmdbId']))

@app.post("/recommend/")
def recommend_movies(user_ratings: UserRatings):
    # Convert TMDB IDs to internal IDs
    user_ratings_dicts = []
    for ur in user_ratings.user_ratings:
        if ur.tmdb_id in tmdb_to_internal:
            user_ratings_dicts.append({"user": user_ratings.user_id, "movie": tmdb_to_internal[ur.tmdb_id], "rating": ur.rating})
        else:
            raise HTTPException(status_code=404, detail=f"TMDB ID {ur.tmdb_id} not found in the dataset")

    # Add new user ratings to the ratings DataFrame
    new_ratings = pd.concat([ratings, pd.DataFrame(user_ratings_dicts)], ignore_index=True)

    # Create a crosstab
    crosstab = pd.crosstab(new_ratings['user'], new_ratings['movie'], values=new_ratings['rating'], aggfunc='sum').fillna(0)

    # Extract vectors for other users and the new user
    other_users = crosstab.loc[crosstab.index != user_ratings.user_id].values
    new_user = crosstab.loc[crosstab.index == user_ratings.user_id].values.reshape(1, -1)

    # Compute similarities
    similarities = F.cosine_similarity(tensor(other_users), tensor(new_user))
    top5 = similarities.topk(5)

    # Compute new user vector
    user_vectors = learn.model.u_weight.weight[1+top5.indices,:]
    new_user_vector = user_vectors.mean(dim=0, keepdim=True)
    user_biases = learn.model.u_bias.weight[1+top5.indices,:]
    new_user_bias = user_biases.mean()

    # Predict ratings
    pred_ratings = torch.matmul(new_user_vector, learn.model.i_weight.weight.T) + learn.model.i_bias.weight.T + new_user_bias
    top5_ratings = pred_ratings.topk(5)

    # Get recommendations
    recommendations_internal = top5_ratings.indices.tolist()[0]

    # Convert internal IDs back to TMDB IDs
    recommendations_tmdb = [internal_to_tmdb[rec] for rec in recommendations_internal if rec in internal_to_tmdb]

    return {"recommendations": recommendations_tmdb}
