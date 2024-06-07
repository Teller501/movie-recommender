from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import ssl
import torch
from fastai.collab import *
from fastai.tabular.all import *
import torch.nn.functional as F
from typing import List
import time

app = FastAPI()

learn = load_learner('movie-recommender.pkl')

class UserRating(BaseModel):
    tmdb_id: int
    rating: float

class UserRatings(BaseModel):
    user_id: int
    user_ratings: List[UserRating]

ssl._create_default_https_context = ssl._create_unverified_context

movies_path = 'movies.csv'
ratings_path = 'ratings.csv'

ratings = pd.read_csv(ratings_path, delimiter=',', skiprows=1, header=None, 
                      names=['user','movie','rating','timestamp'])
movies = pd.read_csv(movies_path, usecols=(0,1), names=('movie','title'), header=None)

movies['movie'] = movies['movie'].astype(str)
ratings['movie'] = ratings['movie'].astype(str)

ratings = ratings.merge(movies)

links = pd.read_csv('links.csv')
tmdb_to_internal = dict(zip(links['tmdbId'], links['movieId']))
internal_to_tmdb = dict(zip(links['movieId'], links['tmdbId']))

@app.post("/recommend/")
def recommend_movies(user_ratings: UserRatings):
    user_ratings_dicts = []
    for ur in user_ratings.user_ratings:
        if ur.tmdb_id in tmdb_to_internal:
            internal_id = tmdb_to_internal[ur.tmdb_id]
            user_ratings_dicts.append(
                {"user": user_ratings.user_id, "movie": internal_id, "rating": ur.rating})
        else:
            raise HTTPException(status_code=404, detail=f"TMDB ID {ur.tmdb_id} not found in the dataset")

    new_ratings = pd.concat(
        [ratings, pd.DataFrame(user_ratings_dicts)], ignore_index=True)
    
    new_ratings['movie'] = new_ratings['movie'].astype(str)

    crosstab = pd.crosstab(new_ratings['user'], new_ratings['movie'],
                           values=new_ratings['rating'], aggfunc='sum').fillna(0)

    other_users = crosstab.loc[crosstab.index != user_ratings.user_id].values
    new_user = crosstab.loc[crosstab.index ==
                            user_ratings.user_id].values.reshape(1, -1)

    similarities = F.cosine_similarity(tensor(other_users), tensor(new_user))
    top5_users = similarities.topk(5)

    user_vectors = learn.model.u_weight.weight[1+top5_users.indices, :]
    new_user_vector = user_vectors.mean(dim=0, keepdim=True)
    user_biases = learn.model.u_bias.weight[1+top5_users.indices, :]
    new_user_bias = user_biases.mean()

    pred_ratings = torch.matmul(
        new_user_vector, learn.model.i_weight.weight.T) + learn.model.i_bias.weight.T + new_user_bias
    top5_ratings = pred_ratings.topk(5)

    recommendations_titles = [learn.dls.classes['title'][i] for i in top5_ratings.indices.tolist()[0]]

    print(f"Recommended Movie Titles: {recommendations_titles}")

    recommendations_internal_ids = []
    for title in recommendations_titles:
        movie_id = movies[movies['title'] == title]['movie'].values[0]
        recommendations_internal_ids.append(movie_id)

    print(f"Internal Movie IDs: {recommendations_internal_ids}")

    recommendations_tmdb = [internal_to_tmdb.get(int(movie_id), None) for movie_id in recommendations_internal_ids]
    recommendations_tmdb = [rec for rec in recommendations_tmdb if rec is not None]

    print(f"TMDB IDs: {recommendations_tmdb}")

    updated_ratings_df = pd.DataFrame(user_ratings_dicts)
    updated_ratings_path = 'updated_ratings.csv'

    try:
        updated_ratings = pd.read_csv(updated_ratings_path)
    except FileNotFoundError:
        updated_ratings = pd.DataFrame(columns=['user', 'movie', 'rating', 'timestamp', 'title'])

    updated_ratings = pd.concat([updated_ratings, updated_ratings_df], ignore_index=True)
    updated_ratings.to_csv(updated_ratings_path, index=False)

    print(f"Updated Ratings DataFrame:\n{updated_ratings}")

    return {"recommendations": recommendations_tmdb}

@app.post("/reload-model/")
def reload_model():
    global learn
    learn = load_learner('movie-recommender.pkl')
    return {"detail": "Model reloaded successfully"}

# start: uvicorn api:app --reload
