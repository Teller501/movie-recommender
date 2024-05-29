from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import ssl
import torch
from fastai.collab import *
from fastai.tabular.all import *
import torch.nn.functional as F
import os
from typing import List



app = FastAPI()

learn = load_learner('movie-recommender.pkl')

class UserRating(BaseModel):
    tmdb_id: int
    rating: float


class UserRatings(BaseModel):
    user_id: int
    user_ratings: List[UserRating]


ssl._create_default_https_context = ssl._create_unverified_context
path = untar_data(URLs.ML_100k)
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user', 'movie', 'rating', 'timestamp'])
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',
                     usecols=(0, 1), names=('movie', 'title'), header=None)
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

    new_ratings_df = pd.DataFrame(user_ratings_dicts)
    global ratings
    ratings = pd.concat([ratings, new_ratings_df], ignore_index=True)

    ratings.to_csv('updated_ratings.csv', index=False)

    crosstab = pd.crosstab(ratings['user'], ratings['movie'],
                           values=ratings['rating'], aggfunc='sum').fillna(0)

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

    recommendations_internal = top5_ratings.indices.tolist()[0]

    recommendations_tmdb = [internal_to_tmdb.get(
        rec, None) for rec in recommendations_internal]
    recommendations_tmdb = [
        rec for rec in recommendations_tmdb if rec is not None]

    return {"recommendations": recommendations_tmdb}

# start: uvicorn api:app --reload
