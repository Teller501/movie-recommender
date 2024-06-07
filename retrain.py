# retrain.py
import pandas as pd
from fastai.collab import *
from fastai.tabular.all import *

def retrain_model():
    movies_path = 'movies.csv'
    ratings_path = 'updated_ratings.csv'

    ratings = pd.read_csv(ratings_path, delimiter=',', skiprows=1, names=['user','movie','rating','timestamp','title'], dtype={'timestamp': float})
    movies = pd.read_csv(movies_path, usecols=(0,1), names=('movie','title'), dtype={'movie': str})

    movies['movie'] = movies['movie'].astype(str)
    ratings['movie'] = ratings['movie'].astype(str)
    
    ratings = ratings.merge(movies, on='movie')

    print(ratings.shape)

    dls = CollabDataLoaders.from_df(ratings, item_name='title_x', bs=64)
    
    learn = collab_learner(dls, n_factors=50, y_range=(0.5, 5.5))
    learn.fit_one_cycle(5, 5e-3)
    
    learn.export('movie-recommender.pkl')

if __name__ == "__main__":
    retrain_model()
