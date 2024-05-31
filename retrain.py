# retrain.py
import pandas as pd
from fastai.collab import *
from fastai.tabular.all import *

def retrain_model():
    ratings = pd.read_csv('updated_ratings.csv')
    movies = pd.read_csv('path/to/movies.csv')
    
    ratings = ratings.merge(movies, on='movie')

    dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
    
    learn = collab_learner(dls, n_factors=50, y_range=(0.5, 5.5))
    learn.fit_one_cycle(5, 5e-3)
    
    learn.export('movie-recommender.pkl')

if __name__ == "__main__":
    retrain_model()
