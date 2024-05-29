import pandas as pd
from fastai.collab import *
from fastai.tabular.all import *

path = untar_data(URLs.ML_100k)
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',
                        usecols=(0, 1), names=('movie', 'title'), header=None)
ratings = pd.read_csv('updated_ratings.csv')

def retrain_model():
    dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
    
    # Create and train the model
    learn = collab_learner(dls, n_factors=50, y_range=(0.5, 5.5))
    learn.fit_one_cycle(5, 5e-3)
    
    # Save the updated model
    learn.export('movie-recommender.pkl')

if __name__ == "__main__":
    retrain_model()