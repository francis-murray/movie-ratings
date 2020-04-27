import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import src.processing.process_credits as credits
import src.processing.join_keywords_ratings as kr
import src.processing.process_ratings as ratings
import src.processing.util_processing as up
import src.processing.process_movies_metadata as md
import pandas as pd
import tkinter as tk
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def linear_regression_predict_imdb_score_from_metadata():
    df=None
    try:
        df = pd.read_csv(up.data_processed_dir+"metadata_rating.csv")
    except:
        df = md.clean(md.raw())[['id','budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']]
        df = df.rename(columns={'id': 'tmdbId', 'vote_average':'vote_average_tmdb', 'vote_count':'vote_count_tmdb'})

        link = pd.read_csv(up.data_raw_dir+"links_small.csv")
        rt = ratings.columns_rating_movieId(ratings.raw())[['movieId', 'number_of_ratings', 'rating_mean']]
        df = pd.merge(df, link, how='inner', on=['tmdbId'])
        df = pd.merge(df, rt, how='inner', on=['movieId'])
        df['tmdbId'] = df['tmdbId'].astype('int')
        df['budget'] = df['budget'].astype('int')
        df['revenue'] = df['revenue'].astype('int')
        df['movieId'] = df['movieId'].astype('int')
        df['imdbId'] = df['imdbId'].astype('int')
        df['number_of_ratings'] = df['number_of_ratings'].astype('int')
        df = df.rename(columns={'rating_mean':'vote_average_imdb', 'number_of_ratings':'vote_count_imdb'})

        df['vote_count'] = df['vote_count_imdb'] + df['vote_count_tmdb']
        df['vote_average'] = (df['vote_average_imdb'].apply(lambda x:2*x) + df['vote_average_tmdb']).apply(lambda x:x/2)

        df = df.drop(['vote_average_tmdb','vote_average_imdb', 'vote_count_tmdb', 'vote_count_imdb'], axis=1)

        df.to_csv(up.data_processed_dir+"metadata_rating.csv")


    df = df[df['vote_average'].notnull() & df['vote_count'].notnull()]

    #According to imdb formula (source : wikipedia), we have a minimum votes required to be listed in the Top 250 (currently 25,000)
    df = df[df['vote_count']>=300]
    #Formula got from wikipedia page of imdb
    def imdb_formula(x):
        c=7.0
        m=25000
        return (x['vote_average']*x['vote_count']+c*m)/(x['vote_count']+m)

    df['weighted_vote_average'] = df.apply(imdb_formula, axis=1)

    #On remarque quand on voit la matrice de pearson que le weighted vote average n'est corrélé avec aucune de ces variables
    #Cependant on remarque d'autres correlations notamment entre revenue, popularité et budget
    df = df[['weighted_vote_average', 'revenue', 'popularity', 'budget', 'runtime']]
    #Draw the heatmap matrix of correlation between numerical features
    def pearson_correlation_heatmap(df):
        plt.figure(figsize=(12, 10))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()

    pearson_correlation_heatmap(df)
    y = df['weighted_vote_average']
    x = df[['revenue', 'popularity', 'budget', 'runtime']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train, y_train)
    pred = lin_reg.predict(X_test)
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    test_set_r2 = r2_score(y_test, pred)
    print("RMSE = ", test_set_rmse, " | Plus la valeur est petite, mieux c'est")
    print("R2 = ", test_set_r2, " | Plus la valeur est proche de 1, mieux c'est, si la valeur est négative, alors le "
                                "modèle ne correspond pas aux données")




if __name__=="__main__":
    linear_regression_predict_imdb_score_from_metadata()