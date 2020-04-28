import random

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

import src.processing.process_credits as credits
import src.processing.join_keywords_ratings as kr
import src.processing.process_ratings as ratings
import src.processing.util_processing as up
import src.processing.process_movies_metadata as md
import pandas as pd
import tkinter as tk
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from src.modeling import knn


def knn_predict_imdb_score_from_metadata():
    df = load_and_visualize_data()
    df = df[['weighted_score_category', 'revenue', 'popularity', 'budget', 'runtime']]
    # pearson_correlation_heatmap(df)
    y = df['weighted_score_category']
    x = df[['revenue', 'popularity', 'budget', 'runtime']]
    col = x.columns
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=col)

    knn_model = knn.apply_knn(x,y)
    knn.test_knn(x, y, knn_model)


def linear_regression_predict_imdb_score_from_metadata():
    df = load_and_visualize_data()
    df = df[['weighted_vote_average', 'revenue', 'popularity', 'budget', 'runtime']]
    #pearson_correlation_heatmap(df)
    y = df['weighted_vote_average']
    x = df[['revenue', 'popularity', 'budget', 'runtime']]
    lin_reg= linear_regression_multiple(x,y)
    test_linear_regression_multiple(x,y,lin_reg)

def random_forest_predict_imdb_score_from_metadata():
    df = load_and_visualize_data()
    df = df[['weighted_score_category', 'revenue', 'popularity', 'budget', 'runtime']]
    # pearson_correlation_heatmap(df)
    y = df['weighted_score_category']
    x = df[['revenue', 'popularity', 'budget', 'runtime']]
    col = x.columns
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=col)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train,y_train)

    pred = clf.predict(x_test)
    s=0
    for i in range(len(pred)):
        if pred[i]==y_test.values[i]:s+=1
    print(s/len(pred))
    #print('Mean Absolute Error:', round(np.mean(error), 2), 'degrees.')

    #mape = 100 * (error / y_test)

def load_and_visualize_data():
    df = None
    try:
        df = pd.read_csv(up.data_processed_dir + "metadata_rating.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    except:
        df = md.clean(md.raw())[['id', 'budget', 'popularity', 'revenue', 'runtime', 'vote_average',
                                 'vote_count', 'adult', 'original_language', 'video']]
        df = df.rename(columns={'id': 'tmdbId', 'vote_average': 'vote_average_tmdb', 'vote_count': 'vote_count_tmdb'})

        link = pd.read_csv(up.data_raw_dir + "links_small.csv")
        rt = ratings.columns_rating_movieId(ratings.raw())[['movieId', 'number_of_ratings', 'rating_mean']]
        df = pd.merge(df, link, how='inner', on=['tmdbId'])
        df = pd.merge(df, rt, how='inner', on=['movieId'])
        df['tmdbId'] = df['tmdbId'].astype('int')
        df['budget'] = df['budget'].astype('int')
        df['revenue'] = df['revenue'].astype('int')
        df['movieId'] = df['movieId'].astype('int')
        df['imdbId'] = df['imdbId'].astype('int')
        df['number_of_ratings'] = df['number_of_ratings'].astype('int')
        df = df.rename(columns={'rating_mean': 'vote_average_imdb', 'number_of_ratings': 'vote_count_imdb'})

        # df['vote_count'] = df['vote_count_imdb'] + df['vote_count_tmdb']
        # df['vote_average'] = (df['vote_average_imdb'].apply(lambda x:2*x) + df['vote_average_tmdb']).apply(lambda x:x/2)

        df['vote_count'] = df['vote_count_tmdb']
        df['vote_average'] = df['vote_average_tmdb']

        df = df.drop(['vote_average_tmdb', 'vote_average_imdb', 'vote_count_tmdb', 'vote_count_imdb'], axis=1)

        df.to_csv(up.data_processed_dir + "metadata_rating.csv")

    df = df[df['vote_average'].notnull() & df['vote_count'].notnull()]

    # According to imdb formula (source : wikipedia), we have a minimum votes required to be listed in the Top 250 (currently 25,000)
    # df = df[df['vote_count']>=300]

    df['weighted_vote_average'] = df.apply(imdb_formula, axis=1)
    df = df[(df['weighted_vote_average'].notnull())]
    print(df.head())
    print('Dataset infos :')
    print(df.info())
    print('Null values in the dataset :')
    print(df.isnull().sum())

    print('Le 95ème centile de chacune des colonnes :')
    d = {}
    for colN in df.columns:
        if np.issubdtype(df[colN].dtype, np.number):
            # col = df[colN].values
            d[colN] = df[colN].quantile(0.05)
    for key, value in d.items():
        print('\t' + key, value, sep=':')

    print(df.shape)
    # df = df[df['vote_count'] >= d['vote_count']]
    print(df.shape)

    def categorize_rating(x):
        val = x
        if val <= 4:
            return 'Bad'
        if 4 < val <= 6:
            return 'Ok'
        if 6 < val <= 8:
            return 'Good'
        if 8 < val:
            return 'Excellent'

    df['weighted_score_category'] = df['vote_average'].apply(categorize_rating)

    plt.title('Distribution des films pour adultes')
    lst = [df[df['adult'] == True]['adult'].count(), df[df['adult'] == False]['adult'].count()]
    plt.pie(lst, labels=('Oui', 'Non'), autopct='%1.1f%%')
    plt.show()
    show_hist_distribution(df['revenue'], x='Revenue', title='Distribution des revenus', bins=200)
    show_hist_distribution(df['vote_count'], x='Nombre de votes', title='Distirbution du nombre de votes', bins=200)
    show_hist_distribution(df['weighted_vote_average'], x='Score imdb', title='Distribution des scores imdb', bins=4)
    show_hist_distribution(df['weighted_score_category'], x='Score imdb', title='Distribution des scores imdb', bins=4)
    show_hist_distribution(df['original_language'], x='Langage original', title='Distribution du langage original',
                           bins=df['original_language'].nunique())

    return df

#On remarque quand on voit la matrice de pearson que le weighted vote average n'est corrélé avec aucune de ces variables
#Cependant on remarque d'autres correlations notamment entre revenue, popularité et budget
#Draw the heatmap matrix of correlation between numerical features
def pearson_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

def show_hist_distribution(col,x='',y='Frequence',title='',bins=5):
    plt.style.use('ggplot')
    plt.hist(col, bins=bins)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.show()

def imdb_formula(x):
    c=7.0
    m=25000
    return round((x['vote_average']*x['vote_count']+c*m)/(x['vote_count']+m),2)

def linear_regression_multiple(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train, y_train)
    pred = lin_reg.predict(X_test)
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    test_set_r2 = r2_score(y_test, pred)
    print("RMSE = ", test_set_rmse, " | Plus la valeur est petite, mieux c'est")
    print("R2 = ", test_set_r2, " | Plus la valeur est proche de 1, mieux c'est, si la valeur est négative, alors le "
                                "modèle ne correspond pas aux données")
    return lin_reg

def test_linear_regression_multiple(x,y,lin_reg):
    def infer(x, y):
        print('Test Multiple Linear Regression:')
        print('given :', x)
        print('predicted : ', lin_reg.predict([x]))
        print('should predict : ', y)
        print('-----------------------')

    for _ in range(10):
        randI = random.randint(0, len(x.values) - 1)
        infer(x.values[randI], y.values[randI])


if __name__=="__main__":
    #linear_regression_predict_imdb_score_from_metadata()
    #knn_predict_imdb_score_from_metadata()
    random_forest_predict_imdb_score_from_metadata()