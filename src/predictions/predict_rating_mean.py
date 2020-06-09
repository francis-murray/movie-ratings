from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

import src.processing.process_credits as credits
import src.processing.join_keywords_ratings as kr
import src.processing.process_ratings as ratings
import src.processing.util_processing as up
import src.processing.process_movies_metadata as md
import pandas as pd
import tkinter as tk
import random
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import GridSearchCV

from src.modeling import knn

"""
    Predict rating mean from 3 keywords of movies. We use Knn algorithm and we variate k from 1 to 25
"""
def knn_predict_rating_mean_from_3_keywords():
    df = prepare_rating_and_3_keywords()
    x = df[['keywordId0', 'keywordId1', 'keywordId2']]
    y = df['rating_mean']
    knn.apply_knn(x,y)

"""
    Predict rating mean from 3 keywords of movies and 7 actors.
    We use Knn algorithm and we variate k from 1 to 25
"""
def knn_predict_rating_mean_from_3_keywords_and_7_actors():
    df = prepare_rating_and_3_keywords_id_and_7_actors_id(process=False)
    x = df[['keywordId0', 'keywordId1', 'keywordId2', 'cast0', 'cast1', 'cast2',
            'cast3', 'cast4', 'cast5', 'cast6']]
    y = df['rating_mean'].astype(int)

    # Rescaling features age, trestbps, chol, thalach, oldpeak.
    col = x.columns
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=col)
    print('Dataset infos :')
    print(df.info())
    print('Null values in the dataset :')
    print(df.isnull().sum())


    knn_model=knn.apply_knn(x, y)
    def infer(x, y):
        print('Test KNN:')
        print('given :',x)
        print('predicted : ',knn_model.predict([x]))
        print('should predict : ',y)
        print('----------------------------------------------')
    for _ in range(25):
        randI = random.randint(0,len(x.values)-1)
        infer(x.values[randI], y[randI])


"""
    Predict rating mean from 3 keywords of movies and 7 actors.
    We use Linear Regression

    We see that the prediction is not good because it's not a regression problem. There is visually no linear relation
    between keywordId and rating_mean, neither betewen actorsId and rating mean. Its more a classification problem.
"""
def linear_regression_predict_rating_mean_from_3_keywords_and_7_actors():
    df = prepare_rating_and_3_keywords_id_and_7_actors_id()
    x = df[['keywordId0','cast0']]
    y = df['rating_mean']
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    # tkinter GUI
    root = tk.Tk()

    canvas1 = tk.Canvas(root, width=500, height=300)
    canvas1.pack()

    # with sklearn
    Intercept_result = ('Intercept: ', regr.intercept_)
    label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
    canvas1.create_window(260, 220, window=label_Intercept)

    # with sklearn
    Coefficients_result = ('Coefficients: ', regr.coef_)
    label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
    canvas1.create_window(260, 240, window=label_Coefficients)


    # plot 1st scatter
    figure3 = plt.Figure(figsize=(5, 4), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df['keywordId0'].astype(float), df['rating_mean'].astype(float), color='r')
    scatter3 = FigureCanvasTkAgg(figure3, root)
    scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax3.legend(['rating_mean'])
    ax3.set_xlabel('keywordId0')
    ax3.set_title('keywordId0 Vs. rating_mean')

    # plot 2nd scatter
    figure4 = plt.Figure(figsize=(5, 4), dpi=100)
    ax4 = figure4.add_subplot(111)
    ax4.scatter(df['cast0'].astype(float), df['rating_mean'].astype(float), color='g')
    scatter4 = FigureCanvasTkAgg(figure4, root)
    scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax4.legend(['rating_mean'])
    ax4.set_xlabel('cast0')
    ax4.set_title('cast0 Vs. rating_mean')

    root.mainloop()




def prepare_rating_and_3_keywords():
    df = kr.processed()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

def prepare_rating_and_3_keywords_id_and_7_actors_id(process = False,
    filename = "keywords_3_first_cast_7_first_and_Films_ratings.csv"):
    if process:
        df = kr.read_raw_and_join_and_save(kr.filename)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df = df.rename(columns={'movieId': 'id'})
        cast = credits.clean(credits.raw())[['cast', 'id']]
        cast['id'] = cast['id'].astype('int')

        def f(x, i):
            if len(x) >= i + 1:
                return x[i]['id']
            else:
                if len(x) == 0:
                    return 0
                else:
                    return x[-1]['id']

        cast['cast0'] = cast['cast'].apply(lambda x: f(x, 0)).astype('int')
        cast['cast1'] = cast['cast'].apply(lambda x: f(x, 1)).astype('int')
        cast['cast2'] = cast['cast'].apply(lambda x: f(x, 2)).astype('int')
        cast['cast3'] = cast['cast'].apply(lambda x: f(x, 3)).astype('int')
        cast['cast4'] = cast['cast'].apply(lambda x: f(x, 4)).astype('int')
        cast['cast5'] = cast['cast'].apply(lambda x: f(x, 5)).astype('int')
        cast['cast6'] = cast['cast'].apply(lambda x: f(x, 6)).astype('int')
        cast = cast.drop(columns=['cast'])
        df = pd.merge(df, cast, how='inner', on=['id'])
        df = df[
            df['id'].notnull() & df['number_of_ratings'].notnull() &
            df['rating_mean'].notnull() & df['rating_median'].notnull()
            & df['keywordId0'].notnull() & df['keywordId1'].notnull() &
            df['keywordId2'].notnull() & df['cast0'].notnull() &
            df['cast1'].notnull() & df['cast2'].notnull() &
            df['cast3'].notnull() & df['cast4'].notnull() &
            df['cast5'].notnull() & df['cast6'].notnull() &
            ((df['keywordId2'] != 0) | (df['keywordId1'] != 0) | (df['keywordId2'] != 0)) &
            ((df['cast0'] != 0) | (df['cast1'] != 0) | (df['cast2'] != 0) |
             (df['cast3'] != 0) | (df['cast4'] != 0) | (df['cast5'] != 0) | (df['cast6'] != 0)
             )
            ]
        df.to_csv(up.data_processed_dir + filename)
    else:
        df = up.processed(filename)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df



"""Show how ratings are distributed (in ratings.csv)"""
def show_ratings_distribution():
    df = pd.read_csv(up.data_raw_dir+"ratings.csv")['rating']

    plt.style.use('ggplot')
    plt.hist(df, bins=5)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of ratings')
    plt.show()




if __name__=="__main__":
    #linear_regression_predict_imdb_score_from_metadata()
    show_ratings_distribution()
    knn_predict_rating_mean_from_3_keywords_and_7_actors()