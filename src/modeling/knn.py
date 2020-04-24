from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import src.processing.process_credits as credits
import src.processing.join_keywords_ratings as kr
import src.processing.util_processing as up
import pandas as pd

"""
    Predict rating mean from 3 keywords of movies. We use Knn algorithm and we variate k from 1 to 25
"""
def knn_predict_rating_mean_from_3_keywords():
    df = kr.processed()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    x = df[['keywordId0', 'keywordId1', 'keywordId2']]
    y = df['rating_mean']
    apply_knn(x,y)

"""
    Predict rating mean from 3 keywords of movies and 7 actors.
    We use Knn algorithm and we variate k from 1 to 25
"""
def knn_predict_rating_mean_from_3_keywords_and_7_actors(process = False,
    filename = "keywords_3_first_cast_7_first_and_Films_ratings.csv"):

    if process:
        df = kr.processed()
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
    print(df)
    print(df.columns)
    print(df.shape)
    print(df.values[0])
    print(df.shape)
    x = df[['keywordId0', 'keywordId1', 'keywordId2', 'cast0', 'cast1', 'cast2',
            'cast3', 'cast4', 'cast5', 'cast6']]
    y = df['rating_mean']
    apply_knn(x, y)



def apply_knn(x,y):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    scores = {}
    scores_list = []
    for k in range(1, 26):
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(Y_test, Y_pred)
        scores_list.append(scores[k])
    plt.plot(range(1, 26), scores_list)
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.show()


if __name__=="__main__":
    knn_predict_rating_mean_from_3_keywords()
