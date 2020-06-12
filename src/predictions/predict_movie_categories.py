from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import src.processing.process_movies_metadata as processing
import src.processing.util_processing as up
import src.processing.process_keywords as keywords_processing
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random
import numpy as np
import nltk
import ast
import timeit
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')

seuil = 0.3

# Clean a text manually
def clean_text_me(text):
    words = re.split(r'\W+', text)
    mapper = str.maketrans('', '', string.punctuation)
    words = [x.translate(mapper).lower() for x in words]
    return words


# Clean a text with NLTK :)
stop_words = [w.lower() for w in set(stopwords.words('english'))]
def clean_text_nltk(text):
    tokens = word_tokenize(text)
    porter = PorterStemmer()
    words = [word for word in tokens if word.isalpha()]
    words = [porter.stem(w).lower() for w in words if not w.lower() in stop_words]
    return words

def clean_text(text):
    return clean_text_nltk(text)

def collect_occurences(list_words,dic):
    for x in list_words:
        if x in dic.keys():
            dic[x] = dic[x]+1
        else:
            dic[x] = 1

def delIfNotKnown(lst,lst_source):
    i=0
    while i<len(lst):
        if lst[i] not in lst_source:
            del(lst[i])
        else:
            i+=1
    return lst


def join(x):
    return ' '.join([
        x['belongs_to_collection'],
        ' '.join(x['original_title']),
        ' '.join(x['title']),
        ' '.join(x['tagline']),
        ' '.join(x['overview']),
        ' '.join(x['keywords'])
    ])

def prepare_data_frame(visu = True, load_raw = False, filename = "predict_movie_categories_dataframe.csv", save=True):
    df=None
    labels = None
    mb = None
    print("start prepare data")
    start = timeit.default_timer()
    genres_with_occurences = {}
    words_with_occurences = {}
    if load_raw:
        df = processing.clean(processing.raw())
        df = df[['id','belongs_to_collection', 'genres', 'original_title', 'overview',
                 'tagline', 'title']]
        df['belongs_to_collection'] = df['belongs_to_collection'] \
            .apply(lambda x: x['name'] if 'name' in x.keys() else '')
        df['genres'] = df['genres'].apply(lambda x: [e['name'] for e in x])

        keywords = keywords_processing.get_films_with_keywords(keywords_processing.raw())

        df = pd.merge(df, keywords, how='left', on=['id'])

        genres_with_occurences = {}
        df['genres'].apply(lambda x: collect_occurences(x, genres_with_occurences))

        genres_with_occurences = {x: y for x, y in
                                  sorted(genres_with_occurences.items(),
                                         key=lambda e: e[1], reverse=True)[:10]
                                  }
        print(genres_with_occurences)

        df['genres'] = df['genres'].apply(lambda x: delIfNotKnown(x, genres_with_occurences.keys()))
        df = df[df['genres'].apply(lambda x: len(x) > 0)]
        # Prepare labels (as a 2D binary array)
        mb = MultiLabelBinarizer()
        labels = mb.fit_transform(df['genres'])

        df['keywords'] = df['keywords'].apply(clean_text)
        df['overview'] = df['overview'].apply(clean_text)
        df['tagline'] = df['tagline'].apply(clean_text)
        df['title_not_modified'] = df['title']
        df['title'] = df['title'].apply(clean_text)
        df['original_title'] = df['original_title'].apply(clean_text)
        df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: x.replace(' ', ''))
        words_with_occurences = {}
        df['overview'].apply(lambda x: collect_occurences(x, words_with_occurences))
        words_with_occurences = {x: y for x, y in
                                 sorted(words_with_occurences.items(),
                                        key=lambda e: e[1], reverse=True)
                                 }



        df['clean_x'] = df.apply(join, axis=1)
        df = df[['title_not_modified', 'clean_x', 'genres']]
        if save:
            df.to_csv(up.data_processed_dir+filename)
    else:
        df = pd.read_csv(up.data_processed_dir+filename)
        df['genres'] = df['genres'].apply(lambda x:ast.literal_eval(x))
        mb = MultiLabelBinarizer()
        labels = mb.fit_transform(df['genres'])

    end = timeit.default_timer()
    print("End prepare data, time : ", end-start)
    if visu:
        print(df.head())
        print(df.columns)
        print(df.values[1])
        if genres_with_occurences!={}:
            print('10 most present genres with their occurences :', genres_with_occurences)

            genres_df = pd.DataFrame({'Genre': list(genres_with_occurences.keys()),
                                      'Occurences': list(genres_with_occurences.values())}) \
                .set_index('Genre').rename_axis(None)

            genres_df = genres_df.sort_values('Occurences', ascending=True)

            genres_df.plot.barh()

            plt.show()
        if words_with_occurences != {}:
            print('Words with their occurences : ', words_with_occurences)

            words_df = pd.DataFrame({'Word': list(words_with_occurences.keys())[:100],
                                     'Occurences': list(words_with_occurences.values())[:100]}) \
                .set_index('Word').rename_axis(None)
            words_df = words_df.sort_values('Occurences', ascending=True)
            words_df.plot.barh(figsize=(15, 20))

            plt.show()

        print("Dataframe shape : ", df.shape)
        print(df.info())

        print("Labels")
        print(labels)
        print("Labels shape : ", labels.shape)
        print("For example, ", labels[0], "Stands for", mb.inverse_transform(labels)[0])
    return df, mb, labels

def get_nearest_films(overview="", belongs_to_collection="", original_title="", title="", tagline="", keywords="",
                      tfidf_vect=None, tfidf_matrix=None, df=None, nb_nearest = 10):
    x = join({
        'overview': clean_text(overview),
        'belongs_to_collection': belongs_to_collection.replace(' ', ''),
        'original_title': clean_text(original_title),
        'title': clean_text(title),
        'tagline': clean_text(tagline),
        'keywords': clean_text(keywords)

    })
    x_tfidf_vect = tfidf_matrix.transform([x])
    sim_positions = linear_kernel(x_tfidf_vect, tfidf_vect).flatten().argsort()[:-nb_nearest:-1]
    print("Nearest "+str(nb_nearest)+" films from "+title+" are : ")
    for pos in sim_positions:
        print(df.values[pos][1])
    print("__________________________________________________________________")
def predict(overview="", belongs_to_collection="", original_title="", title="", tagline="", keywords="",
            multilabel_binarizer=None, classifier=None,
            tfidf_vect=None, threshold_decision=np.vectorize(lambda t: 1 if t>seuil else 0)):
    x = join({
        'overview': clean_text(overview),
        'belongs_to_collection': belongs_to_collection.replace(' ', ''),
        'original_title': clean_text(original_title),
        'title': clean_text(title),
        'tagline': clean_text(tagline),
        'keywords': clean_text(keywords)
    })
    x_tfidf_vect = tfidf_vect.transform([x])
    return multilabel_binarizer.inverse_transform(threshold_decision(classifier.predict_proba(x_tfidf_vect)))


def prepare_data_frame_and_build_model(visu = True):
    df, mb, labels = prepare_data_frame(load_raw=False)

    tfidf_vect = TfidfVectorizer(max_features=50000)

    x_train, x_test, y_train, y_test = train_test_split(df[['clean_x', 'title_not_modified']], labels)
    # df['features'] = tfidf_vect.fit_transform(df['clean_x'])
    x_train_tf_idf = tfidf_vect.fit_transform(x_train['clean_x'])
    x_test_tf_idf = tfidf_vect.transform(x_test['clean_x'])

    clf = OneVsRestClassifier(LogisticRegression())
    #clf = OneVsRestClassifier(RandomForestClassifier()) #Score moins bon (0.60939) et temps énorme !
    #clf = OneVsRestClassifier(svm.SVC())
    #clf = OneVsRestClassifier(DecisionTreeClassifier()) #Mauvais score, et lent fit (0.471)
    clf.fit(x_train_tf_idf, y_train)

    print("Classifier parameters :")
    print(clf.get_params())

    threshold_decision = np.vectorize(lambda t: 1 if t>seuil else 0)
    y_pred = threshold_decision(clf.predict_proba(x_test_tf_idf))
    f1score = f1_score(y_test, y_pred, average='micro')
    eval1 = hamming_score(y_test, y_pred)
    eval2 = true_positive(y_test, y_pred)
    eval3 = false_positive(y_test, y_pred)
    eval4 = true_negative(y_test, y_pred)
    eval5 = false_negative(y_test, y_pred)

    for i in range(10):
        random_pos = random.randint(0, len(x_test))
        y_p = predict(overview=x_test['clean_x'].values[random_pos], multilabel_binarizer=mb, classifier=clf,
                      tfidf_vect=tfidf_vect)
        print("Title : ",x_test['title_not_modified'].values[random_pos], x_test['clean_x'].values[random_pos])
        print('Predicted : ', y_p[0])
        print('Actual :', mb.inverse_transform(y_test)[random_pos])
        print("__________________________________________________")
    for i in range(5):
        random_pos = random.randint(0, len(x_test))
        get_nearest_films(overview=x_test['clean_x'].values[random_pos], title=x_test['title_not_modified']
                          .values[random_pos] ,tfidf_vect=x_train_tf_idf,
                          tfidf_matrix=tfidf_vect ,df=x_train)
    if visu:
        print("Hamming SCORE ", eval1)
        print('F1 SCORE ',f1score)
        print("Taux de vrai positifs ", eval2)
        print("Taux de faux positifs ", eval3)
        print("Taux de vrai négatifs ", eval4)
        print("Taux de faux négatifs ", eval5)

def hamming_score(y_test, y_pred):
    score = 0
    for pair in zip(y_test, y_pred):
        cur_score = 0
        for i in range(len(pair[0])):
            if pair[0][i] == pair[1][i]:
                cur_score+=1
        score += cur_score / len(pair[0])
    return score / len(y_test)

def true_positive(y_test, y_pred):
    score = 0
    for pair in zip(y_test, y_pred):
        cur_score = 0
        cur_div = 0
        for i in range(len(pair[1])):
            if pair[0][i] == 1 :
                cur_div += 1
                if pair[1][i] == 1:
                    cur_score += 1
        score += cur_score / cur_div
    return score / len(y_test)
def false_positive(y_test, y_pred):
    score = 0
    for pair in zip(y_test, y_pred):
        cur_score = 0
        cur_div = 0
        for i in range(len(pair[1])):
            if pair[0][i] == 0 :
                cur_div += 1
                if pair[1][i] == 1:
                    cur_score += 1
        score += cur_score / cur_div
    return score / len(y_test)
def true_negative(y_test, y_pred):
    score = 0
    for pair in zip(y_test, y_pred):
        cur_score = 0
        cur_div = 0
        for i in range(len(pair[1])):
            if pair[0][i] == 0 :
                cur_div += 1
                if pair[1][i] == 0:
                    cur_score += 1
        score += cur_score / cur_div
    return score / len(y_test)
def false_negative(y_test, y_pred):
    score = 0
    for pair in zip(y_test, y_pred):
        cur_score = 0
        cur_div = 0
        for i in range(len(pair[1])):
            if pair[0][i] == 1 :
                cur_div += 1
                if pair[1][i] == 0:
                    cur_score += 1
        score += cur_score / cur_div
    return score / len(y_test)


if __name__=='__main__':
    prepare_data_frame_and_build_model()