from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import src.processing.process_movies_metadata as processing
import src.processing.util_processing as up
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
nltk.download('punkt')
nltk.download('stopwords')


# Clean a text manually
def clean_text(text):
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
        ' '.join(x['overview'])
    ])

def prepare_data_frame(visu = True, load_raw = False, filename = "predict_movie_categories_dataframe.csv", save=True):
    df=None
    labels = None
    mb = None
    genres_with_occurences = {}
    words_with_occurences = {}
    if load_raw:
        df = processing.clean(processing.raw())
        df = df[['belongs_to_collection', 'genres', 'original_title', 'overview',
                 'tagline', 'title']]
        df['belongs_to_collection'] = df['belongs_to_collection'] \
            .apply(lambda x: x['name'] if 'name' in x.keys() else '')
        df['genres'] = df['genres'].apply(lambda x: [e['name'] for e in x])

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

        df['overview'] = df['overview'].apply(clean_text_nltk)
        df['tagline'] = df['tagline'].apply(clean_text_nltk)
        df['title_not_modified'] = df['title']
        df['title'] = df['title'].apply(clean_text_nltk)
        df['original_title'] = df['original_title'].apply(clean_text_nltk)
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

def prepare_data_frame_and_build_model(visu = True):
    df, mb, labels = prepare_data_frame(load_raw=True)

    tfidf_vect = TfidfVectorizer(max_features=30000)

    x_train, x_test, y_train, y_test = train_test_split(df[['clean_x','title_not_modified']], labels)
    #df['features'] = tfidf_vect.fit_transform(df['clean_x'])
    x_train_tf_idf = tfidf_vect.fit_transform(x_train['clean_x'])
    x_test_tf_idf = tfidf_vect.transform(x_test['clean_x'])

    clf = OneVsRestClassifier(LogisticRegression())
    #clf = OneVsRestClassifier(RandomForestClassifier()) #Score moins bon (0.60939) et temps énorme !
    #clf = OneVsRestClassifier(svm.SVC())
    #clf = OneVsRestClassifier(DecisionTreeClassifier()) #Mauvais score, et lent fit (0.471)
    clf.fit(x_train_tf_idf, y_train)

    seuil = 0.25
    threshold_decision = np.vectorize(lambda t: 1 if t>seuil else 0)
    y_pred = threshold_decision(clf.predict_proba(x_test_tf_idf))
    f1score = f1_score(y_test, y_pred, average='micro')

    def predict(overview="",belongs_to_collection="",original_title="",title="",tagline=""):
        x = join({
            'overview':clean_text_nltk(overview),
            'belongs_to_collection':belongs_to_collection.replace(' ',''),
            'original_title':clean_text_nltk(original_title),
            'title':clean_text_nltk(title),
            'tagline':clean_text_nltk(tagline)
        })
        x_tfidf_vect = tfidf_vect.transform([x])
        return mb.inverse_transform(threshold_decision(clf.predict_proba(x_tfidf_vect)))

    for i in range(100):
        random_pos =  random.randint(0, len(x_test))
        y_p = predict(x_test['clean_x'].values[random_pos])
        print("Title : ",x_test['title_not_modified'].values[random_pos], x_test['clean_x'].values[random_pos])
        print('Predicted : ', y_p[0])
        print('Actual :', mb.inverse_transform(y_test)[random_pos])
        print("__________________________________________________")
    if visu:
        print('SCORE ',f1score)
if __name__=='__main__':
    prepare_data_frame_and_build_model()