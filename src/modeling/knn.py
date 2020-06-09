import random

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""
    Apply basic knn with k variating from 1 to 100 then plot the score foreach value of k
"""


def apply_knn(x, y):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
    kmax = 100
    scores = {}
    scores_list = []
    for k in range(1, kmax + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(Y_test, Y_pred)
        scores_list.append(scores[k])
    plt.plot(range(1, kmax + 1), scores_list)
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.title('Variation du score selon la valeur de K')
    plt.show()

    """
    leaf_size = list(range(1, 50))
    n_neighbors = list(range(58, 62))
    p = [1, 2]  # Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)  # Create new KNN object
    knn_2 = KNeighborsClassifier()  # Use GridSearch
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)  # Fit the model
    best_model = clf.fit(X_train, Y_train)  # Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    """
    bestK = scores_list.index(max(scores_list)) + 1

    print('Le meilleur K est : ', bestK, "avec le score de ", max(scores_list))
    Y_pred = knn.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    knn = KNeighborsClassifier(n_neighbors=bestK)
    # weights='uniform', algorithm ='auto', leaf_size = 1, metric ='minkowski', p = 2, metric_params = None)
    knn.fit(X_train, Y_train)

    # On retourne le meilleur modèle
    return knn


def test_knn(x, y, knn_model):
    def infer(x, y):
        print('Test KNN:')
        print('given :', x)
        print('predicted : ', knn_model.predict([x]))
        print('should predict : ', y)
        print('----------------------------------------------')

    for _ in range(25):
        randI = random.randint(0, len(x.values) - 1)
        infer(x.values[randI], y[randI])
