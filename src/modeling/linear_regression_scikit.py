import random

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def test_linear_regression_multiple(x, y, lin_reg):
    def infer(x, y):
        print('Test Multiple Linear Regression:')
        print('given :', x)
        print('predicted : ', lin_reg.predict([x]))
        print('should predict : ', y)
        print('-----------------------')

    for _ in range(10):
        randI = random.randint(0, len(x.values) - 1)
        infer(x.values[randI], y.values[randI])


def linear_regression_multiple(x, y):
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
