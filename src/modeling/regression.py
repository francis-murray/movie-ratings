import numpy as np


class RegressionError(Exception):
    pass


class SimpleLinearRegression:
    def __init__(self, x, y):
        if len(x) != len(y):
            raise RegressionError('x and y of different len')
        self.obs = np.array([x, y])
        self.n = len(x)
        self.Ex = 0.
        self.Ey = 0.
        self.Sxy = 0.
        self.Sxx = 0.
        self.Syy = 0.
        self.Sr = 0.
        self.b0 = 0.
        self.b1 = 0.
        self.sse = 0.
        self.ssr = 0.
        self.sst = 0.
        self.sr = 0.
        self.compute_e_and_s(x, y)
        self.compute_coefficients()

    def compute_e_and_s(self, x, y):
        self.Ex = sum(x) / self.n
        self.Ey = sum(y) / self.n
        self.Sxy = sum([C[0] * C[1] for C in self.obs.T]) - (self.n * self.Ey * self.Ex)
        self.Sxx = sum([a * a for a in x]) - self.n * (self.Ex ** 2)
        self.Syy = sum([a * a for a in y]) - self.n * (self.Ey ** 2)
        self.Sr = (self.Sxx * self.Syy - self.Sxy * self.Sxy) / self.Sxx

    def compute_coefficients(self):
        self.b1 = self.Sxy / self.Sxx
        self.b0 = self.Ey - self.b1 * self.Ex

    def compute_error(self):
        self.sse = sum([(y - self.Ey) ** 2 for (_, y) in self.obs])
        self.ssr = (self.n - 1) * ((self.Sxy ** 2) / self.Sxx)
        self.sst = self.ssr + self.sse

    def compute_determination_coefficient(self):
        self.sr = self.ssr / self.sst

    def predict(self, x):
        return self.b0 + self.b1 * x

    def add_observations(self, x, y):
        if len(x) != len(y):
            raise RegressionError('x and y of different len')
        self.n += len(x)
        self.obs += zip(x, y)
        self.compute_e_and_s()
        self.compute_coefficients()
        self.compute_error()
        self.compute_determination_coefficient()


class MultipleLinearRegression:
    def __init__(self, X, Y):
        if len(X) != len(Y):
            raise RegressionError('x and y of different len')
        self.n = len(X)
        if self.n > 0:
            self.degree = len(X[0])
        else:
            self.degree = -1
        self.X = np.column_stack(([1] * len(X), np.array(X)))
        self.Y = np.array(Y)
        self.Ey = 0.
        self.sse = 0.
        self.ssr = 0.
        self.sst = 0.
        self.r = 0.
        self.coef = None
        self.compute_coef()

    def compute_coef(self):
        self.coef = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.Y))

    def compute_params(self):
        self.Ey = sum(self.Y) / self.n
        self.sse = 0.
        self.ssr = 0.
        self.sst = 0.
        for i in range(self.n):
            predicted_yi = sum(self.coef * self.X[i])
            self.sse += (predicted_yi - self.Ey) ** 2
            self.ssr += (self.Y[i] - predicted_yi) ** 2
            self.sst += (self.Y[i] - self.Ey) ** 2

    def add_observations(self, X, Y):
        self.X = np.row_stack((self.X, np.column_stack(([1] * len(X), X))))
        self.Y = np.append(self.Y, Y)
        self.compute_params()

    def predict(self, X):
        X = np.append([1], np.array(X))
        return sum(self.coef * X)
