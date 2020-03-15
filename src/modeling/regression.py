import numpy as np

class RegressionError(Exception):
    pass

class SimpleLinearRegression:
    def __init__(self,x,y):
        if len(x)!=len(y):
            raise RegressionError('x and y of different len')
        self.obs = zip(x,y)
        self.Ex=0.
        self.Ey=0.
        self.Sxy=0.
        self.Sxx=0.
        self.Syy=0.
        self.Sr=0.
        self.b0=0
        self.b1=0.
        self.compute_e_and_s(x,y)
        self.compute_coefficients()

    def compute_e_and_s(self,x,y):
        self.Ex = sum(x) / len(x)
        self.Ey = sum(y) / len(y)
        self.Sxy = sum([a * b for (a, b) in self.obs]) - len(x)
        self.Sxx = sum([a * a for a in x]) - len(x) * (self.Ex ** 2)
        self.Syy = sum([a * a for a in y]) - len(y) * (self.Ey ** 2)
        self.Sr = (self.Sxx * self.Syy - self.Sxy * self.Sxy) / self.Sxx

    def compute_coefficients(self):
        self.b1 = self.Sxy / self.Sxx
        self.b0 = self.Ey - self.b1 * self.Ex

    def predict(self,x):
        return self.b0 + self.b1 * x

    def add_observations(self,x,y):
        self.obs += zip(x,y)