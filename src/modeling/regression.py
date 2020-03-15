import numpy as np

class RegressionError(Exception):
    pass

class SimpleLinearRegression:
    def __init__(self,x,y):
        if len(x)!=len(y):
            raise RegressionError('x and y of different len')
        self.obs = np.array([x,y])
        self.n = len(x)
        self.Ex=0.
        self.Ey=0.
        self.Sxy=0.
        self.Sxx=0.
        self.Syy=0.
        self.Sr=0.
        self.b0=0.
        self.b1=0.
        self.sse=0.
        self.ssr=0.
        self.sst=0.
        self.sr =0.
        self.compute_e_and_s(x,y)
        self.compute_coefficients()

    def compute_e_and_s(self,x,y):
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
        self.sse = sum([(y-self.Ey)**2 for (_,y) in self.obs])
        self.ssr = (self.n-1) * ((self.Sxy**2) / self.Sxx)
        self.sst = self.ssr+self.sse

    def compute_determination_coefficient(self):
        self.sr = self.ssr / self.sst

    def predict(self,x):
        return self.b0 + self.b1 * x

    def add_observations(self,x,y):
        self.obs += zip(x,y)
        self.compute_e_and_s()
        self.compute_coefficients()
        self.compute_error()
        self.compute_determination_coefficient()

class MultipleLinearRegression:
    def __init__(self,X,Y):
        print([1]*len(X))
        self.X =  np.column_stack(([1]*len(X),np.array(X)))
        self.Y = np.array(Y)
        self.coef = None
        self.compute_coef()

    def compute_coef(self):
        self.coef = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.Y))
    def add_observations(self,X,Y):
        self.X = np.row_stack((self.X,np.column_stack(([1]*len(X),X))))
        self.Y = np.append(self.Y,Y)
if __name__=='__main__':
    x = [20,24,28,22,32,28,32,36,41,41]
    y = [16,18,23,24,28,29,26,31,32,34]
    rls = SimpleLinearRegression(x,y)
    print(rls.b0)
    print(rls.b1)
    print(rls.predict(2000))
    xm = [
        [2768,252,22,324,8760219,438465.0625],
        [4108,333,29,308,8760195,438374.0625]
    ]
    ym = [95,150]#,4,0,0,80,95,20,90,10,10,50,45,60,55,3,33]
    rsm = MultipleLinearRegression(xm,ym)
    rsm.add_observations(xm,ym)
