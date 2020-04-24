from sklearn.model_selection import train_test_split
import src.processing.join_dataset as j
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
if __name__=='__main__':

    df = j.joined()
    x, yref = df[['revenue','popularity','runtime','vote_count']], df['vote_average']

    for i in range(5):
        try:
            y=yref
            y = y==i+2
            X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75 ,random_state=30)
            svc = SVC(random_state=42)
            svc.fit(X_train, y_train)
            svc_disp = plot_roc_curve(svc, X_test, y_test)
            plt.show()


        except ValueError as e:
            print(e)
            print('attention ' + str(i))