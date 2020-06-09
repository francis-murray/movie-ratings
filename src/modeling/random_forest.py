import matplotlib.pyplot as plt
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

def apply_random_forest(x,y):
    rf = RandomForestClassifier(max_depth=20, random_state=0)
    rf.fit(x,y)
    return rf


def vizualize_random_forest(x, y, clf, filename):
    # Standard code taken from web
    tree = clf.estimators_[5]
    export_graphviz(tree, out_file=filename + '.dot', feature_names=x.columns, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file(filename + '.dot')
    graph.write_png(filename + '.png')

    importances = list(clf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x.columns, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # Set the style
    plt.style.use('fivethirtyeight')  # list of x locations for plotting
    x_values = list(range(len(importances)))  # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')  # Tick labels for x axis
    plt.xticks(x_values, x.columns, rotation='vertical')  # Axis labels and title
    plt.ylabel('Importance');
    plt.xlabel('Variable');
    plt.title('Importance des variables dans l\'algorithme des random forests')
    plt.show()
