from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search(model, x, y, grid_params):
    grid = GridSearchCV(estimator=model, param_grid=grid_params)
    grid.fit(x, y)

    print("grid : ",grid)

    print("best score : ", grid.best_score_)
    print("best params : ",grid.best_params_)
    return grid.best_estimator_

def random_search(model, x, y, grid_params, nb_iterations=30):
    grid = RandomizedSearchCV(estimator=model, param_distributions=grid_params, n_iter=nb_iterations)
    grid.fit(x, y)

    print("grid : ", grid)

    print("best score : ", grid.best_score_)
    print("best params : ", grid.best_params_)
    return grid.best_estimator_