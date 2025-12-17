from src import utils, Dataset
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


def regression_metrics(y_true, y_pred):
    """
    Returns RMSE and R^2.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def t_model_search_boost_random(
    data,
    reg_range,
    leaf_range,
    nodes_range,
    n_iter=20,
    seed=utils.SEED):
    """
    Randomized hyperparameter search for HistGradientBoostingRegressor.
    Selection criterion: RMSE (lower is better).
    """

    rng = utils.rng
    Xtrain, Xtest, ytrain, ytest = data

    def sample_params():
        return {
            "l2_regularization": rng.choice(reg_range),
            "max_leaf_nodes": rng.choice(leaf_range),
            "min_samples_leaf": rng.choice(nodes_range),
        }

    best_rmse = np.inf
    best_r2 = None
    best_params = None

    for i in range(n_iter):
        params = sample_params()

        model = HistGradientBoostingRegressor(
            max_iter=500,               # reduced for search
            early_stopping=True,
            random_state=seed,
            **params,
        )

        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xtest)

        rmse, r2 = regression_metrics(ytest, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_params = params

        print(
            f"[{i+1:02d}/{n_iter}] "
            f"RMSE={rmse:.4f}, R²={r2:.4f}, params={params}"
        )

    print("\nBest result:")
    print("RMSE:", best_rmse)
    print("R²:", best_r2)
    print("Best params:", best_params)

    return best_params, best_rmse, best_r2

def best_grad_boost():
    """
    The best hyperparameters found during a long hyperparameter search.

    Returns the hyperparameters of the best gradient boosted model we found.
    """
    params = {
        'l2': np.float64(0.046415888336127774),
        'max_leaf_nodes': np.int64(63),
        'min_leaf_samples': np.int64(5)
    }
    return params

def main():

    # Load datasets
    Xtrain, ytrain = Dataset.GalaxyDataset('train').flat()
    Xtest, ytest = Dataset.GalaxyDataset('test').flat()

    # Shuffle
    tr_idx = utils.shuffle_idx(ytrain)
    te_idx = utils.shuffle_idx(ytest)
    Xtrain, ytrain = Xtrain[tr_idx], ytrain[tr_idx]
    Xtest, ytest = Xtest[te_idx], ytest[te_idx]

    # Small regression tree
    start_time = time.time()

    model = DecisionTreeRegressor(
        max_depth=3,
        random_state=utils.SEED
    ).fit(Xtrain, ytrain)

    y_pred = model.predict(Xtest)
    rmse, r2 = regression_metrics(ytest, y_pred)

    print(f'small tree test RMSE: {rmse:.4f}, R²: {r2:.4f}')
    print(f'small tree took: {(time.time() - start_time)/60:.2f} min')

    # Dummy regressor
    start_time = time.time()

    model = DummyRegressor(strategy="mean").fit(Xtrain, ytrain)
    tr_pred = model.predict(Xtrain)
    te_pred = model.predict(Xtest)

    tr_rmse, tr_r2 = regression_metrics(ytrain, tr_pred)
    te_rmse, te_r2 = regression_metrics(ytest, te_pred)

    print(f'dummy train RMSE: {tr_rmse:.4f}, R²: {tr_r2:.4f}')
    print(f'dummy test  RMSE: {te_rmse:.4f}, R²: {te_r2:.4f}')
    print(f'dummy regressor took: {(time.time() - start_time)/60:.2f} min')

    # DONT RUN UNBOUNDED TREE UNLESS:
    #   - you have a system with a lot of RAM
    #   - you have two hours to watch paint dry...

    # # Unbounded regression tree
    # start_time = time.time()

    # model = DecisionTreeRegressor(
    #     random_state=utils.SEED
    # ).fit(Xtrain, ytrain)

    # tr_pred = model.predict(Xtrain)
    # te_pred = model.predict(Xtest)

    # tr_rmse, tr_r2 = regression_metrics(ytrain, tr_pred)
    # te_rmse, te_r2 = regression_metrics(ytest, te_pred)

    # print(f'deep tree train RMSE: {tr_rmse:.4f}, R²: {tr_r2:.4f}')
    # print(f'deep tree test  RMSE: {te_rmse:.4f}, R²: {te_r2:.4f}')
    # print(f'deep tree took: {(time.time() - start_time)/60:.2f} min')

    # Gradient Boosting Regressor
    start_time = time.time()

    model = HistGradientBoostingRegressor(
        random_state=utils.SEED
    ).fit(Xtrain, ytrain)

    tr_pred = model.predict(Xtrain)
    te_pred = model.predict(Xtest)

    tr_rmse, tr_r2 = regression_metrics(ytrain, tr_pred)
    te_rmse, te_r2 = regression_metrics(ytest, te_pred)

    print(f'grad boost train RMSE: {tr_rmse:.4f}, R²: {tr_r2:.4f}')
    print(f'grad boost test  RMSE: {te_rmse:.4f}, R²: {te_r2:.4f}')
    print(f'grad boost took: {(time.time() - start_time)/60:.2f} min')

    # DONT RUN THE CODE BELOW UNLESS:
    #   - you have all the time in the world, approx 3hrs on a very good CPU
    #   - you have a system with atleast 60GB of RAM

    # Use the validate set for search
    # free memory
    # del Xtest
    # del ytest

    # Xval, yval = Dataset.GalaxyDataset('validate').flat()
    # va_idx = utils.shuffle_idx(ytest)
    # Xval, yval = Xtest[va_idx], ytest[va_idx]

    # Hyperparameter search for boosting
    # start_time = time.time()

    # regs = np.logspace(-4, 4, 10)
    # leafs = np.array([15, 31, 63])
    # sampls = np.array([5, 20])
    # data = (Xtrain, Xval, ytrain, yval)

    # t_model_search_boost_random(data, regs, leafs, sampls)

    # print(f'grad boost model selection took: {(time.time() - start_time)/60:.2f} min')


if __name__ == '__main__':
    main()
