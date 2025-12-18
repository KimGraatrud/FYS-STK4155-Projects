from src import utils, Dataset
import joblib
from joblib import Parallel, delayed
import numpy as np
import time, gc
import matplotlib.pyplot as plt
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

def main():

    # Load datasets
    Xtrain, ytrain = Dataset.GalaxyDataset('train').flat()
    Xtest, ytest = Dataset.GalaxyDataset('test').flat()

    # Shuffle
    tr_idx = utils.shuffle_idx(ytrain)
    te_idx = utils.shuffle_idx(ytest)
    Xtrain, ytrain = Xtrain[tr_idx], ytrain[tr_idx]
    Xtest, ytest = Xtest[te_idx], ytest[te_idx]

    # Dummy regressor
    dummy = DummyRegressor(strategy="mean")
    # Small regression tree
    shallow_tree = DecisionTreeRegressor(
        max_depth=3,
        random_state=utils.SEED
    )
    # OOB Gradient Boost
    OOB_grad = HistGradientBoostingRegressor(
        random_state=utils.SEED
    )
    # Gradient Boost with tuned parameters
    tuned_grad = HistGradientBoostingRegressor(
        random_state=utils.SEED,
        l2_regularization=0.046415888336127774,
        max_leaf_nodes=63,
        min_samples_leaf=5
    )
    models = {
        # 'Dummy_Tree': dummy,
        # 'Shallow_Tree': shallow_tree,
        # 'OOB_Gradient_Boost': OOB_grad,
        # 'tuned_Gradient_Boost': tuned_grad
    }

    # DONT RUN UNBOUNDED TREE UNLESS:
    #   - you have a system with a lot of RAM
    #   - you have two hours to watch paint dry...

    models['Unbounded Tree'] = DecisionTreeRegressor(
        random_state=utils.SEED
    )

    model_names = list(models.keys())
    for n in model_names:
        model = models[n]
        print('Now running for:', n)

        # Train
        train_start = time.time()
        model.fit(Xtrain, ytrain)        
        train_end = time.time()

        # Predict
        tr_pred = model.predict(Xtrain)
        predict_start = time.time()
        te_pred = model.predict(Xtest)
        predict_end = time.time()

        # Calculate Preformance
        tr_rmse, tr_r2 = regression_metrics(ytrain, tr_pred)
        te_rmse, te_r2 = regression_metrics(ytest, te_pred)

        print(f' {n} train RMSE: {tr_rmse}, R²: {tr_r2}')
        print(f' {n} test RMSE: {te_rmse}, R²: {te_r2}')
        print(f' {n} training took: {(train_end - train_start)/60} min')
        print(f' {n} prediction (test only) took: {(predict_end - predict_start)/60} min')

        # Save model
        joblib.dump(model, utils.RESULTS_URL+f'{n}.joblib')
        print(f'model {n} saved.')

        # Attempt to save the memory we can
        del models[n]
        del model
        gc.collect()

    # Run Parameter Search for Gradient Boosting
    # DONT RUN IF:
    #   - you have > 60GB of RAM
    #   - can leave the computer running for all day

    # Use the validate set for search
    # free memory
    del Xtest
    del ytest

    Xval, yval = Dataset.GalaxyDataset('validate').flat()
    va_idx = utils.shuffle_idx(ytest)
    Xval, yval = Xtest[va_idx], ytest[va_idx]

    # Hyperparameter search for boosting
    start_time = time.time()

    regs = np.logspace(-4, 4, 10)
    leafs = np.array([15, 31, 63])
    sampls = np.array([5, 20])
    data = (Xtrain, Xval, ytrain, yval)

    t_model_search_boost_random(data, regs, leafs, sampls)

    print(f'grad boost model selection took: {(time.time() - start_time)/60:.2f} min')


if __name__ == '__main__':
    main()
