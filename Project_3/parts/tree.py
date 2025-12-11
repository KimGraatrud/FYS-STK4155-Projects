from src import utils, FacesDataset, GradBoosting 
import numpy as np
from torch.utils.data import DataLoader
import torch
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


# Set torch seed to be sure.
torch.manual_seed(utils.SEED)

def t_model_search_boost(data, reg_range, leaf_range, nodes_range):
    """
    Variational model selection approach for gradient boosting,
    here varying the L2 regularization, max number of leafs and min number of samples per leaf.

    Multithread the search using joblib.
    """

    Xtrain, Xtest, ytrain, ytest = data

    boost = GradBoosting.TreeClassifiers(Xtrain)

    def train_and_eval(reg, lr, nr):
        boostedtest = boost.HistBoost(ytrain, Xtest, l2=reg, max_leaf=lr, min_samples=nr)
        err = utils.error_rate(ytest, boostedtest)
        return err

    results = Parallel(n_jobs=-1)(
        delayed(train_and_eval)(reg, lr, nr)
        for reg in reg_range 
        for lr in leaf_range
        for nr in nodes_range
    )

    # Reshape the 1D results array back into the 2D err matrix
    err = np.array(results).reshape(len(reg_range), len(leaf_range), len(nodes_range))

    best_tree = np.argmin(err)
    print('idx:',best_tree)
    print('error rate:',err.flatten()[best_tree])
    i,j, k = np.unravel_index(best_tree, err.shape)
    print(f'With l2={reg_range[i]}, number of leafs={leaf_range[j]}, and min samples={nodes_range[k]}')

def plot_feature_importances(model, filename, L, H):
    """
    Plot the feature importance of a trained model.

    To show the reason for the bad preformance.
    """


    importances = model.feature_importances_.reshape(L, H)
    plt.imshow(importances, cmap="viridis")
    plt.colorbar()
    plt.savefig(filename)
    plt.show()

def main():

    # Load datasets
    Xtrain, ytrain = FacesDataset.FacesDataset(utils.DATA_URL, train=True).flat()
    Xtest, ytest = FacesDataset.FacesDataset(utils.DATA_URL, train=False).flat()

    print(Xtrain.shape)
    print(Xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)

    ytrain_shuffled = ytrain.copy()
    np.random.shuffle(ytrain_shuffled)

    tr_idx = utils.shuffle_idx(ytrain)
    te_idx = utils.shuffle_idx(ytest)
    Xtrain, ytrain = Xtrain[tr_idx], ytrain[tr_idx]
    Xtest, ytest = Xtest[te_idx], ytest[te_idx]

    print(np.unique(ytrain, return_counts=True))
    print(np.unique(ytest, return_counts=True))

    model = DecisionTreeClassifier(max_depth=3, random_state=utils.SEED).fit(Xtrain, ytrain_shuffled)
    print(accuracy_score(ytest, model.predict(Xtest)))

    # Dummy predicting the training data
    # model = DummyClassifier(random_state=utils.SEED).fit(Xtrain, ytrain)
    # dummytrain = model.predict(Xtrain)
    # # dummytest = model.predict(Xtest)

    # print('dummy error rate train',utils.error_rate(ytrain, dummytrain))
    # # print('dummy error rate test',utils.error_rate(ytest, dummytest))
    
    # # Non-bounded tree predicting training data
    # model = DecisionTreeClassifier(random_state=utils.SEED).fit(Xtrain, ytrain)
    # normalTreetrain = model.predict(Xtrain)
    # normalTreetest = model.predict(Xtest)

    # print('tr_tree accuracy',accuracy_score(ytrain, normalTreetrain))
    # print('te_tree accuracy',accuracy_score(ytest, normalTreetest))

    # utils.print_tree_data(model)

    # confusion = confusion_matrix(ytest, normalTreetest, )
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=confusion,
    #     display_labels=FacesDataset.LABELS
    # ).plot(cmap='viridis')
    # plt.savefig('figures/deep-tree-CM.pdf')


    # Dont actually plot this
    # plt.figure(figsize=(18, 10))
    # plot_tree(
    #     model,
    #     filled=True,
    # )
    # plt.savefig('figures/bigtree.pdf')
    # plt.show()

    # # Gradient boost
    # model = HistGradientBoostingClassifier(
    #         loss='log_loss',
    #         # learning_rate=learningrate,
    #         # l2_regularization=l2,
    #         # max_leaf_nodes=max_leaf,
    #         # min_samples_leaf=min_samples,
    #         # early_stopping=False, # try and save on some compute 
    #         random_state=utils.SEED
    # ).fit(Xtrain, ytrain)
    # trboosted = model.predict(Xtrain)
    # teboosted = model.predict(Xtest)
    # print('grad boost train error rate',utils.error_rate(ytrain, trboosted))
    # print('grad boost test error rate',utils.error_rate(ytest, teboosted))

    # # Boosted tree
    # regs = np.logspace(-4,4, 10)
    # leafs = np.array([15, 31, 63])
    # sampls = np.array([5, 20])
    # data = (Xtrain, Xtest, ytrain, ytest)
    # t_model_search_boost(data, regs, leafs, sampls)

if __name__=='__main__':
    main()
    



