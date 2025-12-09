from src import utils, FacesDataset, GradBoosting 
import numpy as np
from torch.utils.data import DataLoader
import torch
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Set torch seed to be sure.
g = torch.Generator()
g.manual_seed(utils.SEED)

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

def plot_feature_importances(model, filename, imgsize):
    """
    Plot the feature importance of a trained model.

    To show the reason for the bad preformance.
    """


    importances = model.feature_importances_.reshape(imgsize, imgsize)
    plt.imshow(importances, cmap="viridis")
    plt.colorbar()
    plt.savefig(filename)
    plt.show()

def dummy_train_fit(data):
    Xtrain, ytrain, Xtest, ytest = data
    clf = DummyClassifier(random_state=utils.SEED)
    clf.fit(Xtrain, ytrain)
    return clf.predict(Xtest)

def classifiertree_train_fit(data, Xpred, max_depth=None):
    Xtrain, ytrain, Xtest, ytest = data
    clf = DecisionTreeClassifier(max_depth=maxDepth, random_state=utils.SEED)
    clf.fit(Xtrain, ytrain)
    return clf.predict(Xpred)

def histboost_train_fit(data, Xpred, l2=0.0, learningrate=0.1,
                    max_leaf=8, max_depth=8, min_samples=20):
    Xtrain, ytrain, Xtest, ytest = data
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_leaf_nodes=max_leaf),
        n_estimators=n_weak_learners,
        random_state=utils.SEED
    )
    clf.fit(Xtrain, ytrain)
    return clf.predict(Xpred)



def main():

    # Load datasets
    Xtrain, ytrain = FacesDataset.FacesDataset(utils.DATA_URL, train=True).flat()
    Xtest, ytest = FacesDataset.FacesDataset(utils.DATA_URL, train=False).flat()

    tr_idx = utils.shuffle_idx(ytrain)
    te_idx = utils.shuffle_idx(ytest)
    Xtrain, ytrain = Xtrain[tr_idx], ytrain[tr_idx]
    Xtest, ytest = Xtest[te_idx], ytest[te_idx]

    print('train', np.unique(ytrain))
    print('test', np.unique(ytest))

    # Set batchsize, number of images to train on
    batchsz = len(ytrain)
    Xtrain, ytrain = Xtrain[:batchsz], ytrain[:batchsz]
    Xtest, ytest = Xtest[:batchsz], ytest[:batchsz]
    

    boost = GradBoosting.TreeClassifiers(Xtrain)

    # Dummy predicting the training data
    dummytrain = boost.dummyTree(ytrain, Xtrain)
    dummytest = boost.dummyTree(ytrain, Xtest)

    print('dummy error rate train',utils.error_rate(ytrain, dummytrain))
    print('dummy error rate test',utils.error_rate(ytest, dummytest))
    
    # Non-bounded tree predicting training data
    normalTreetrain = boost.classifierTree(ytrain, Xtrain)
    normalTreetest = boost.classifierTree(ytrain, Xtest)
    print('normal tree error rate',utils.error_rate(ytrain, normalTreetrain))
    print('normal tree error rate',utils.error_rate(ytest, normalTreetest))

    trboosted = boost.HistBoost(ytrain, Xtrain)
    teboosted = boost.HistBoost(ytrain, Xtest)
    print('grad boost train error rate',utils.error_rate(ytrain, trboosted))
    print('grad boost test error rate',utils.error_rate(ytest, teboosted))

    # # Boosted tree
    # regs = np.logspace(-4,4, 10)
    # leafs = np.array([15, 31, 63])
    # sampls = np.array([5, 20])
    # data = (Xtrain, Xtest, ytrain, ytest)
    # t_model_search_boost(data, regs, leafs, sampls)

def cuda_test():
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)

if __name__=='__main__':
    main()
    



