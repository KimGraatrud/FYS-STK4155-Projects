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

def plot_feature_importances(model, filename):
    """
    Plot the feature importance of a trained model.

    To show the reason for the bad preformance.
    """

    importances = model.feature_importances_.reshape(height, width)
    plt.imshow(importances, cmap="viridis")
    plt.colorbar()
    plt.savefig(filename)
    plt.show()


def main():

    # Set batchsize, number of images to train on
    batchsz = 256

    # Load datasets
    from torchvision.io import decode_image
    trainset = FacesDataset.FacesDataset(utils.DATA_URL, train=True)
    testset = FacesDataset.FacesDataset(utils.DATA_URL, train=False)


    Xtrain, ytrain = trainset.as_numpy(flatten=True)
    Xtest,  ytest  = testset.as_numpy(flatten=True)
    raise



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
    



