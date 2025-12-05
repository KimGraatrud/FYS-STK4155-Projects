from src import utils, FacesDataset, GradBoosting 
import numpy as np
from torch.utils.data import DataLoader
import torch
from joblib import Parallel, delayed

# Set torch seed to be sure.
g = torch.Generator()
g.manual_seed(utils.SEED)

def t_model_search_boost(data, reg_range, depth_range):
    """
    Variational model selection approach for gradient boosting,
    here varying the L2 regularization and tree depth.

    Multithread the search using joblib.
    """

    Xtrain, Xtest, ytrain, ytest = data

    boost = GradBoosting.TreeClassifiers(Xtrain)

    def train_and_eval(reg, d):
        boostedtest = boost.HistBoost(ytrain, Xtest, l2=reg, max_depth=d)
        err = utils.error_rate(ytest, boostedtest)
        return err

    results = Parallel(n_jobs=-1)(
        delayed(train_and_eval)(reg, d)
        for reg in reg_range 
        for d in depth_range
    )

    # Reshape the 1D results array back into the 2D err matrix
    err = np.array(results).reshape(len(reg_range), len(depth_range))

    best_tree = np.argmin(err)
    print('idx:',best_tree)
    print('error rate:',err.flatten()[best_tree])
    i,j = np.unravel_index(best_tree, err.shape)
    print(f'With l2={reg_range[i]}, and depth={depth_range[j]}')

def main():

    # Set batchsize, number of images to train on
    batchsz = 256

    # Load datasets
    trainset = FacesDataset.FacesDataset(utils.DATA_URL, train=True)
    # print("trainset", len(trainset))
    trainloader = DataLoader(trainset, batch_size=batchsz, shuffle=True)

    testset = FacesDataset.FacesDataset(utils.DATA_URL, train=False)
    # print("testset", len(testset))
    testloader = DataLoader(testset, batch_size=batchsz, shuffle=True)

    # Convert to numpy for scikitlearn
    imgstrain, labelstrain = next(iter(trainloader))
    Xtrain = imgstrain.numpy().reshape(imgstrain.shape[0], -1)
    ytrain = labelstrain.numpy()
    imgstest, labelstest = next(iter(testloader))
    Xtest = imgstest.numpy().reshape(imgstest.shape[0], -1)
    ytest = labelstest.numpy()

    
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

    # Boosted tree
    regs = np.logspace(-4,4, 10)
    depths = np.arange(3,8+1)
    data = (Xtrain, Xtest, ytrain, ytest)
    t_model_search_boost(data, regs, depths)



if __name__=='__main__':
    main()



