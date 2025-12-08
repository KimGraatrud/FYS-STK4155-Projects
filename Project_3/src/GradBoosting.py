from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import numpy as np 
from . import utils


class TreeClassifiers:
    """
    Abstraction of the scikitlearn methods we used.

    Contains all the methods we require to create the data in the report.
    """


    def __init__(self, Xtrain):
        
        self.X = Xtrain
        self.criterion = 'log_loss' 

    def dummyTree(self, ytrain, Xpred):
        """
        Return a dummy classifier that randomly guesses, for comparison.

        Here it will train on ytrain and predict Xpred, Xpred can either be 
        the training data or test data.
        """
        clf = DummyClassifier(random_state=utils.SEED)
        clf.fit(self.X, ytrain)
        return clf.predict(Xpred)


    def classifierTree(self, ytrain, Xpred, maxDepth=None):
        """
        Uses a single deep grown tree to learn the data.

        Used for comparison between boosting.

        ytrain, Xpred used for training and predicting.
        """

        clf = DecisionTreeClassifier(max_depth=maxDepth, random_state=utils.SEED)
        clf.fit(self.X, ytrain)
        return clf.predict(Xpred)

    def SAMME(self, ytrain, Xpred, n_weak_learners=150, max_leaf=8,
                Xoverride=None):
        """
        Uses the scikitlearn implementation of the SAMME AdaBoost algorithm.

        Returns the prediction of the Xpred feature matrix.
        """

        # If we wish to change X, for ex. in the case where we have feature selected for it.
        if Xoverride is not None: self._X = Xoverride

        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_leaf_nodes=max_leaf),
            n_estimators=n_weak_learners,
            random_state=utils.SEED
        )
        clf.fit(self.X, ytrain)
        return clf.predict(Xpred)

    def HistBoost(self, ytrain, Xpred, l2=0.0, learningrate=0.1,
                    max_leaf=8, max_depth=8, Xoverride=None):

        # If we wish to change X, for ex. in the case where we have feature selected for it.
        if Xoverride is not None: self._X = Xoverride

        clf = HistGradientBoostingClassifier(
            scoring='log_loss',
            learning_rate=learningrate,
            l2_regularization=l2,
            max_leaf_nodes=max_leaf,
            max_depth=max_depth,
            random_state=utils.SEED
        )
        clf.fit(self.X, ytrain)
        return clf.predict(Xpred)
