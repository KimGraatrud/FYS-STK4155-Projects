import numpy as np
from . import utils


class GD:
    """
    Gradient-Decent based methods for finding the optimal parameter of a linear regression

    Currently has regular GD, AdaGrad, RMSGrad, and ADAM algorythms implemented.

    Parameters
    -------------
    eta : float, default=1e-3
        Learning rate.
    n_iterations : int, default=1e5
        Maximum number of iterations.
    lamb_R : float, default=0
        L2 regularization coefficient (ridge).
    lamb_L : float, default=0
        L2 regularization coefficient (lasso).
    mass : float, default=0
        Momentum factor.
    atol : float, default=1e-8 lamb, mass, atol
        Absolute tolerance for stopping criterion.

    Usage
    -------------
    >>> y = some data to fit
    >>> X = some feature matrix
    >>> GradientDecent_OLS_no_momentum = GD(eta, n_iterations, 0, 0)
    >>> theta = GradientDecent_OLS_no_momentum.Grad(X,y)
    >>> y_hat = X @ theta

    Bsed on what you initialise the penalty and mass to be you can also do OLS with/without momentum or Ridge regresssion with/without momentum, like so

    >>> GradientDecent_Ridge_no_momentum = GD(eta, n_iterations, .001, .3)
    >>> theta = GradientDecent_Ridge_no_momentum.Grad(X,y)
    >>> y_hat = X @ theta
    """

    def __init__(
        self,
        X,
        y,
        eta=1e-3,
        n_iterations=1e5,
        lamb=0,
        mass=0,
        atol=1e-8,
        M=None,
        verbose=False,
        has_interecpt=True,
    ):
        self.X = X
        self.y = y
        self.set_niterations(n_iterations)
        self.eta = eta
        self.mass = mass
        self.atol = atol
        self.lamb = lamb
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=utils.RANDOM_SEED)
        self._M = int(np.round(M)) if M is not None else len(y)
        self.has_interecpt = has_interecpt

    def set_niterations(self, n):
        """
        Sets a new number of iterations that will be used.

        Used to update the number of iterations after having the instance created.

        Parameters
        -----------
        n  :  int
            Number of iterations that will be used in the next GD method.
        """
        self.n = int(np.round(n))

    def _make_minibatch(self, replace=False):
        """
        Splits X and y into m minibatches, returns the split versions and then we can take a random int and chose a batch
        """
        n, p = self.X.shape
        indices = self.rng.choice(n, size=self._M, replace=replace)

        return self.X[indices], self.y[indices]

    def _calc_penalty(self, theta):
        S = np.eye(len(theta))
        if self.has_interecpt:  # don't penalize the intercept
            S[0, 0] = 0

        return self.lamb * theta @ S

    def Grad(self):
        """
        Gradient decent method that does both OLS and Ridge based on the choice of lamb and mass in the constructor.

        See the documentation for the whole class for a usage guide.
        """
        n, p = self.X.shape
        theta = np.zeros(p)
        change = np.zeros_like(theta)

        record = []  # recording of parameters at each step of the gradient descent

        for i in range(self.n):
            # if i % 1e3 == 0:
            #     print(i)

            Xi, yi = self._make_minibatch()

            penalty = self._calc_penalty(theta)
            momentum = self.mass * change

            grad = 2 * ((1 / len(yi)) * Xi.T @ (Xi @ theta - yi) + penalty)
            change = (-1 * self.eta * grad) + momentum
            theta += change

            if self.verbose:
                record.append(np.copy(theta))

            if (self.atol is not None) and np.linalg.norm(grad) < self.atol:
                break

        if self.verbose:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def AdaGrad(self, delta=1e-7):
        n, p = self.X.shape
        theta = np.zeros(p)
        change = np.zeros_like(theta)
        r = np.zeros_like(theta)

        record = []

        for i in range(self.n):

            Xi, yi = self._make_minibatch()

            penalty = self._calc_penalty(theta)
            momentum = self.mass * change

            grad = ((2 / len(yi)) * Xi.T @ (Xi @ theta - yi)) + penalty
            r += grad**2
            weights = self.eta / (delta + np.sqrt(r))
            change = (-1 * weights * grad) + momentum
            theta += change

            if self.verbose:
                record.append(np.copy(theta))

            if (self.atol is not None) and (np.linalg.norm(grad) < self.atol):
                break

        if self.verbose:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def RMSGrad(self, delta=1e-7, decay=0.99):
        n, p = self.X.shape
        theta = np.zeros(p)
        change = np.zeros_like(theta)
        r = np.zeros_like(theta)

        record = []
        r_record = []
        sqrt = []
        grads = []

        for i in range(self.n):

            Xi, yi = self._make_minibatch()

            penalty = self._calc_penalty(theta)
            momentum = self.mass * change

            grad = 2 * (1 / len(yi)) * Xi.T @ (Xi @ theta - yi) + penalty
            r = decay * r + (1 - decay) * grad**2

            weights = self.eta / np.sqrt(delta + r)
            change = (-1 * weights * grad) + momentum
            theta += change

            if self.verbose:
                record.append(np.copy(theta))
                r_record.append(np.copy(r))
                sqrt.append(np.sqrt(delta + r))
                grads.append(np.copy(grad**2))

            if (self.atol is not None) and np.linalg.norm(grad) < self.atol:
                break

        if self.verbose:
            stats = {
                "n": i + 1,
                "record": np.array(record),
                "r": np.array(r_record),
                "sqrt": np.array(sqrt),
                "grads": np.array(grads),
            }
            return theta, stats

        return theta

    def ADAM(self, delta=1e-8, decay_1=0.9, decay_2=0.9):
        n, p = self.X.shape
        theta = np.zeros(p)
        change = np.zeros_like(theta)
        s = np.zeros_like(theta)  # first moment estimates
        r = np.zeros_like(theta)  # second moment estimates

        record = []

        for i in range(self.n):

            Xi, yi = self._make_minibatch()

            penalty = self._calc_penalty(theta)
            t = i + 1

            grad = ((2 / len(yi)) * Xi.T @ (Xi @ theta - yi)) + penalty
            s = decay_1 * s + (1 - decay_1) * grad
            r = decay_2 * r + (1 - decay_2) * grad**2

            s_hat = s / (1 - decay_1**t)
            r_hat = r / (1 - decay_2**t)

            change = -1 * self.eta * s_hat / (np.sqrt(r_hat) + delta)
            theta += change

            if self.verbose:
                record.append(np.copy(theta))

            if (self.atol is not None) and np.linalg.norm(grad) < self.atol:
                break

        if self.verbose:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def Lasso(self):
        # TODO: fix me
        n, p = self.X.shape
        theta = np.zeros(p)

        record = []

        for i in range(self.n):

            X, y = self._make_minibatch()

            S = np.eye(len(theta))
            if self.has_interecpt:  # don't penalize the intercept
                S[0, 0] = 0
            grad = ((1 / n) * X.T @ (X @ theta - y)) + self.lamb * np.sign(theta) @ S

            theta += -1 * self.eta * grad

            if self.verbose:
                record.append(np.copy(theta))

            if (self.atol is not None) and (np.linalg.norm(grad) < self.atol):
                break

        if self.verbose:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def _learning_schedule(self, t):
        """
        Helper for StochasticGD method.

        Updates the learning rate for SGD dynamically.
        """
        t0, t1 = 5, 10
        return t0 / (t + t1)
