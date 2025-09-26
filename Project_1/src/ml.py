import numpy as np


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
        self, eta=1e-3, n_iterations=1e5, lamb=0, mass=0, atol=1e-8, full_output=False
    ):
        self.set_niterations(n_iterations)
        self.eta = eta
        self.mass = mass
        self.atol = atol
        self.lamb = lamb
        self.full_output = full_output

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

    def Grad(self, X, y):
        """
        Gradient decent method that does both OLS and Ridge based on the choice of lamb and mass in the constructor.

        See the documentation for the whole class for a usage guide.
        """
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)

        record = []  # recording of parameters at each step of the gradient descent

        for i in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change
            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            change = (-1 * self.eta * grad) + momentum
            theta += change

            if self.full_output:
                record.append(np.copy(theta))

            if (self.atol is not None) and np.linalg.norm(grad) < self.atol:
                break

        if self.full_output:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def AdaGrad(self, X, y, delta=1e-7):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        r = np.zeros_like(theta)

        record = []

        for i in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change

            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            r += grad**2
            weights = self.eta / (delta + np.sqrt(r))
            change = (-1 * weights * grad) + momentum
            theta += change

            if self.full_output:
                record.append(np.copy(theta))

            if (self.atol is not None) and (np.linalg.norm(grad) < self.atol):
                break

        if self.full_output:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def RMSGrad(self, X, y, delta=1e-7, decay=0.99):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        r = np.zeros_like(theta)

        record = []

        for i in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change

            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            r = decay * r + (1 - decay) * grad**2

            weights = self.eta / np.sqrt(delta + r)
            change = (-1 * weights * grad) + momentum
            theta += change

            if self.full_output:
                record.append(np.copy(theta))

            if (self.atol is not None) and np.linalg.norm(grad) < self.atol:
                break

        if self.full_output:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def ADAM(self, X, y, delta=1e-8, decay_1=0.9, decay_2=0.9):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        s = np.zeros_like(theta)  # first moment estimates
        r = np.zeros_like(theta)  # second moment estimates

        record = []

        for i in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change
            t = i + 1

            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            s = decay_1 * s + (1 - decay_1) * grad
            r = decay_2 * r + (1 - decay_2) * grad**2

            s_hat = s / (1 - decay_1**t)
            r_hat = r / (1 - decay_2**t)

            change = -1 * self.eta * s_hat / (np.sqrt(r_hat) + delta)
            # change = (-1 * weights * grad) + momentum
            theta += change

            if self.full_output:
                record.append(np.copy(theta))

            if (self.atol is not None) and np.linalg.norm(grad) < self.atol:
                break

        if self.full_output:
            stats = {"n": i + 1, "record": np.array(record)}
            return theta, stats

        return theta

    def LassoRegression(self, X, y):
        n = y.shape[0]
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)

        for _ in range(self.n):
            grad = ((1 / n) * X.T @ (X @ theta - y)) + self.lmbda * np.sign(theta)
            z = theta + (-1 * self.eta * grad)
            theta = np.sign(z) * np.maximum((np.abs(z) - self.eta * self.lmbda), 0)
            if np.linalg.norm(grad) < self.atol:
                break

        return theta

    # def StochasticGD(X: np.ndarray,
    #                  y: np.ndarray,
    #                  batchsize: int)

    #     n = y.size[0]
    #     m = int(n/batchsize)

    #     for i in range(self.n)
