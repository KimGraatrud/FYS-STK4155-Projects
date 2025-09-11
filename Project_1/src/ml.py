import numpy as np


class GD:
    def __init__(self, eta=1e-3, n_iterations=1e5, lamb=0, mass=0, atol=1e-8):
        self.set_niterations(n_iterations)
        self.eta = eta
        self.mass = mass
        self.atol = atol  # TODO implement
        self.lamb = lamb

    def set_niterations(self, n):
        self.n = int(np.round(n))

    def Grad(self, X, y):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)

        for _ in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change
            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            change = (-1 * self.eta * grad) + momentum
            theta += change

        return theta

    def AdaGrad(self, X, y, delta=1e-7):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        r = np.zeros_like(theta)

        for _ in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change

            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            r += grad**2
            weights = self.eta / (delta + np.sqrt(r))
            change = (-1 * weights * grad) + momentum
            theta += change

        return theta

    def RMSGrad(self, X, y, delta=1e-7, decay=0.99):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        r = np.zeros_like(theta)

        for i in range(self.n):
            penalty = self.lamb * theta
            momentum = self.mass * change

            grad = 2 * ((1 / y.shape[0]) * X.T @ (X @ theta - y) + penalty)
            r = decay * r + (1 - decay) * grad**2

            weights = self.eta / np.sqrt(delta + r)
            change = (-1 * weights * grad) + momentum
            theta += change

        return theta

    def ADAM(self, X, y, delta=1e-8, decay_1=0.9, decay_2=0.9):
        theta = np.zeros(X.shape[1])
        change = np.zeros_like(theta)
        s = np.zeros_like(theta)  # first moment estimates
        r = np.zeros_like(theta)  # second moment estimates

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

        return theta
