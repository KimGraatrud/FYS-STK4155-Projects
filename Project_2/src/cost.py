import numpy as np

class Cost:

    def __init__(self, lamb1: float = None, lamb2: float = None, regularize_bias: bool = False):

        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.regularize_bias = regularize_bias

        if not(lamb1 == None) and not(lamb2 == None):
            print('Applying both L1 and L2 norm at the same time!')
            raise ValueError('Choose either lamb1 or lamb2 to be a value, not both.')


    def mse(self, predict, target):
        # Wb unused to not get errors when calling mse instead of penalized mse
        return np.sum((target - predict) ** 2) / len(target)

    def mse_der(self, predict, target):
        # Wb unused to not get errors when calling mse der instead of penalized mse der
        return (2 / len(predict)) * (predict - target)

    def _loss(self, Wb, n):
        """
        Computes the combined penalty over every layer.

        Uses logic to compute L1 or L2 penalty
        """
        loss = 0.0
        for (W,b) in Wb:
            if lamb1 is not None:
                loss += self.lamb1 * (np.sum(np.abs(W)) + np.sum(np.abs(b)))
            elif lamb2 is not None:
                loss += self.lamb2 * (np.sum(W**2) + np.sum(b**2))
        return loss/n

    def _grads(self, Wb, n):
        new_grads = []
        for (W,b) in Wb:

            # Check L1
            if lamb1 is not None:
                dW = self.lamb1 / n * np.sign(W)
                if self.regularize_bias: 
                    db = self.lamb1 / n * np.sign(b) 
                else:
                    db = np.zeros_like(b)

            # Check L2
            elif lamb2 is not None:
                dW = 2 * self.lamb2 / n * W 
                db = (2.0 * self.lamb2 / n) * b
            
            # No penalty case.
            else:
                dW = np.zeros_like(W)
                db = np.zeros_like(b)
            # Append
            new_grads.append((dW,db))

        return new_grads

    def penalized_mse(self, predict, target, Wb): 
        """
        Calculates the penalized result for either L1 or L2.

        Wb is the list of tuples calculated for each layer
        """

        mse = np.sum((target - predict) ** 2) / len(target)
        result = mse + self._costL1(W) + self._costL2(W)
        return result

    def penalized_mse_der(self, predict, target, Wb):
        """
        Calculates the penalized derivative result for either L1 or L2.

        Wb is the list of tuples calculated for each layer
        """
        n = target.shape
        dWb = self._grads(predict, target, n)
        return dWb

    def __call__(self):
        pass