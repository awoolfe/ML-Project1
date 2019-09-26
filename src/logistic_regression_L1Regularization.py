import math
from scipy.special import expit
import numpy as np


class LogisticRegular:
    def __init__(self, params, iterations, lr, lr_func, lam):
        """
        initialize random weights
        params is the number of parameters
        iterations is how many time we do gradiant descent
        """
        self.weights = np.random.rand(params)
        self.iterations = iterations
        self.lr = lr
        self.lr_func = lr_func
        self.lam = lam

    #    def sigmoid(self,z):
    #        """
    #        sigmoid function
    #        """
    #        return 1./(1.+math.exp(-z))

    def loglikelyhood(self, X, y):
        sum_ = 0
        weights = self.weights

        for i in range(len(X)):
            sig = self.sigmoid(np.dot(weights.T, X[i]))
            sum_ += y[i] * np.log(sig) + (1 - y[i]) * np.log(1 - sig) + self.lam*(self.sign(weights))
        return sum_

    def sigmoid(self, x):
        """
        Numerically stable sigmoid function.
        Taken from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: np.array (m x n)
            The data
        Y: np.array (m x 1))
            The training output data were fitting to
        lr: float
            the learning rate ("alpha")

        """
        X = np.insert(X, 0, 1, axis=1)

        # assert(n == y.shape[0], "the data and output array shapes are not equal")

        self.weights = np.random.rand(np.shape(self.weights)[0])
        weights = self.weights

        #print(X.shape, y.shape, weights.shape)
        # it = 0
        #print(weights)
        for i in range(self.iterations):
            sum_ = np.zeros((len(weights),))
            for j, row in enumerate(X):
                sig = self.sigmoid(np.dot(weights, row.T))
                sum_ += np.multiply(row, (y[j] - sig))


            weights += np.multiply(sum_, self.lr_func(self.lr, i))
            # for k in weights:
            #     k += self.lam*(self.sign(k))
            self.weights = weights
            # print(self.loglikelyhood(X,y))
        print('weights: ',self.weights)

    def sign(self, weights):
        #print(weights)
        weights[weights<0] = -1
        weights[weights>0] = 1
        #print(weights)
        return weights

    def predict(self, X, threshhold=0.5):
        X0 = np.zeros((X.shape[0]))
        X0[X0 == 0] = 1
        X_prime = np.concatenate((X0[:, np.newaxis], X), axis=1)
        z = np.dot(X_prime, self.weights)
        prob = [self.sigmoid(a) for a in z]

        return [1 if i > 0.5 else 0 for i in prob]

        # if prob > 0.5:
        #    return 1
        # else:
        #    return 0


if __name__ == '__main__':
    import load_files
    import CrossValidation

    wines, wine_headers = load_files.load_wine()
    #X = wines[:, :-1]
    #y = wines[:, -1]
    #print(X.shape, y.shape)
    model = LogisticRegular(wines.shape[1], 1000, 0.004, (lambda x, y: x), 0.1)
    print(CrossValidation.CrossValidation(wines, model, 5))

    # cancer, cancer_header = load_files.load_cancer()
    #
    # x = cancer[:, :-1]
    # model2 = LogisticRegular(x.shape[1], 1000)
    # print(CrossValidation.CrossValidation(cancer, model2, 5))
    #
    #
