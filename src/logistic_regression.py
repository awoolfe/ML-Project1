import math
from scipy.special import expit
import numpy as np

class Logistic:
    def __init__(self,params):
        """
        initialize random weights
        params is the number of parameters
        """
        self.weights = np.random.rand(params)
#    def sigmoid(self,z):
#        """
#        sigmoid function 
#        """
#        return 1./(1.+math.exp(-z))


    def loglikelyhood(self, X, y):
        sum_ = 0
        weights = self.weights

        for i in range(len(X)):
            sig = self.sigmoid(np.dot(weights.T,X[i]))
            sum_ += y[i]*np.log(sig)+ (1-y[i])*np.log(1-sig)
        return sum_
    def sigmoid(self,x):
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
    def fit(self, X, y, iterations, lr = 0.1):
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
#        X = np.insert(X, 0, 1, axis=1)

        #assert(n == y.shape[0], "the data and output array shapes are not equal")
        
        weights = self.weights
        
        print(X.shape, y.shape, weights.shape)
        #it = 0
        
        for i in range(iterations):
            sum_ = 0.
            for j,row in enumerate(X):
                sig = self.sigmoid(np.dot(weights, row.T))
                sum_ += row * (y[j] - sig)
            
            weights += lr*sum_


            self.weights = weights
            print(self.loglikelyhood(X,y))    
        print('weights: ',self.weights)

    def predict(self, X, threshhold= 0.5):
        z = np.dot(X,self.weights)
        prob = sigmoid(z)
        
        return [1 if i > 0.5 else 0 for i in prob]   

        #if prob > 0.5:
        #    return 1
        #else:
        #    return 0

if __name__ == '__main__':
    import load_files 
    wine,wine_headers = load_files.load_wine()
    X = wine[:,:-1]
    y = wine[:,-1]
    
    print(X.shape, y.shape)
    model = Logistic(X.shape[1])
    model.fit(X,y,100)
    


