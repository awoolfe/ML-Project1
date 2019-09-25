import math
from scipy.special import expit
import numpy as np

class Logistic:
    def __init__(self,params, iterations):
        """
        initialize random weights
        params is the number of parameters
        iterations is how many time we do gradiant descent
        """
        self.weights = np.random.rand(params+1)
        self.iterations = iterations
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
    def fit(self, X, y, lr = 0.004):
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

        #assert(n == y.shape[0], "the data and output array shapes are not equal")
        
        weights = self.weights
        
        print(X.shape, y.shape, weights.shape)
        #it = 0
        
        for i in range(self.iterations):
            sum_ = np.zeros((len(weights),))
            for j,row in enumerate(X):
                sig = self.sigmoid(np.dot(weights, row.T))
                sum_ += np.multiply(row, (y[j] - sig))
            
            weights += np.multiply(sum_, lr/np.round(np.log10(i),0) if i > 10 else lr)


            self.weights = weights
            #print(self.loglikelyhood(X,y))
        print('weights: ',self.weights)

    def predict(self, X, threshhold= 0.5):
        X0 = np.zeros((X.shape[0]))
        X0[X0 == 0] = 1
        X_prime = np.concatenate((X0[:, np.newaxis], X),axis=1)
        z = np.dot(X_prime,self.weights)
        prob = [self.sigmoid(a) for a in z]
        
        return [1 if i > 0.5 else 0 for i in prob]   

        #if prob > 0.5:
        #    return 1
        #else:
        #    return 0

if __name__ == '__main__':
    import load_files
    import CrossValidation
    wine,wine_headers = load_files.load_wine()
    X = wine[:,:-1]
    y = wine[:,-1]
    
    print(X.shape, y.shape)
    model = Logistic(X.shape[1], 1000)
    print(CrossValidation.CrossValidation(wine, model, 5))
    


