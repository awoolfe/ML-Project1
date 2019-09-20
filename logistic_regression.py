import numpy as np

class Logistic():
    def _init_(self, data, labels, learning_rate= 0.01):
        self.lr = learning_rate
        self.labels = labels
        self.data = data

    def sigmoid(z): 
    """
    sigmoid function 
    """
        return 1./(1.+math.exp(-z))
    def predict(X, yhat, threshhold= 0.5):
        
