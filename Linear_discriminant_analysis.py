import numpy as np
import pandas as pd



class LDA():
    '''iterations: number of iterations of gradient descent,
     attribute: list of indicies of model attributes
     data: numpy array
     y = index of classification column
     labels: column headers'''
    def __init__(self, iterations, data, attribute, labels, y):
        self.iter = iterations
        self.data = data
        self.y = y
        self.attribute = attribute
        self.labels = labels

    def fit(self, X, y):
        '''calculate probability for Y=0 and Y=1'''

        pass
    def predict(self, X):
        pass