import numpy as np
import pandas as pd



class LDA():

    '''constructor to initialize weights'''
    def __init__(self, w, w0):
        self.w = w
        self.w0 = w0

    def fit(self, X, y):
        '''calculate probability for Y=0 and Y=1'''
        N_0 = y.count(0)
        N_1 = y.count(1)
        total = N_0+N_1
        probY_0 = N_0/(total)
        probY_1 = N_1/(total)
        '''calculate mean vector for each class'''
        X_0 = []
        X_1 = []
        for i,j in zip(X,y):
            if y[j] == 0:
                X_0.append(X[i])
            else:
                X_1.append(X[i])
        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        covariance = [] #todo: compute covariance
        try:
            cov_inv = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            print("matrix not invertible!")
        else:
            w0 = np.log((probY_1/probY_0)) - (0.5)*[(np.transpose(mean_1)).dot(cov_inv)].dot(mean_1) + (0.5)*[(np.transpose(mean_0)).dot(cov_inv)].dot(mean_0)
            w = cov_inv.dot(mean_1 - mean_0)
            self.w0 = w0
            self.w = w
    def predict(self, X):
        pass