import numpy as np
import pandas as pd



class LDA():

    '''constructor to initialize weights'''
    def __init__(self, w, w0):
        self.w = w
        self.w0 = w0

    def fit(self, X, y):
        '''calculate probability for Y=0 and Y=1'''
        print(X)
        N_0 = np.count_nonzero(y == 0)
        N_1 = np.count_nonzero(y == 1)
        total = N_0 + N_1
        probY_0 = N_0/(total)
        probY_1 = N_1/(total)
        '''calculate mean vector for each class'''
        X_0 = np.array([row for i,row in enumerate(X) if y[i] == 0])
        X_1 = np.array([row for i,row in enumerate(X) if y[i] == 1])

        mean_0 = np.mean(X_0, axis=0)
        mean_1 = np.mean(X_1, axis=0)
        covariance = np.cov(np.transpose(X))
        try:
            cov_inv = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            print("matrix not invertible!")
        else:
            w0 = np.log((probY_1/probY_0)) - (0.5)*((np.transpose(mean_1)).dot(cov_inv)).dot(mean_1) + (0.5)*((np.transpose(mean_0)).dot(cov_inv)).dot(mean_0)
            w = cov_inv.dot(mean_1 - mean_0)
            self.w0 = w0
            self.w = w

    def predict(self, X):
        prediction = []
        for i in X:
            prediction.append(self.w0 + np.transpose(i).dot(self.w))
        y =[1 if i > 0 else 0 for i in prediction]
        print(y)
        return y


if __name__ == '__main__':
    import load_files
    wine, wine_headers = load_files.load_wine()
    X = wine[:, :-1]
    y = wine[:, -1]

    print(X.shape, y.shape)
    model = LDA(0,0)
    model.fit(X, y)
    model.predict(X)

