#########################################################
# File name: CrossValidation.py
# Author: Thomas Racine
# Purpose: implement k-fold cross validation
#########################################################

import numpy as np


# This is where we run the training and validation steps of our model
# Input:
# data - the data set
# model - model to train
# k - number of folds
def CrossValidation(data, model, fold):
    folds = np.split(data, fold)  # we separate the data set into k different sub lists
    for i in range(fold):
        trainingset = []
        for j in range(fold):
            if j != i:
                trainingset = trainingset + folds[j]  # we create our training set by adding everysub lists except the ith one used for validation
            model.fit(trainingset)  # train the model with the training set
            model.predict(folds[i])  # predict using validation set
