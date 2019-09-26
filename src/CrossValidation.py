#########################################################
# File name: CrossValidation.py
# Author: Thomas Racine
# Purpose: implement k-fold cross validation
#########################################################

import numpy as np
import src.eval


# This is where we run the training and validation steps of our model
# Input:
# data - the data set
# model - model to train
# fold - number of folds
def CrossValidation(data, model, fold, params = None):
    folds = np.array_split(data, fold, axis=0)  # we separate the data set into k different sub lists
    accuracy = 0
    for i in range(fold):
        trainingset = np.zeros(shape=(0,len(data[1])))
        for j in range(fold):
            if j != i:
                trainingset = np.concatenate((trainingset, folds[j]))  # we create our training set by adding everysub lists except the ith one used for validation
        if params is not None:
            assert(len(params) == 3, 'params should have iterations, lr, lr_func')
        else:
            params = []

        model.fit(trainingset[:,:-1], trainingset[:,-1], *params)  # train the model with the training set

        predictions = model.predict(folds[i][:,:-1])  # predict using validation set
        accuracy += src.eval.evaluate_acc(folds[i][:,-1], predictions)
    return accuracy / fold
