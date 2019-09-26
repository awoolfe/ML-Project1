import sys
sys.path.append('/Users/ianbenlolo/Desktop/Mcgill/comp_551/ML-Project1/')
import src.load_files
import src.Linear_discriminant_analysis
import src.logistic_regression
import time
import numpy as np
# This is where we evaluate the accuracy of our models
# Input:
# target_y : result that we wish to obtain based on real data
# true_y : results obtained by the model


def evaluate_acc(target_y, true_y):
    correct_labels = 0
    if len(target_y) != len(true_y):  # to prevent indexing exceptions
        print("can't compare those sets, not the same size")
        return -1  # return error code
    for i in range(len(target_y)):
        if target_y[i] == true_y[i]:
            correct_labels += 1  # we count how many labels the model got right
    return correct_labels/len(target_y)  # we return the ratio over correct over total



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
        accuracy += evaluate_acc(folds[i][:,-1], predictions)
    return accuracy / fold

def main():
    wines, wine_headers = src.load_files.load_wine()
    cancer = src.load_files.load_cancer()

    model_lda_wine1 = src.Linear_discriminant_analysis.LDA(0,0)

    model_linear_regression_wine1 = src.logistic_regression.Logistic(wines.shape[1])
    params1 = [1000, 0.004, lambda x, y: x]

    model_linear_regression_wine2 = src.logistic_regression.Logistic(wines.shape[1])
    params2 = [1000, 0.004,  lambda x, y: x/np.round(np.log10(y),0) if y > 10 else x]

    print('*******Runtime and accuracy of our models*******')
    print('Linear Discriminant Analysis')
    start_time = time.time()
    acc = CrossValidation(wines.copy(), model_lda_wine1, 5)
    print(time.time() - start_time)
    print(acc)

    print('\nLinear regression: 1000 steps, learning rate 0.004 ')
    start_time = time.time()
    acc = CrossValidation(wines.copy(), model_linear_regression_wine1, 5, params1)
    print(time.time() - start_time)
    print(acc)

if __name__ == '__main__':
    main()