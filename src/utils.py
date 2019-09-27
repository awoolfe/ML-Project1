import sys
sys.path.append('/Users/ianbenlolo/Desktop/Mcgill/comp_551/ML-Project1/')

import load_files
import Linear_discriminant_analysis
import logistic_regression
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


def get_uncorrelated_dataset(data,data_headers, threshold = 0.6):

    # get correlation coefficient matrix using numpy
    corrcoefficients = np.corrcoef(data[:,:-1], rowvar = False)

#   save dictionary of correlated item with absolute threshhold
    d= {}
    for i in range(len(corrcoefficients)):
        for j in range(len(corrcoefficients)- i):
            if  np.abs(corrcoefficients[i][j+i]) > threshold:

                if i in d:
                    d[i].append(data_headers[j+i])
                else:
                    d[i] = []
#   save the headers which we want to remove
    headers_to_rem = []
    s = set(data_headers)
    #print('d: ', d)
    for key, value in d.items():
        if len(value) > 0:
            headers_to_rem.append(data_headers[key])
        #for i in value:
        #    headers_to_rem.append(i)
    headers_to_rem = set(headers_to_rem)
    #print('to remove: ', headers_to_rem)
    #remove the headers and store index of uncorrelated items
    #this includes the last column with the classification
    indeces_to_keep = []
    for item in s.difference(headers_to_rem):
        indeces_to_keep.append(data_headers.index(item))
#    indeces_to_keep =[1,2,3,4,6,7,8,9,10,11]
    #return the array containing only the uncorrelated params
    return  np.take(data, sorted(indeces_to_keep),axis = 1), [data_headers[x] for x in indeces_to_keep]


def CrossValidation(data, model, fold, params = None):
    # np.random.shuffle(data)  # last step of preprocessing
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
    wines, wine_headers = load_files.load_wine()

    ## testing uncorrelated data only
    uncorr_wines, uncorr_headers = get_uncorrelated_dataset(wines,wine_headers)
    cancer, cancer_headers = load_files.load_cancer()

    model_lda_wine1 = Linear_discriminant_analysis.LDA(0, 0)
    model_lda_cancer1 = Linear_discriminant_analysis.LDA(0, 0)

    model_logistic_regression_cancer1 = logistic_regression.Logistic(cancer.shape[1])
    model_logistic_regression_wine1 = logistic_regression.Logistic(wines.shape[1])
    model_logistic_regression_uncorr_wine = logistic_regression.Logistic(uncorr_wines.shape[1])
    params1 = [1000, 0.01, (lambda x, y: x), 0.5]

    model_linear_regression_wine2 = logistic_regression.Logistic(wines.shape[1])
    params2 = [1000, 0.004, (lambda x, y: x / np.round(np.log10(y), 0) if y > 10 else x), 0.5]

    #Experiments of different learning rates
    print('*******Learning Rates and Functions  x = learning rate y = step# *******')
    print('Learning rate:0.01; function (x,y)=> x')
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params1))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params1))
    print('Learning rate:0.005; function (x,y)=> x')
    params3 = [1000, 0.005, (lambda x, y: x), 0.5]
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params3))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params3))
    print('Learning rate:0.001; function (x,y)=> x')
    params4 = [1000, 0.001, (lambda x, y: x), 0.5]
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params4))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params4))
    print('Learning rate:0.0001; function (x,y)=> x')
    params5 = [1000, 0.0001, (lambda x, y: x), 0.5]
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params5))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params5))
    print('Learning rate:0.004; function (x,y)=> x')
    params6 = [1000, 0.004, (lambda x, y: x), 0.5]
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params6))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params6))
    print('Learning rate:0.004; function (x,y)=> x/log_10(y) if y > 10 else x')
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params2))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params2))
    print('Learning rate:0.004; function (x,y)=> 10x/y')
    params7 = [1000, 0.004, (lambda x, y: 10 * x / (y + 1)), 0.5]
    print('wine: %s' % CrossValidation(wines, model_logistic_regression_wine1, 5, params7))
    print('cancer: %s' % CrossValidation(cancer, model_logistic_regression_cancer1, 5, params7))

    #
    print('*******Runtime and accuracy of our models*******')
    print("Dataset Wine")
    print('Linear Discriminant Analysis')
    start_time = time.time()
    acc = CrossValidation(wines.copy(), model_lda_wine1, 5)
    print(time.time() - start_time)
    print(acc)

    print('\nLinear regression: 1000 steps, learning rate 0.001 ')
    start_time = time.time()
    acc = CrossValidation(wines.copy(), model_logistic_regression_wine1, 5, params4)
    print(time.time() - start_time)
    print(acc)

    print('Dataset Cancer')
    start_time = time.time()
    acc = CrossValidation(cancer.copy(), model_lda_cancer1, 5)
    print(time.time() - start_time)
    print(acc)

    print('\nLinear regression: 1000 steps, learning rate 0.001 ')
    start_time = time.time()
    acc = CrossValidation(cancer.copy(), model_logistic_regression_cancer1, 5, params4)
    print(time.time() - start_time)
    print(acc)

    print("*******Testing different feature set on the wine data set using logistic regression******")
    print('All features')
    print(wine_headers)
    acc = 0.
    for i in range(5):
        acc += CrossValidation(wines, model_logistic_regression_wine1, 5, params4)
    print('Average accuracy of 25 runs of the algorithm: {}'.format(acc/5))

    print("Uncorrelated features")
    print(uncorr_headers)
    acc = 0.
    for i in range(5):
        acc += float(CrossValidation(uncorr_wines, model_logistic_regression_uncorr_wine, 5, params4))
    print('Average accuracy of 25 runs of the algorithm: {}'.format(acc/5))

if __name__ == '__main__':
    main()
