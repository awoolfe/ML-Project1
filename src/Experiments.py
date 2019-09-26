######################################################
# Filename: Experiments.py
# Author: Thomas Racine
# Purpose: Run Experiments on our different models
######################################################
import sys
sys.path.append('/Users/ianbenlolo/Desktop/Mcgill/comp_551/ML-Project1/')
import src.load_files
import src.Linear_discriminant_analysis
import src.logistic_regression
import src.CrossValidation
import time
import numpy as np

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
acc = src.CrossValidation.CrossValidation(wines.copy(), model_lda_wine1, 5)
print(time.time() - start_time)
print(acc)

print('\nLinear regression: 1000 steps, learning rate 0.004 ')
start_time = time.time()
acc = src.CrossValidation.CrossValidation(wines.copy(), model_linear_regression_wine1, 5, params1)
print(time.time() - start_time)
print(acc)
