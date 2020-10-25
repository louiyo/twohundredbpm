# matplotlib inline
import numpy as np
# import matplotlib.pyplot as plt
# load_ext autoreload
# autoreload 2

from proj1_helpers import *
from helpers import *
from implementations import *
from cost import *
from gradients import *
from preprocessing import *
from cross_validation import *


print('Loading train and test Data...')

DATA_TRAIN_PATH = 'Data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = 'Data/test.csv'
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'Data/output.csv'

print('Loading done')

# setting up parameters
degrees = [11, 12, 13, 12]
lambdas = [0.001009312, 0.001009312, 1.1212e-05, 0.0000696969]

# Removing irrelevant columns from the jet groups:
tX = preprocess(tX)
tX_test = preprocess(tX_test)

# Instancing predictions
y_pred = []

for idx in range(len(tX)):
    print("Beginning training on model ", idx+1)

    # extracting the values of specific group :
    train_x_jet = tX[idx]
    train_y_jet = y[idx]
    test_x_jet = tX_test[idx]

    # Polynomial feature expansion :
    tX_train_poly = polynomial_expansion(train_x_jet, degrees[idx])
    tX_test_poly = polynomial_expansion(test_x_jet, degrees[idx])

    w_, loss_ = ridge_regression(train_y_jet, tX_train_poly, lambdas[idx])

    accuracy_ = compute_accuracy(train_y_jet, tX_train_poly, w_)
    print('The accuracy of model {} is equal to {}'.format(int(idx+1), accuracy_))

    y_pred_jet = predict_labels(w_, tX_test_poly)

    y_pred.append(y_pred_jet)


y_total = np.zeros(len(np.hstack(ids_test)))

min_id_test = min(np.hstack(ids_test))

ids_total = np.arange(len(y_total))
ids_total += min_id_test

for jet_num in range(len(y)):
    for j in range(len(y_pred[jet_num])):
        y_total[ids_test[jet_num][j] - min_id_test] = y_pred[jet_num][j]

y_total.reshape(-1, 1)

print('Building output file in ', OUTPUT_PATH)
create_csv_submission(ids_total, y_total, OUTPUT_PATH)
