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

# If needed for initializing the weights :
# seed = 12

# Loading of the train and test data :
print('Loading train and test Data')

# TODO: download train data and supply path here
DATA_TRAIN_PATH = 'Data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = 'Data/test.csv'  # TODO: download train data and supply path here
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# TODO: fill in desired name of output file for submission
OUTPUT_PATH = 'Data/output.csv'

# Setting hyper-parameters values :
degrees = [5, 10, 12, 15]
lambdas = [0.0001, 0.01, 0.01, 0.1]
# gammas = [0.0001, 0.001, 0.01, 0.1]

# Isolating the 4 groups with distinct PRI_jet_num values:
#train_x_jet = extract_PRI_jet_num(tX)
#test_x_jet = extract_PRI_jet_num(tX_test)

# Removing NaN columns from the different groups (defined above):
train_x_jet = remove_non_defined_columns(train_x_jet)
test_x_jet = remove_non_defined_columns(test_x_jet)

y_pred = np.zeros(len(y))

# Mauvaise itération - itérer sur chaque dataset spécifique au particules 0,1,2, et 3.
# Puis compute le model pour les particules avec le PRI_jet_num spécifique.
for idx in range(len(tX)):
    # initializing the prediction vector :
    y_pred_ = np.zeros(len(tX[idx]))
    print(y_pred_.shape)

    # extracting the values of group idx :
    train_x_jet_ = train_x_jet[idx]
    train_y_jet_ = y[idx]
    test_x_jet_ = test_x_jet[idx]

    # Removing additional outliers :
    train_selected_x_jet = replace_non_defined(train_x_jet_)
    test_selected_x_jet = replace_non_defined(test_x_jet_)

    # standardize :
    tX_train, mean_x_train, std_x_train = standardize(train_selected_x_jet)
    tX_test, _, _ = standardize(test_selected_x_jet, mean_x_train, std_x_train)

    # Polynomial feature expansion :
    tX_train_poly = build_poly_tx(tX_train, degrees[idx])
    tX_test_poly = build_poly_tx(tX_test, degrees[idx])

    w_, loss_ = ridge_regression(train_y_jet_, tX_train_poly, lambdas[idx])

    accuracy_ = compute_accuracy(train_y_jet_, tX_train_poly, w_)
    print('The accuracy of model [] is equal to []'.format(int(idx), accuracy_))

    # Computing test accuracy : (To be changed -> à mettre dans une nouvelle boucle)
    y_pred_ = predict_labels(w_, tX_test_poly)
    y_pred[test_x_jet_[1] == idx] = y_pred_.flatten()

    test_acc = np.mean(y_pred == y_test, axis=0)
    print('The accuracy over the test data is equal to []'.format(test_acc))

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
