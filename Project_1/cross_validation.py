from implementations import *
from helpers import *
import numpy as np


def polynomial_expansion(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_index(y, k_fold, seed=12):
    """
        Building the the indices for k-fold cross validation.
    """
    nrow = y.shape[0]
    num_ex = int(k_fold/nrow)

    np.random.seed(seed)
    index = np.random.permutation(nrow)

    k_fold_index = [index[k*num_ex: (k+1) * num_ex] for k in range(k_fold)]

    return np.array(k_fold_index)


def compute_model(y, x, w0, k, indices, gamma_, lambda_, max_iters, model):
    """
        Compute the weights of the model using the specified model.
    """
    # Splitting into training and validation data :
    x_test = x[indices[k]]
    y_test = y[indices[k]]

    # Removing the test examples (rows) from the indices :
    indices = (np.delete(indices, k, axis=0)).flatten()

    x_train = x[indices]
    y_train = y[indices]

    # training the model :
    if model == "least_squares_GD":
        w_, loss_ = least_squares_GD(y_train, x_train, w0, max_iters, gamma_)
    elif model == "least_squares_SGD":
        w_, loss_ = least_squares_SGD(y_train, x_train, w0, max_iters, gamma_)
    elif model == "least_squares":
        w_, loss_ = least_squares(y_train, x_train)
    elif model == "ridge_regression":
        w_, _ = ridge_regression(y_train, x_train, lambda_)
    elif model == "logistic regression":
        w_, loss_ = logistic_regression(y_train, x_train, w0, max_iters, gamma_)
    elif model == "reg_logistic_regression":
        w_, loss_ = reg_logistic_regression(y_train, x_train, w0, max_iters, gamma_, lambda_)
    else:
        raise NameError("Invalid model : {}".format(model))

    # Compute the accuracy of the model with the validation dataset :
    acc_ = compute_accuracy(y_test, x_train, w_)

    return w_, acc_


def cross_validation(y, tX, w0, model="ridge_regression", k_fold=12,
                     degrees=[1], lambdas=[0], gammas=[0], max_iters=50):
    """
        Implementing cross_validation on a given model.
    """
    indices = build_index(y, k_fold)
    performances = []
    accuracy_model = 0.0
    best_parameters = (0, 0, 0)

    for degree in degrees:
        tX_poly = polynomial_expansion(tX, degree)
        w0_ = init_weights(tX_poly)

        for lambda_ in lambdas:
            for gamma_ in gammas:
                acc_ = []
                w_ = []

                for k in range(k_fold):
                    w, acc = compute_model(y, tX_poly, w0_, k, indices, gamma_,
                                           lambda_, max_iters, model)

                    acc_.append(acc)
                    w_.append(w)

                mean_weights = np.mean(w_, axis=0)
                mean_accuracy = np.mean(acc_)
                print("mean for ", lambda_, " ", mean_accuracy)
                performances.append(
                    (degree, lambda_, gamma_, mean_weights, mean_accuracy))

                # Implement best model : print the hyper-paramaters values of the best model:
                if mean_accuracy > accuracy_model:
                    accuracy_model = mean_accuracy
                    best_parameters = (degree, lambda_, gamma_)

    return performances, best_parameters
