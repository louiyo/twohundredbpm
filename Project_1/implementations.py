from cost import *
from gradients import *
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using gradient descent
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w -= gamma * gradient

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using stochastic gradient descent
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)

        loss = compute_mse(y, tx, w)
        w -= gamma*gradient

    return w, loss


def least_squares(y, tx):
    # Least squares regression using normal equations
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations
    (nrows, ncols) = tx.shape
    regularizer = np.identity(ncols) * lambda_*(2*nrows)
    w = np.linalg.solve(tx.T.dot(tx) + regularizer, tx.T.dot(y))
    loss = compute_mse(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent or SGD
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma*gradient

    loss = compute_log_likelihooh(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    # Regularized logistic regression using gradient descent or SGD
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma*gradient

    loss = compute_log_likelihooh(y, tx, w, lambda_)

    return w, loss
