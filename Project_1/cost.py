from gradients import *
import numpy as np


def compute_mse(y, tx, w):
    """
    Compute loss with the mean square error
    """
    N = len(y)
    e = y - tx.dot(w)
    loss = np.sum(e**2, axis=0) / (2*N)

    return loss


def compute_mae(y, tx, w):
    """
    Compute loss with the mean absolute error
    """
    e = y - tx.dot(w)
    loss = np.sum(abs(e), axis=0) / (2*N)

    return loss


def compute_log_likelihood(y, tx, w, lambda_=0):
    """
        Compute the negative log likelihood of the data
        lambda_ : if = 0 it computes it for logistic regression, else it is
                Regularized logistic regression.
    """
    sigma = sigmoid(tx.dot(w))
    cost_ = - np.log(sigma) - (y * tx.dot(w))
    cost = np.sum(cost_)

    regression = (lambda_ / 2) * (np.linalg.norm(w)**2)

    return cost + regression
    #raise NotImplementedError
