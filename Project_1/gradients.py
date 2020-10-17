import numpy as np

def compute_gradient(y, tx, w):
    """
        Compute the gradient of the MSE for linear regression.
    """
    N = len(y)
    e = y - tx.dot(w)
    gradient = - tx.T.dot(e) / N

    return gradient

def compute_stoch_gradient(y, tx, w):
    """
        Compute the gradient of the MSE for linear regression.
    """
    return compute_gradient(y, tx, w)
    #raise NotImplementedError

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_logistic_gradient(y, tx, w, lambda_ = 0):
    """
        Compute the gradient of the logistic function for logistic regression.
    """
    gradient = tx.T.dot(sigmoid(tx.dot(w)) - y)
    regularizer = lambda_ * np.linalg.norm(w)

    return gradient + regularizer
