import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cross_validation import *

def display_cross_validation(performances, degrees):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(10,10))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    ax = [ax1, ax2, ax3, ax4]
    color = ['r','b','g','c','k']

    for idx, performance in enumerate(performances) :
        for idx_color, degree_ in enumerate(degrees):
            acc = [(lambda_, acc) for degree, lambda_, gamma_, mean_weight, acc in performance if degree == degree_]
            ax[idx].semilogx(*zip(*acc), marker=".", color= color[idx_color], label='Accuracy with degree = {}'.format(degree_))
            ax[idx].set_xlabel("lambda")
            ax[idx].set_ylabel("accuracy")
            ax[idx].set_title("Cross validation accuracy of model {}".format(idx+1))
        ax[0].legend(loc=(2.3,0.25))
    plt.show()
    plt.savefig("cross_validation_accuracy")
