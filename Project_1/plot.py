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


# lambda donné meilleur trouvé --> courbe qui va du degré 8 au 16  qui plot l'accuracy
def display_ridge_performances(performances, degrees, lambdas):
    """Display the accuracy for different parameters."""
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    ax = [ax1, ax2, ax3, ax4]

    for idx, performance in enumerate(performances) :
        acc_means = []
        for i in range(1,len(degrees)):
            # acc_means_lambda = []
            max_acc = 0
            for j in range(1, len(lambdas)):
                for x in performance :
                    if (x[0]==degrees[i] and x[1]==lambdas[j] and x[4] > max_acc):
                        max_acc = x[4]
                        # acc_mean_lambda.append(x[4])
            acc_means.append(max_acc)

            # lambdas = [round(x,5) for x in lambdas]
        ax[idx].plot(degrees, acc_means, '-r')
        ax[idx].invert_yaxis()
        ax[idx].set_xlabel('degrees', labelpad=20)
        ax[idx].set_ylabel('accuracy', labelpad=20)
        # ax[idx].set_title('Accuracy of data using Ridge Regression\n\n')
    plt.savefig('ridge_regression.pdf', bbox_inches='tight')
    plt.show()
