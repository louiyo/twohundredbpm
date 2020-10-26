# CS-433 Course - 2020/2021
*Participants : Anas Machraoui, Camil Hamdane, Nathan Girard.*

## What's there ?
In this repository you can find the first project of the course of Machine learning (CS-433) of epfl. The second project of the semester is to be released soon.

## Project 1 :

### Description :
The goal of this project is to predict whether a particule is a boson or not depending on specific features. The dataset we analyize are the training and test set, of size (250000,30) and (568238,30) respectively. 

### Architecture :
The project is coded with Python3 using the numpy library only, and seaborn for visualization.

### Implementations :
For the computer to learn the prediction model, we give a variety of methods including **Least Squares** (with (stochastic) gradient descent), **ridge regression**, and **logistic regression**. These are the basic implementations :

- least squares GD(y, tx, initial w, max iters, gamma)
- least squares SGD(y, tx, initial w, max iters, gamma)
- least squares(y, tx)
- ridge regression(y, tx, lambda )
- logistic regression(y, tx, initial w, max iters, gamma)
- reg logistic regression(y, tx, lambda , initial w, max iters, gamma)

However these methods are not enough to achieve good performance. First we noticed that depending on the value of the *PRI_jet_num* feature of the data, some of the other features are irrelevant for predicting the label of this particle (we found that for the jet num zero, less than twenty out of 30 are defined, whereas 29 are in model 3). Therefore we *split* the dataset in *4*, already sorting the examples by their PRI_jet_num value. The algorithm then learns 4 different model.

Thus we added the **preprocessing** part, which *standardizes* the data, performs *polynomial expansion* on the features, removes *useless features* and replaces non defined data by the median of the corresponding feature when necessary. 

In addition, **cross-validation** is implemented in order to find the best hyper-parameter values depending on the Machine Learning method you want to use. 

### How to run the algorithm :
All the files of the project contains the different methods named above, such that running the run.py file allows your computer to learn the model with the training data. Then it predicts the labels of the examples in the test dataset. 

There is an alternative way to run the code : the Jupyter Notebook (Project.ipynb). With this tool, one can cross-validate the hyper-parameters values, compute the prediction model (ridge regression by default) and visualize training accuracy for example.

### Libraries and Programming language:
- [Numpy](https://numpy.org)
- [Python](https://www.python.org)
- [Seaborn](https://seaborn.pydata.org/)

## Project 2 :

# TBA
