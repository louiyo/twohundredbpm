# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False, parameter='PRI_jet_num'):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",",
                      skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)

    ids = x[:, 0].astype(np.int)
    input_values = x[:, 2:]
    headers = np.genfromtxt(data_path, delimiter=",",
                            dtype=str, max_rows=1)[2:]
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1
    yb[np.where(y == 's')] = 1

    parameter_id = np.where(headers == parameter)[0]

    values = set(input_values[:, parameter_id].reshape(-1,))  # [0, 1, 2, 3]
    print(len(values))

    data = [[i for i in zip(zip(input_values, yb), ids)
             if i[0][0][parameter_id] == x] for x in values]

    ids = [np.array([i[1] for i in x]) for x in data]

    input_data = [np.array([i[0][0] for i in x])for x in data]

    yb = [np.array([i[0][1] for i in x]).reshape(-1, 1) for x in data]

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
