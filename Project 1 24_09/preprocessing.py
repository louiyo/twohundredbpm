import numpy as np


def standardize(x, mean_x = 0, std_x = 0):
    """Standardize the original data set."""
    print(mean_x)
    if mean_x == 0 :
         mean_x = np.nanmean(x, axis = 0)
    if std_x == 0 :
        std_x = np.nanstd(x, axis = 0)

    x = x - mean_x / std_x

    return x, mean_x, std_x

def extract_PRI_jet_num(data):
    """
        Extract the PRI_jet_num from the data set and assign the rows to the
        4 groups into a dictionnary, namely 0,1,2, and 3.
    """
    new_data = { 0: x[x[:,24] == 0],
                 1: x[x[:,24] == 1],
                 2: x[x[:,24] == 2],
                 3: x[x[:,24] == 3]}
    return new_data

def replace_non_defined(data):
    """
        Replace supplementary non defined values by the median of the column.
    """
    data[data == -999] = np.nan
    return np.where(np.isnan(data), np.nanmedian(data, axis = 0), data)

def remove_non_defined_columns(data):
    """
        Remove the columns where all examples from a specific PRI_jet_num have
        non defined values (nan).
    """
    # data[data == -999] = np.nan
    non_defined_constant = -999
    new_data = [np.delete(data[PRI_jet_num], data[PRI_jet_num] == non_defined_constant, axis = 1)
                   for PRI_jet_num in range(len(data))]

    # [np.delete(isnan(data[PRI_jet_num])) for PRI_jet_num in range(0,4)]

    return new_data
