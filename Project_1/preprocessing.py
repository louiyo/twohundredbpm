import numpy as np


def standardize(x):
    """Standardize the original data set."""

    mean_x = np.mean(x, axis=0)
    x -= mean_x
    std_x = np.std(x, axis=0)
    x /= std_x

    return x, mean_x, std_x


def extract_PRI_jet_num(data):
    """
        Extract the PRI_jet_num from the data set and assign the rows to the
        4 groups into a dictionnary, namely 0,1,2, and 3.
    """
    new_data = {0: x[x[:, 24] == 0],
                1: x[x[:, 24] == 1],
                2: x[x[:, 24] == 2],
                3: x[x[:, 24] == 3]}
    return new_data


def replace_non_defined(data):
    """
        Replace supplementary non defined values by the median of the column.
    """
    data[data == -999] = np.nan
    return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)


def remove_non_defined_columns(data):
    """
        Remove the columns where all examples from a specific PRI_jet_num have
        non defined values (nan).
    """

    for jet_num in range(len(data)):

        index = 0
        mask = []

        means = np.mean(data[jet_num], axis=0)

        for mean in means:
            if(mean == -999.0):
                mask.append(index)
            index += 1

        data[jet_num] = np.delete(data[jet_num], mask, axis=1)

    return data
