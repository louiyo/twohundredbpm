import numpy as np


def preprocess(data):

    data = remove_irrelevant_columns(data)

    for idx in range(len(data)):
        data[idx] = replace_non_defined(data[idx])
        data[idx] = data[idx][:, np.nanstd(data[idx], axis=0) != 0]
        data[idx], _, _ = standardize(data[idx])

    return data


def standardize(x):
    """Standardize the original data set."""

    mean_x = np.mean(x, axis=0)
    x -= mean_x
    std_x = np.std(x, axis=0)
    x /= std_x

    return x, mean_x, std_x


def replace_non_defined(data):
    """
        Replace supplementary non defined values by the median of the column.
    """
    data[data == -999] = np.nan
    return np.where(np.isnan(data), np.nanmedian(data, axis=0), data)


def remove_irrelevant_columns(data):
    """
        Remove the columns where all examples from a specific PRI_jet_num have
        non defined values (-999), thus being irrelevant to the model.
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
