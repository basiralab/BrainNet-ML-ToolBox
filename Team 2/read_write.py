# Code Owners: Göktuğ Güvercin - Uğur Tepecik - Ege Apak
# Code Editor: Göktuğ Güvercin

import numpy as np
import pandas as pd


def load_data(directory1, directory2):

    """
    It reads the content of training and testing data files.
    Then, it returns them as numpy arrays

    Parameters
    ----------
    directory1: directory of training file
    directory2: directory of testing file
    """

    tra_data = np.array(pd.read_csv(directory1))
    tst_data = np.array(pd.read_csv(directory2))
    return tra_data, tst_data


def split_data(dataset):

    """
    The "dataset" array is split into features and labels.
    CAUTION: "dataset" numpy array must contain label values at the last column.
    """

    labels = dataset[:, len(dataset[0]) - 1]
    features = dataset[:, :len(dataset[0]) - 1]
    return features, labels


def write_output(predictions, directory):

    size = len(predictions)
    indices = np.array([i for i in range(1, size + 1)])

    indices.shape = (size, 1)
    predictions.shape = (size, 1)
    submission_array = np.concatenate((indices, predictions), 1)

    np.savetxt(directory, submission_array, delimiter=",", fmt="%d", header="ID,Predicted", comments="")
