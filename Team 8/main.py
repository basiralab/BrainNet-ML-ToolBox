"""
Target Problem:
---------------
* A classifier for the diagnosis of Autism Spectrum Disorder (ASD)

Proposed Solution (Machine Learning Pipeline):
----------------------------------------------
* Standard Scaling -> PCA -> Logistic Regression

Input to Proposed Solution:
---------------------------
* Directories of training and testing data in csv file format
* These two types of data should be stored in n x m pattern in csv file format.

  Typical Example:
  ----------------
  n x m samples in training csv file (n number of samples, m - 1 number of features, ground truth labels at last column)
  k x s samples in testing csv file (k number of samples, s number of features)

* These data set files are ready by load_data() function.
* For comprehensive information about input format, please check the section
  "Data Sets and Usage Format of Source Codes" in README.md file on github.

Output of Proposed Solution:
----------------------------
* Predictions generated by learning model for testing set
* They are stored in "submission.csv" file.

Code Owner:
-----------
* Copyright © Team 8. All rights reserved.
* Copyright © Istanbul Technical University, Learning From Data Spring 2019. All rights reserved. """

import csv
import warnings
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(x_paths):

    """
    The method reads train and test data from data set files.
    Then, it splits train data into two pieces: features and labels.

    Parameters
    ----------
    x_paths: directory of train and test data files

    """

    data = np.matrix(np.genfromtxt(x_paths+'train.csv', delimiter=','))
    x_train = np.asarray(data[1:, 0:595])
    y_train = np.asarray(data[1:, 595])

    data2 = np.matrix(np.genfromtxt(x_paths+'test.csv', delimiter=','))
    x_test = np.asarray(data2[1:, 0:595])

    return x_train, y_train, x_test


def preprocessing(x_train, x_test):

    """
    The method performs standard scaling on training and testing data.
    Then, it reduces the dimension of training and testing data by using pca.

    Parameters
    ----------
    x_train: features of training data
    x_test: features of testing data

    """

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    pca = PCA(n_components=2)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test


def train_model(x_train, y_train):

    """
    The method creates a logistic regression classifier, and trains it with training data.

    Parameters
    ----------
    x_train: features of training data
    y_train: labels of training data

    """

    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, np.ravel(y_train, order='C'))
    return classifier


def predict(x_test, model):

    """
    The method predicts labels for testing data by using model object.

    Parameters
    ----------
    x_test: features of testing data
    model: trained learning model

    """

    y_pred = model.predict(x_test)
    return y_pred


def write_output(y_pred):

    ID = 1
    lines = [["ID", "Predicted"]]

    for i in y_pred:
        # Reobtaining the ID is simple since the samples remain in order
        temp = [ID, int(i)]
        ID += 1
        lines.append(temp)

    # Write the output in a file
    with open('submission.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)
    writeFile.close()

# ********** MAIN PROGRAM ********** #


Data = load_data("")
x_train, y_train, x_test = load_data("")
x_train_pca, x_test_pca = preprocessing(x_train, x_test)

model = train_model(x_train_pca, y_train)
predictions = predict(x_test_pca, model)
write_output(predictions)
