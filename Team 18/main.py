"""
Target Problem:
---------------
* A classifier for the diagnosis of Autism Spectrum Disorder (ASD)

Proposed Solution (Machine Learning Pipeline):
----------------------------------------------
* SelectKBest Algorithm -> PCA -> Variance Thresholding -> Voting Classifier

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
* Copyright © Team 18. All rights reserved.
* Copyright © Istanbul Technical University, Learning From Data Spring 2019. All rights reserved. """

import warnings
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest

from classifiers import *

warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def load_data(x_train_path, x_test_path):

    """
    The method reads train and test data from dataset csv files.
    Then, train data is decomposed into features and labels.
    Finally, the method returns features and labels of train data and test data itself.

    Parameters
    ----------
    x_train_path: directory of training data set file
    x_test_path: directory of testing data set file
    """

    all_data = pd.read_csv(x_train_path)
    x_test = pd.read_csv(x_test_path)

    y_train = all_data["class"]
    x_train = pd.read_csv(x_train_path)
    x_train.drop("class", axis=1, inplace=True)
    return x_train, y_train, x_test


def preprocessing(x_train, y_train, x_test):

    """
    * The method performs 3 dimensionality reduction methods: Variance Threshold - SelectKBest Algorithm - PCA.
    * It at first performs variance threshold, and eliminates all features whose variance values are lower than 0.001.
    * Then, it computes chi square value for each feature, and chooses top 10 features with highest chi square value.
      When doing this, it benefits from SelectKBest algorithm.
    * In final step, the method synthesizes 2 new features by using pca.

    Parameters
    ----------
    x_train: features of training data
    y_train: labels of training data
    x_test: features of testing data
    """

    selector_threshold = VarianceThreshold(0.001)
    selector_threshold.fit(x_train)

    x_train_new = selector_threshold.transform(x_train)
    x_test_new = selector_threshold.transform(x_test)

    selector = SelectKBest(chi2, k=10)
    selector.fit(x_train_new, y_train)

    x_train_new = selector.transform(x_train_new)
    x_test_new = selector.transform(x_test_new)

    pca = PCA(n_components=2, whiten=True)
    pca.fit(x_train_new)

    x_train_pca = pca.transform(x_train_new)
    x_test_pca = pca.transform(x_test_new)
    return x_train_pca, x_test_pca


def train_model(x_train, y_train):

    """
    * The method performs GridSearch operation to choose best parameter set for each classification model.
    * Then, the classification models with best parameter set and their names are stored in two different lists.
    * These two lists are used to combine these best-parametrized classification models in voting classifier.
    * That voting classifier is trained with training data and returned.

    Parameters
    ----------
    x_train: features of training data
    y_train: labels of training data

    """

    best_models = []
    model_names = []

    for i in range(len(classifiers)):

        model = classifiers[i]
        grid_search = GridSearchCV(model, param_grids[i], cv=5)

        grid_search.fit(x_train, y_train)
        best_models.append(grid_search.best_estimator_)

        name = model.__class__.__name__
        model_names.append(name)

    estimators = [('knn', best_models[0]), ('SVC', best_models[1]), ('DT', best_models[2]), ('LA', best_models[3]),
                  ('QA', best_models[4])]

    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(x_train, y_train)
    return ensemble


def predict(model, x_test):
    """
    The method predicts labels for testing data samples by using trained learning model, that is voting classifier.

    Parameters
    ----------
    model: trained learning model
    x_test: features of testing data
    """
    return model.predict(x_test)


def write_output(prediction, file_name):
    ID = np.arange(1, len(prediction) + 1)
    Id_Predict = list(zip(ID, prediction))
    Id_Predict = pd.DataFrame(Id_Predict, columns=['ID', 'Predicted'])
    Id_Predict.to_csv(file_name, index=False)


# ********** MAIN PROGRAM ********** #

x_train, y_train, x_test = load_data("train.csv", "test.csv")
x_train_pca, x_test_pca = preprocessing(x_train, y_train, x_test)

model = train_model(x_train_pca, y_train)
predictions = predict(model, x_test_pca)
write_output(predictions, "submission.csv")