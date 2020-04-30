# Code Owners: Bulut Karabıyık - Cankurt Kostur
# Code Editor: Göktuğ Güvercin

import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

param_grids = [
    {'n_neighbors': np.arange(1, 30, 2),
     },
    {
        'kernel': ['rbf', 'linear'],
        'C': np.arange(0.025, 5, 0.025)},
    {
        'max_depth': np.arange(3, 10)},

    {
        'tol': [1e-4]
    },
    {
        'tol': [1.0e-4]
    }
]

classifiers = [
    KNeighborsClassifier(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

"""
* The list "param_grids" contains dictionary objects.
* Each dictionary can have one or more than one parameter name and corresponding value range.
* The values in that range are tried in cross validation by GridSearch to determine which one is
  the best value for that parameter.

* The list "classifiers" contains learning model objects.
* For each learning model in that list, best parameter set is determined by GridSearch.
* Then, those models and their best parameter sets are used to construct powerful voting classifier
"""
