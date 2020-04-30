# Code Owner: Mümtaz Cem Eriş - İsmet Ata Yardımcı
# Code Editor: Göktuğ Güvercin

import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

import xgboost as xgb
from sklearn.ensemble import *
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import GridSearchCV

# ************************************************************************************************
# This source file only provides classifiers for other source files. The main program is main.py *
# ************************************************************************************************

seed = 1075
np.random.seed(seed)

# Classifiers
rf = RandomForestClassifier()
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
dt = DecisionTreeClassifier(max_depth=1)

# Bagging Classifiers
bagging_clf = BaggingClassifier(rf, max_samples=0.4, max_features=10, random_state=seed)

# Boosting Classifiers
ada_boost = AdaBoostClassifier()
ada_boost_svc = AdaBoostClassifier(base_estimator=svc, algorithm='SAMME')
grad_boost = GradientBoostingClassifier()
xgb_boost = xgb.XGBClassifier()

# Voting Classifiers
vclf = VotingClassifier(estimators=[('ada_boost', ada_boost), ('grad_boost', grad_boost),
                                    ('xgb_boost', xgb_boost), ('BaggingWithRF', bagging_clf)], voting='hard')

ev_clf = EnsembleVoteClassifier(clfs=[ada_boost_svc, grad_boost, xgb_boost], voting='hard')

# Grid Search
params = {'gradientboostingclassifier__n_estimators': [10, 200],
          'xgbclassifier__n_estimators': [10, 200]}

grid = GridSearchCV(estimator=ev_clf, param_grid=params, cv=5)
