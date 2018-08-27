# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:11:00 2018

@author: Daniel
"""

from os import getcwd, chdir
getcwd()
chdir('C:/Users/Daniel/scikit_learn')

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
# %matplotlib inline

# scikit-learn commonly used classes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# model ensemble
# voting ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# data loading and split
from sklearn.datasets import load_digits
digits = load_digits()
y = digits.target == 9

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        digits.data, y, random_state=0)

logreg = LogisticRegression()
tree = DecisionTreeClassifier()
mlp = MLPClassifier()
voting = VotingClassifier(
        estimators = [('logreg', logreg), ('tree', tree), ('mlp', mlp)],
        voting = 'hard')

from sklearn.metrics import accuracy_score
for clf in (logreg, tree, mlp, voting) :
    clf.fit(X_train, y_train)
    print(clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test)))

# averaging predictions
averaging = VotingClassifier(
        estimators = [('logreg', logreg), ('tree', tree), ('mlp', mlp)],
        voting = 'soft')

averaging.fit(X_train, y_train)
ave_pred = averaging.predict(X_test)
accuracy_score(y_test, ave_pred)

# stacking
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

rf = RandomForestClassifier()
tree = DecisionTreeClassifier()
mlp = MLPClassifier()
logreg = LogisticRegression()
stacking = StackingClassifier(classifiers=[rf, tree, mlp],
                           meta_classifier = logreg, 
                           use_probas=False,
                           average_probas=False)

for clf in (rf, tree, mlp, stacking) :
    clf.fit(X_train, y_train)
    print(clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test)))


# scale transformations
 
# the effect of preprocessing on supervised learning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)

from sklearn.svm import SVC
svm = SVC(C=100)
svm.fit(X_train, y_train).score(X_test, y_test)

# preprocessing using 0 - 1 scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)
svm.fit(X_train_scaled, y_train).score(X_test_scaled, y_test)

# preprocessing using zero mean and unit variance scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
svm.fit(X_train_scaled, y_train).score(X_test_scaled, y_test)

# feature selection
# model based feature selection

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(), threshold=None)

X_train_fs = select.fit(X_train, y_train).transform(X_train)
print('X_train.shape : {}, X_train_fs.shale: {}'.format(X_train.shape, X_train_fs.shape))

mask = select.get_support() ## get_support?? 무슨의미??
plt.matshow(mask.reshape(1, -1), cmap='gray_r')

X_test_fs = select.transform(X_test)
svm.fit(X_train_fs, y_train).score(X_test_fs, y_test)


































