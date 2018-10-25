# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:31:04 2018

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

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# load and split the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
pipe.fit(X_train, y_train).score(X_test, y_test)

from sklearn.model_selection import GridSearchCV
param_grid = {'svm__C' : [0.001, .01, .1, 1, 10, 100],
              'svm__gamma' : [.001, .01, .1, 1, 10, 100]}
# parameters of the estimators in the pipeline shoud be defined using
# the **estimator__parameter** syntax

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: {:.2f}'.format(grid.best_score_))
print('Test set_score : {:.2f}'.format(grid.score(X_test, y_test)))
print('Best parameters : {}'.format(grid.best_params_))

# convenient pipeline creation with make_pipeline
from sklearn.pipeline import make_pipeline

# standard syntax
pipe_line = Pipeline([('scaler', MinMaxScaler()),
                      ('svm', SVC(C=100))])

# abbreviated syntax
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
print('Pipeline stpes:\n{}'.format(pipe_short.steps))
# make_pipeline does not require, and does not permit, naming the estimators, 
# Indtead, their names will be set to the lowercase of their automatically

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=2),
                     StandardScaler())

print('Pipeline steps:\n{}'.format(pipe.steps))

# Accessing step attributes
# fit the pipeline defined before to the dataset
pipe.fit(cancer.data)

# extract the first two principal components from the 'pca'step
components = pipe.named_steps['pca'].components_
print('components.shape: {}'.format(components.shape))

# Accessing attributes in a pipeline inside GridSearchCV
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C' : [.001, .1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print('Best estimator:\n{}'.format(grid.best_estimator_))

























