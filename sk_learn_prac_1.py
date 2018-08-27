# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 21:11:30 2018

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

df = pd.read_csv('./data/M3T1_data_pepTestCustomers.csv')
new = pd.read_csv('./data/M3T1_data_pepNewCustomers.csv')

df.head()
new.head()
df.shape
new.shape

df.info()
df.head()
df.describe()
df.hist(bins=30, figsize=(20,15))

df.corr()
plt.matshow(df.corr())
df.corr().pep.sort_values(ascending=False)

from pandas.plotting import scatter_matrix
df.head()
scatter_matrix(df.iloc[:, [1,4]], c=df['pep'], figsize=(10,10), marker='o', 
               s=50, diagnal='kde')

df.loc[:, ['age', 'income']].plot.box(subplots=True, layout=(2, 1), figsize=(10, 10))

# construct data
# derive attributes

mdf = df.copy()
mdf['realincome'] = np.where(mdf['children'] == 0, mdf['income'], mdf['income']/mdf['children'])

# select data
# filter attributes

mdf = mdf.drop(['income', 'children'], axis=1)
mdf.head()

dfX = mdf.drop(['id', 'pep'], axis=1)
dfy = mdf['pep']
X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=.25, 
                                                    random_state=0)

print(X_train.shape, X_test.shape)
X_train.head()

#  Modeling

# decison tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=6, random_state=0)

tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)
pred_tree

# Linear Support Vector Classification (Linear SVM)
from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=0)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
pred_svc

# Neunal Networks (Multi-layer Perceptron)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5,2), random_state=1)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)
pred_mlp

# Assess Model

# decision tree
tree.score(X_train, y_train)
tree.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred_tree)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_tree, target_names=['not buy', 'buy']))

# Linear Support Vector Classification (Linear SVM)
confusion_matrix(y_test, pred_svc)
print(classification_report(y_test, pred_svc, target_names=['not buy', 'buy']))

# Neunal Networks (Multi-layer Perceptron)
confusion_matrix(y_test, pred_mlp)
print(classification_report(y_test, pred_mlp, target_names=['not buy', 'buy']))

# Evaluation
best_model = tree
best_model.score(X_test, y_test)

from sklearn.dummy import DummyClassifier
print(y_test.value_counts())
DummyClassifier(strategy='most_frequent').fit(X_train, y_train).score(X_test, y_test)

# Deployment
ndf = new.copy()
ndf['realincome'] = np.where(ndf['children']==0, ndf['income'], ndf['income'] / ndf['children'])
ndf = ndf.drop(['income', 'children'], axis=1)
ndf.head()

ndf['pred'] = best_model.predict(ndf.loc[:, 'age':'realincome'])
ndf['pred_prob'] = best_model.predict_proba(ndf.loc[:, 'age':'realincome'])[:, 1]
ndf.head()

target = ndf.query('pred==1 & pred_prob > .7')
target.shape
target.sort_values(by='pred_prob', ascending=False).to_csv('target.csv', index=False)
pd.read_csv('target.csv').tail()

# export the best model for future use in other programs or systems
from sklearn.externals import joblib

joblib.dump(best_model, 'pep_model.sav')

loaded_model = joblib.load('pep_model.sav')
loaded_model.score(X_test, y_test)

# confusion matrix
from sklearn.datasets import load_digits
digits = load_digits()
y = digits.target == 9
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

# training models
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print('Dummy model:')
print(accuracy_score(y_test, pred_dummy))
print('Decision tree:')
print(accuracy_score(y_test, pred_tree))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print('Dummy model:')
print(confusion_matrix(y_test, pred_dummy))
print('Decision tree : ')
print(confusion_matrix(y_test, pred_tree))

# Classification Report
from sklearn.metrics import classification_report
print('Dummy model:')
print(classification_report(y_test, pred_dummy, target_names=['not 9', '9']))
print('Decision tree : ')
print(classification_report(y_test, pred_tree, target_names=['not 9', '9']))

# ROC & AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pylab as plt

def plot_roc_curve(fpr, tpr, model, color=None) :
    model = model + ' (auc = %0.3f)' %auc(fpr, tpr)
    plt.plot(fpr, tpr, label = model, color=color)
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR (1-specificity)')
    plt.ylabel('TPR (recall)')
    plt.title('ROC curve')
    plt.legend(loc='lower right')


fpr_dummy, tpr_dummy, _ = roc_curve(y_test, dummy.predict_proba(X_test)[:, 1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy model', 'hotpink')

fpr_tree, tpr_tree, _ = roc_curve(y_test, tree.predict_proba(X_test)[:, 1])
plot_roc_curve(fpr_tree, tpr_tree, 'decision tree', 'darkgreen')

# cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree, X_train, y_train)
scores

scores = cross_val_score(tree, X_train, y_train, cv=5)
scores

cross_val_score(tree, X_train, y_train, cv=5, scoring='roc_auc')


# set the parameters for grid search
param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}

param_grid

# Grid search with cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

# Evaluate the model with best parameters

grid_search.score(X_test, y_test)

print('Best parameters :  {}'.format(grid_search.best_params_))
print('Best CV score :  {:.2f}'.format(grid_search.best_score_))
print('Best estimator:\n{}'.format(grid_search.best_estimator_))

# when the parameters are asymmetric
param_grid = [{'kernel' : ['rbf'],
               'C' : [.001, .01, .1, 1, 10, 100],
               'gamma' : [.001, .01, .1, 1, 10, 100]},
    {'kernel' : ['linear'],
     'C' : [.001, .01, .1, 1, 10, 100]}]

param_grid

grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search.score(X_test, y_test)

print('Best parameters :  {}'.format(grid_search.best_params_))
print('Best CV score :  {:.2f}'.format(grid_search.best_score_))
print('Best estimator:\n{}'.format(grid_search.best_estimator_))




