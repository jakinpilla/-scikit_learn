# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:04:43 2018

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

import copy

data = pd.read_csv('./data/Allstate_train.csv')
data.head()
data.info()

#record_type - 0=shopping point, 1=purchase point
#day - Day of the week (0-6, 0=Monday)
#time - Time of day (HH:MM)
#state - State where shopping point occurred
#location - Location ID where shopping point occurred
#group_size - How many people will be covered under the policy (1, 2, 3 or 4)
#homeowner - Whether the customer owns a home or not (0=no, 1=yes)
#car_age - Age of the customer’s car
#car_value - How valuable was the customer’s car when new
#risk_factor - An ordinal assessment of how risky the customer is (1, 2, 3, 4)
#age_oldest - Age of the oldest person in customer's group
#age_youngest - Age of the youngest person in customer’s group
#married_couple - Does the customer group contain a married couple (0=no, 1=yes)
#C_previous - What the customer formerly had or currently has for product option C (0=nothing, 1, 2, 3,4)
#duration_previous -  how long (in years) the customer was covered by their previous issuer
#A,B,C,D,E,F,G - the coverage options
#cost - cost of the quoted coverage options

dataP = data.loc[data.record_type==1].copy()
dataP.head()
con = ['group_size', 'car_age', 'age_oldest', 'age_youngest', 'duration_previous', 'cost']
cat = ['homeowner', 'car_value',  'risk_factor', 'married_couple', 'C_previous', 
       'state', 'location', 'shopping_pt']

dataP.info()
dataP.isnull().sum() # risk_factor :: 34346, C_previous :: 836, duration_previous :: 836

# drop
dataP_drop = dataP.dropna(subset = ['risk_factor', 'C_previous', 'duration_previous'])
dataP_drop.shape
dataP_drop.isnull().sum()

# impute
dataP[con].dtypes
dataP[con].head()

from sklearn.preprocessing import Imputer
imputer_con = Imputer(strategy = 'median')
imputer_con.fit(dataP[con])

# strategy :: mean, median, most_frequent
X = imputer_con.transform(dataP[con])
dataP_imp = dataP.copy()
dataP_imp[con] = pd.DataFrame(X, columns=dataP[con].columns, index=dataP.index)
dataP_imp.head()

# Categorical feature의 결측값 대체
dataP_imp[cat].dtypes

obj = ['car_value', 'state']
print(dataP['car_value'].astype('category').cat.categories)
print(dataP['state'].astype('category').cat.categories)

# object type의 feature만 추출, category type으로 바꾼 후 숫자로 encoding
dataP_imp[obj] = pd.DataFrame(dataP_imp[obj].apply(lambda x : x.astype('category').cat.codes),
         index=dataP_imp.index)
dataP_imp.dtypes

# strategy = 'most_frequent'을 사용하여 impute
imputer_cat = Imputer(strategy = 'most_frequent')
imputer_cat.fit(dataP_imp[cat])
X = imputer_cat.transform(dataP_imp[cat])
dataP_imp[cat] = pd.DataFrame(X, columns=dataP_imp[cat].columns, index=dataP_imp.index)

dataP_imp[cat].head()

# handling categorical variables
# one-hot encoding
dataP_imp = pd.get_dummies(dataP_imp, columns=['day'])
dataP_imp.head()
dataP_imp.filter(like='day').head()

# label encoding
dataP['car_value'].value_counts()
dataP['car_value'] = dataP['car_value'].astype('category')
dataP['car_value'] = dataP['car_value'].cat.codes
dataP['car_value'].value_counts()

# time 변수의 처리
dataP_imp.head()
dataP_imp.time.head()
time = dataP_imp.time
time = pd.to_datetime(time, format='%H:%M')
time.head()

dataP_imp['hour'] = time.dt.hour
dataP_imp['minute'] = time.dt.minute
dataP_imp[['hour', 'minute', 'time']].head()

# hour minute feature 생성
dataP_imp['hour_group'] = pd.cut(dataP_imp.hour, bins=[0, 6, 12, 18, 24], 
         labels=['0_6', '6_12', '12_18', '18_24'])

dataP_imp[['hour', 'hour_group']]

# one hot으로 hour group 변환
dataP_imp = pd.get_dummies(dataP_imp, columns=['hour_group'])
dataP_imp.filter(like='hour').head()

# feature transformation
dataP_imp.groupby(by='state')['cost'].mean()
# map을 사용, match?? 이용방법을 세부적으로 알아볼 것
dataP_imp['stcost'] = dataP_imp.state.map(dataP_imp.groupby(by='state')['cost'].mean())
dataP_imp[['state', 'cost', 'stcost']].head(50)

# state 별 cost의 평균 계산 후 각 state에 mapping
dataP_imp['ppCost'] = dataP_imp.cost / dataP_imp.group_size
dataP_imp[['state', 'cost', 'stcost', 'ppCost']].head()



















