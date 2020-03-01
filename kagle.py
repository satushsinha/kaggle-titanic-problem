# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:14:15 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('train_data.csv')
dataset_test =pd.read_csv('test_data.csv')



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset_train.iloc[:,[1,2,5,6,7,9]])
dataset_train.iloc[:,[1,2,5,6,7,9]] = imputer.transform(dataset_train.iloc[:,[1,2,5,6,7,9]])

dataset_test=dataset_test.iloc[:,[1,3,4,5,6,8,10]]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset_test.iloc[:,[0,2,3,4,5]])
dataset_test.iloc[:,[0,2,3,4,5]] = imputer.transform(dataset_test.iloc[:,[0,2,3,4,5]])


dataset_train=dataset_train.iloc[:,[1,2,4,5,6,7,9,11]]


dataset_train.dropna(inplace = True)

y_train = dataset_train.iloc[:, 0].values
x_train = dataset_train.iloc[ :, 1:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x_train[:, 1] = labelencoder_X.fit_transform(x_train[:, 1])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X2 = LabelEncoder()
dataset_test.iloc[:, 1] = labelencoder_X2.fit_transform(dataset_test.iloc[:, 1])

x_train=pd.DataFrame(x_train)

labelencoder_y = LabelEncoder()
x_train.iloc[:, 6] = labelencoder_y.fit_transform(x_train.iloc[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
x_train = onehotencoder.fit_transform(x_train).toarray()


labelencoder_y2 = LabelEncoder()
dataset_test.iloc[:, 6] = labelencoder_y2.fit_transform(dataset_test.iloc[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
dataset_test = onehotencoder.fit_transform(dataset_test).toarray()
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.25, random_state = 0)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
dataset_test= sc.transform(dataset_test)
'''
from xgboost import XGBClassifier
classifier = XGBClassifier(booster='gbtree', eta=0.3,min_child_weight=2.8,max_depth=9,gamma=0, subsample=1)
classifier.fit(X_train, y_train)
'''
'''
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)
'''


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =100, criterion = 'entropy', max_depth=50, min_samples_split=3,min_samples_leaf=1,random_state=0)
classifier.fit(x_train, y_train)

'''
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
# Sequential Forward Selection(sfs)
sfs = SFS(RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0),
           k_features=8,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)

sfs.fit(X_train, y_train)
z=sfs.k_feature_names_
'''

y_pred = classifier.predict(dataset_test)

'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
