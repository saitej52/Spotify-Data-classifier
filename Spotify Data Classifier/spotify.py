# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:13:33 2018

@author: sunka
"""

import pandas as pd

data = pd.read_csv('data.csv')

print(data)

data.columns

data = data[['acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence', 'target']]

x = data.iloc[:,:-1]
y = data.iloc[:,-1:]


from sklearn import model_selection

train_data, test_data, train_target, test_target = model_selection.train_test_split(x,y)

from sklearn import tree

classifier = tree.DecisionTreeClassifier()
fitting = classifier.fit(train_data,train_target)
result = classifier.predict(test_data)

import seaborn as sns
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
sns.set_style("whitegrid")
sns.kdeplot(result, ax=ax)
sns.kdeplot(test_data, ax=ax)

from sklearn import metrics

acuuracy = metrics.accuracy_score(result,test_target)