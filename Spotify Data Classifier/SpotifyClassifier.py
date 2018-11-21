# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:57:49 2018

@author: sunka
"""

import pandas as pd
from sklearn import tree

# To classify data, we are using the following steps:

# 1. Import Dataset
# 2. EDA to visualize data and observe structure
# 3. Train Data(Decision Tree Classifier)


data = pd.read_csv('data.csv')
print(data)
description = data.describe()
data.info()


data.isnull().any()

# In data 'target' feature represents whether song is best or worst, If song is best then target is put 1 else target is put 0

features = [ 'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence'] # We are colecting only required features for defining training and testing

x = data[features]
y = data["target"]

from sklearn import model_selection

train_data,test_data,train_target,test_target = model_selection.train_test_split(x,y,test_size=0.15)

# Here data will be divided into two parts, tarin part and test part.
# Test part will be of size 15% of original data size

train_data.shape
test_data.shape


data.columns
# Let us first plot the histogram for tempo based on target(as said in above comments)
# Similarly like tempo, also plot for remaining all features based on target
# Now we draw histogram for tempo if target is 1 (liked song) and also we draw histogram for tempo if target is 0 (disliked song)
pos_tempo = data[data['target'] == 1]['tempo']
neg_tempo = data[data['target'] == 0]['tempo']

pos_acousticness = data[data['target'] == 1]['acousticness']
neg_acousticness = data[data['target'] == 0]['acousticness']

pos_danceability = data[data['target'] == 1]['danceability']
neg_danceability = data[data['target'] == 0]['danceability']


pos_duration_ms = data[data['target'] == 1]['duration_ms']
neg_duration_ms = data[data['target'] == 0]['duration_ms']

pos_energy = data[data['target'] == 1]['energy']
neg_energy = data[data['target'] == 0]['energy']

pos_key = data[data['target'] == 1]['key']
neg_key = data[data['target'] == 0]['key']

pos_instrumentalness = data[data['target'] == 1]['instrumentalness']
neg_instrumentalness = data[data['target'] == 0]['instrumentalness']

pos_liveness = data[data['target'] == 1]['liveness']
neg_liveness = data[data['target'] == 0]['liveness']

pos_mode = data[data['target'] == 1]['mode']
neg_mode = data[data['target'] == 0]['mode']

pos_loudness = data[data['target'] == 1]['loudness']
neg_loudness = data[data['target'] == 0]['loudness']

pos_speechiness = data[data['target'] == 1]['speechiness']
neg_speechiness = data[data['target'] == 0]['speechiness']

pos_time_signature = data[data['target'] == 1]['time_signature']
neg_time_signature = data[data['target'] == 0]['time_signature']

pos_valence = data[data['target'] == 1]['valence']
neg_valence = data[data['target'] == 0]['valence']



from matplotlib import pyplot


pyplot.hist(pos_tempo)
pyplot.hist(neg_tempo, color='orange')
pyplot.title("Tempo")
pyplot.hist(pos_acousticness)
pyplot.hist(neg_acousticness, color='orange')
pyplot.title("Acousticness")
pyplot.hist(pos_danceability)
pyplot.hist(neg_danceability, color='orange')
pyplot.title("Dancebility")
pyplot.hist(pos_duration_ms)
pyplot.hist(neg_duration_ms, color='orange')
pyplot.title("Duration")
pyplot.hist(pos_energy)
pyplot.hist(neg_energy, color='orange')
pyplot.title("Energy")
pyplot.hist(pos_key)
pyplot.hist(neg_key, color='orange')
pyplot.title("key")
pyplot.hist(pos_instrumentalness)
pyplot.hist(neg_instrumentalness, color='orange')
pyplot.title("Instrumentalness")
pyplot.hist(pos_liveness)
pyplot.hist(neg_liveness, color='orange')
pyplot.title("Liveness")
pyplot.hist(pos_mode)
pyplot.hist(neg_mode, color='orange')
pyplot.title("Mode")
pyplot.hist(pos_loudness)
pyplot.hist(neg_loudness, color='orange')
pyplot.title("Loudness")
pyplot.hist(pos_speechiness)
pyplot.hist(neg_speechiness, color='orange')
pyplot.title("Speechiness")
pyplot.hist(pos_time_signature)
pyplot.hist(neg_time_signature, color='orange')
pyplot.title("Time Signature")
pyplot.hist(pos_valence)
pyplot.hist(neg_valence, color='orange')
pyplot.title("Valence")


# We will be using decision tree calssifier to train our training set and then to test our test data

trees  = tree.DecisionTreeClassifier(min_samples_split = 200)
fitting = trees.fit(train_data,train_target)
result = trees.predict(test_data)
pyplot.hist(result, color = 'orange')
pyplot.hist(test_target)

import seaborn as sns

print(result)

from sklearn import metrics

varience = metrics.r2_score(result,test_target)
print(varience)
mean_square_error = metrics.mean_squared_error(result,test_target)
print(mean_square_error)
accu = metrics.accuracy_score(result,test_target)
print("Accuracy Score is "+str(round(accu*100))+" %")
















