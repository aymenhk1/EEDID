# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:50:49 2023

@author: Aymen
"""

# Import packages
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
import numpy as np 
import pandas as pd 
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import Sequential
 
# load LSTM models from file
def load_all_models(n_models):
 all_models = list()
 for i in range(n_models):
 # define filename for this ensemble
  filename = 'models/model_' + str(i + 1) + '.h5'
 # load model from file
  model = load_model(filename)
 # add to list of members
  all_models.append(model)
 print('>loaded %s' % filename)
 return all_models
 
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
 stackX = None
 for model in members:
 # make prediction
  yhat = model.predict(inputX, verbose=0)
 # stack predictions into [rows, members, probabilities]
 if stackX is None:
  stackX = yhat
 else:
  stackX = dstack((stackX, yhat))
 # flatten predictions to [rows, members x probabilities]
  stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
 return stackX
 
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
 # create dataset using ensemble
 stackedX = stacked_dataset(members, inputX)
 print(stackedX.shape)
 # fit standalone model
 model = RandomForestClassifier()
 model.fit(stackedX, inputy)
 return model
 
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
 # create dataset using ensemble
 stackedX = stacked_dataset(members, inputX)
 # make a prediction
 yhat = model.predict(stackedX)
 return yhat
 
traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)


X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

print(trainX.shape, testT.shape)
# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# evaluate standalone models on test dataset
for model in members:
 y_test1 = np.array(C)
 y_test= to_categorical(y_test1)
 _, acc = model.evaluate(testT, y_test, verbose=0)
 print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
model = fit_stacked_model(members, testT, y_test1)
# evaluate model on test set
yhat = stacked_prediction(members, model, testT)
acc = accuracy_score(y_test1, yhat)
print('Stacked Test Accuracy: %.3f' % acc)