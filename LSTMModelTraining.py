# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:28:46 2023

@author: Aymen
"""

# Import packages
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.layers import BatchNormalization
from os import makedirs
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
 

# Load data
traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)


X_kdd = traindata.iloc[:,1:42]
Y_kdd = traindata.iloc[:,0]
C_kdd = testdata.iloc[:,0]
T_kdd = testdata.iloc[:,1:42]

# Pre-process data
scaler = Normalizer().fit(X_kdd)
trainX = scaler.transform(X_kdd)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T_kdd)
testT = scaler.transform(T_kdd)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train1 = np.array(Y_kdd)
y_test1 = np.array(C_kdd)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


# reshape data
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


print(trainX.shape, testT.shape)

# create directory 
makedirs('models')



# fit LSTM model on dataset
# LSTM - 1 
batch_size = 32
model = Sequential()
model.add(LSTM(32,input_dim=41, return_sequences=True))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(LSTM(32,input_dim=41, return_sequences=False))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X_train, y_train, verbose=0, batch_size=batch_size, epochs=50)
filename = 'models/model_1.h5'
# fit and save model
model.save(filename)
print('>Saved %s' % filename)
 
# LSTM - 2
batch_size = 32
model = Sequential()
model.add(LSTM(64,input_dim=41, return_sequences=True))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(LSTM(64,input_dim=41, return_sequences=False))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X_train, y_train, verbose=0, batch_size=batch_size, epochs=100)
# fit and save model
filename = 'models/model_1.h5'
model.save(filename)
print('>Saved %s' % filename)

# LSTM - 3
batch_size = 32
model = Sequential()
model.add(LSTM(128,input_dim=41, return_sequences=True))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(LSTM(128,input_dim=41, return_sequences=False))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X_train, y_train, verbose=0, batch_size=batch_size, epochs=100)
# fit and save model
filename = 'models/model_1.h5'
model.save(filename)
print('>Saved %s' % filename)

