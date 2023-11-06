# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:53:48 2023

@author: Aymen
"""

import shap
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


columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

X_kdd = traindata.iloc[:,1:42]
Y_kdd = traindata.iloc[:,0]
C_kdd = testdata.iloc[:,0]
T_kdd = testdata.iloc[:,1:42]

X_kdd.columns = columns
T_kdd.columns = columns

# Load metatlearner model
model = load_model('models/modelF.h5')
  

# Use the training data for deep explainer => can use fewer instances
explainer = shap.TreeExplainer(model) 
shap_values = explainer.shap_values(trainX,check_additivity=False) 
 
# init the JS visualization code

# Local explanation 
shap.initjs()
i=24199
shap.force_plot(explainer.expected_value[0], shap_values[0][i], trainX.loc[[i]], feature_names = trainX.columns)

shap.initjs()
i=5416
shap.force_plot(explainer.expected_value[2], shap_values[2][i], trainX.loc[[i]], feature_names = trainX.columns)

shap.initjs()
i=16982
shap.force_plot(explainer.expected_value[1], shap_values[1][i], trainX.loc[[i]], feature_names = trainX.columns)

shap.initjs()
i=123808
shap.force_plot(explainer.expected_value[4], shap_values[4][i], trainX.loc[[i]], feature_names = trainX.columns)


# Global explanation 
shap.summary_plot(shap_values, trainX.values, plot_type="bar", class_names= class_names, feature_names = trainX.columns)

shap.summary_plot(shap_values[0], multi_train_X)

shap.summary_plot(shap_values[1], multi_train_X)

shap.summary_plot(shap_values[2], multi_train_X)

shap.summary_plot(shap_values[3], multi_train_X)