# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:07:30 2023

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
 
print('Welcome MetaLearner Training!')

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
 
#Load data
df = pd.read_csv("KDDTrain+.txt")
test_df = pd.read_csv("KDDTest+.txt")


# add the column labels
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

df.columns = columns
test_df.columns = columns

# sanity check
df.head()

# lists to hold our attack classifications
dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

# we will use these for plotting below
attack_labels = ['Normal','DoS','Probe','Privilege','Access']

# helper function to pass to data frame mapping
def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type

class_names =[0,1,2,3,4]

# map the data and join to the data set
attack_map = df.attack.apply(map_attack)
df['attack_map'] = attack_map

test_attack_map = test_df.attack.apply(map_attack)
test_df['attack_map'] = test_attack_map

# view the result
df.head()

set(df['attack_map'])

df.head()


df['attack'] = df['attack_map']
test_df['attack'] = test_df['attack_map']

X_kdd = df.drop(['protocol_type',	'service','flag','level','attack_map','attack'],axis=1)
T_kdd = test_df.drop(['protocol_type',	'service','flag','level','attack_map','attack'],axis=1)

# create our target classifications
Y_kdd = df['attack']
C_kdd = test_df['attack']


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

# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# evaluate standalone models on test dataset
for model in members:
 y_test1 = np.array(C_kdd)
 y_test= to_categorical(y_test1)
 _, acc = model.evaluate(testT, y_test, verbose=0)
 print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
model = fit_stacked_model(members, testT, y_test1)
filename = 'models/model_F.h5'
model.save(filename)
print('>Saved %s' % filename)
# evaluate model on test set
yhat = stacked_prediction(members, model, testT)
acc = accuracy_score(y_test1, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
