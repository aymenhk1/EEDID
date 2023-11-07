# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 06:30:26 2023

@author: Aymen
"""

# module imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random

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

# model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print('Welcome LSTM Training!')

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
model.add(LSTM(32,input_dim=38, return_sequences=True))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(LSTM(32,input_dim=38, return_sequences=False))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(5))
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
model.add(LSTM(64,input_dim=38, return_sequences=True))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(LSTM(64,input_dim=38, return_sequences=False))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X_train, y_train, verbose=0, batch_size=batch_size, epochs=100)
# fit and save model
filename = 'models/model_2.h5'
model.save(filename)
print('>Saved %s' % filename)

# LSTM - 3
batch_size = 32
model = Sequential()
model.add(LSTM(128,input_dim=38, return_sequences=True))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(LSTM(128,input_dim=38, return_sequences=False))  # try using a GRU instead, for fun
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X_train, y_train, verbose=0, batch_size=batch_size, epochs=100)
# fit and save model
filename = 'models/model_3.h5'
model.save(filename)
print('>Saved %s' % filename)


