# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:53:48 2023

@author: Aymen
"""

import shap
from keras.models import load_model
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

print('Welcome Model Explaining!')

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
# Load metatlearner model
filename = 'models/model_F.h5'
model = pickle.load(open(filename, 'rb'))



# Use the training data for deep explainer => can use fewer instances
explainer = shap.TreeExplainer(model) 
shap_values = explainer.shap_values(X_kdd,check_additivity=False) 
 
# init the JS visualization code

# Local explanation 
shap.initjs()
i=24199
shap.force_plot(explainer.expected_value[0], shap_values[0][i], X_kdd.loc[[i]], feature_names = X_kdd.columns)

shap.initjs()
i=5416
shap.force_plot(explainer.expected_value[2], shap_values[2][i], X_kdd.loc[[i]], feature_names = X_kdd.columns)

shap.initjs()
i=16982
shap.force_plot(explainer.expected_value[1], shap_values[1][i], X_kdd.loc[[i]], feature_names = X_kdd.columns)

shap.initjs()
i=123808
shap.force_plot(explainer.expected_value[4], shap_values[4][i], X_kdd.loc[[i]], feature_names = X_kdd.columns)


# Global explanation 
shap.summary_plot(shap_values, X_kdd.values, plot_type="bar", class_names= class_names, feature_names = X_kdd.columns)

shap.summary_plot(shap_values[0], X_kdd)

shap.summary_plot(shap_values[1], X_kdd)

shap.summary_plot(shap_values[2], X_kdd)

shap.summary_plot(shap_values[3], X_kdd)
