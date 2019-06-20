import pandas as pd
import io
import requests
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
import tensorflow.contrib.learn as skflow
import string


#from sklearn.utils.multiclass import unique_labels
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve
from inspect import signature
#%matplotlib inline

path = "UNSW-training.csv"
# This file is a CSV, just no CSV extension or headers
df = pd.read_csv(path)
print(df[0:3])

print("Read {} rows.".format(len(df)))
# df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
#df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)


print("Antes de borrar: {}".format(df.shape))
df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)

print("Despues de borrar: {}".format(df.shape))

# # The CSV file has no column heads, so add them
# df.columns = [
#     'id',
#     'dur',
#     'proto',
#     'service',
#     'state',
#     'spkts',
#     'dpkts',
#     'sbytes',
#     'dbytes',
#     'rate',
#     'sttl',
#     'dttl',
#     'sload',
#     'dload', 
#     'sloss',
#     'dloss', 
#     'sintpkt',
#     'dintpkt', 
#     'djit',
#     'sjit',
#     'swin',
#     'stcpb', 
#     'dtcpb',
#     'dwin', 
#     'tcprtt',
#     'synack',
#     'ackdat', 
#     'smean',
#     'dmean', 
#     'trans_depth', 
#     'response', 
#     'ct_srv_src',
#     'ct_state_ttl', 
#     'ct_dst_ltm', 
#     'ct_src_dport_ltm', 
#     'ct_src_sport_ltm', 
#     'ct_dst_src_ltm', 
#     'is_ftp_login', 
#     'ct_ftp_cmd', 
#     'ct_flw_http_mthd', 
#     'ct_src_ltm',
#     'ct_srv_dst',
#     'is_sm_ips_ports',
#     'attack_cat',
#     'Label'
# ]

ENCODING = 'utf-8'

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))
        
def analyze(filename):
    print()
    print("Analyzing: {}".format(filename))
    df = pd.read_csv(filename,encoding=ENCODING)
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

analyze(path)