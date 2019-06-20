import pandas as pd
import io
import requests
import numpy as np
import os

import tensorflow.contrib.learn as skflow
import string

from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn import preprocessing


path = "NSL-KDD.csv"
# This file is a CSV, just no CSV extension or headers
df = pd.read_csv(path, header=None)


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