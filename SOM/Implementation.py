import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Fraud Detection
dataset = pd.read_csv('Credit_Card_Applications.csv')
# archive.ics.uci.edu
# Australian credit card approval Datset
# All Column attributes except customer id are made meaningless to protect data privacy

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

