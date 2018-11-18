#Recurrent Neural Network

import numpy as np
import matplotlib as plt
import pandas as pd

#importing the dataset

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values # 1:2 ensures we are creating a matrix instead of array

# feature scaling using normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
Y_train = []

len,_ = training_set.shape
for i in range(60, len):
    X_train.append(training_set_scaled[i-60:i,0]) # its a list
    Y_train.append(training_set_scaled[i,0])  # its a list
    
# convert into numpy array   
X_train, Y_train = np.array(X_train), np.array(Y_train)
    
