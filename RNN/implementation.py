"""Recurrent Neural Network
Predict Google Stock Price on the start of day, based on previous 60 days stock price using RNN
 <t0>.....<t59> 
 Run a sliding window of batch 60 readings over the sample data with step of 1.
 i.e. 1st example would be <t0>-<t59> and <y> = <x60>
 2nd example will be <t1>-<t60>  <y> = <x61>
 Also we could have more than 1 sequence parameter in RNN i.e. not just opening day stock prices 
 of previous 60 days, but some other parameters such as overall sensex or other indices
"""
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
    
#Reshaping. Imagine we have more than 1 parameters in X such as some other indices
# Lets go with single parameter to being with, if we have more than 1 then the number of indicators

# number of examples, number of timesteps, number of indicators 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#Importing Keras libraries and Packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()
