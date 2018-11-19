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
import matplotlib.pyplot as plt
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

length,_ = training_set.shape
for i in range(60, length):
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

#Initializing the LSTM and Dropout
# number of hidden layers in LSTM = n_a
n_a = 50
regressor.add(LSTM(n_a, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(n_a, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(n_a, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(n_a))
regressor.add(Dropout(rate=0.2))

regressor.add(Dense(1,activation =None))

# Compiling and Fitting the data

regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(x=X_train,y=Y_train,batch_size = 32,epochs=100 )

# testing the model on test set

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#Gettingt the predicted stock price
#Important 
"""
The test contains Jan2017 data, inorder to generate previous 60 timesteps for the test data of Jan2017
we need to get previous 60 records from the Training set and concatinate it with the test set.
But we should not make use of the normalized training set for concatenation as the training set
was fit-transformed using this set. Instead we should concatenate the dataframe objects and then 
normalize them to generate our test set and then transform it
"""
# lets merge the training and test set
#only interested in the Open Column index 1
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis=0)
dataset_total.shape[0]
#notice dataset_total is of shape (1278,)
#len(dataset_total)
#dataset_total.size
#dataset_total.shape[0]
#get entire test set + last 60 records of training data

inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) -60 : ].values


#Reshape the input with 1 column
inputs = inputs.reshape(-1,1)

#tranform the input by fitting it to the standard scalar object 
inputs = sc.transform(inputs)
#len(inputs) = 60
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i,-1])
     
X_test = np.array(X_test)

#Reshape the X_test in keras input format  size,tx,n_x
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


# predicting on test data
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the Results

plt.plot(real_stock_price, color='blue', label='real Google stock price ')
plt.plot(predicted_stock_price, color='red', label='real stock price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(np.arange(20),np.arange(1,X_test.shape[0]))
plt.legend()
plt.show()