# Boltsmann machine
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing the dataset

movies = pd.read_csv('ml-1m/movies.dat',sep='::',header = None, engine = 'python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat',sep='::',header = None, engine = 'python', encoding='latin-1')
# Column3 is age of user
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::',header = None, engine = 'python', encoding='latin-1')
# user, movie id, ratings, timestamp..do not care

#Preparing the training and test set
# ml -100k u1base and u1test for example is a subset of base and test data for k-fold cross validation
# sep and delimiter are alais of each other, if delimiter is not provided then it defaults to sep
training_set  = pd.read_csv('ml-100k/u1.base',sep = '\t')  #80k
# user, movie id, ratings
training_set = np.array(training_set, dtype = 'int')  # 20k


test_set  = pd.read_csv('ml-100k/u1.test',sep = '\t')
# user, movie id, ratings   80K
test_set = np.array(test_set, dtype = 'int')

# getting the number of users and movies required to create a matrix
# total number of users = max or user ids from both the test and training set
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Coverting the data into an array with users in lines and movies in the columns and rating as the cell value
# this is required by RBM, movies will be our features

# we create a list of lists and not a 2-D numpy array as you might have thought
def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1):
      #  id_movies = data[data[:,0]==id_users,1]
        id_movies = data[:,1][data[:,0]==id_users]   # ids of movies for a user. a user has not watched all 1682 movies
        id_ratings = data[:,2][data[:,0]==id_users]  # ratings of movies for a user. A userh as not rated all watched movies.
        ratings = np.zeros(nb_movies)        # create a array of 1682 movies
        ratings[id_movies -1] =  id_ratings # get rating for the movie rated by user else 0
            
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)
        

#converting the data into Torch Sensors

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
#convert no ratings i.e 0 to -1
training_set[training_set == 0] = -1
# convert rating 1,2 = 0
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
# convert rating 1,2 = 0
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)  # 1 is for batch column as the tensor works with batch 
        self.b = torch.randn(1, nv)
    def sample_h(self, x):           # visible neurons V in the probability p_h_given_V
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

























