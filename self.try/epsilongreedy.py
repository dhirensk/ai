# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:49:19 2019

@author: dk186039
"""

import numpy as np

#preparing simulation test set for n rows
#number of bandits  x
#pd is the initial probability distribution we have for 10000 rows
N = 10000
n = 10
pd = [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.10, 0.11]  # totals 1
rewards = np.array([0]* n)
totaltries = np.array([0]*n)
epsilons = [0.01, 0.05, 0.1]

X = np.zeros((N,n))
# getting matrix of X
for i in range(N):
    for j in range(n):
        if np.random.random() <= pd[j]:
            X[i,j] = 1

          
class bandit():
    def __init__(self):
        self.mean = 0
        self.reward = 0;
    def pull(self, m, x):
        if X[m,x] == 1:
           self.reward = 1
        else:
            self.reward = 0
        self.update(m,self.reward)
        return self.reward
    def update(self, m, reward):
        # m is 1 based index otherwise we get divide by 0
        m = m+1
        self.mean = ( 1 - 1/m)* self.mean + 1/m*reward
#bandit_groups = [] 
#for eps in range(len(epsilons)):
#bandit_group = [bandit()]* n
bandit_group = [bandit(), bandit(), bandit(), bandit(), bandit(), bandit(), bandit(), bandit(), bandit(), bandit()]
for m in range(N):
    x = 0
    choose = np.random.random()
    if choose <= 0.1 or m < 2000 :
        # we do exploration
        x =  np.random.choice(n)
    else:
        # we do exploitation
        # get the max mean for each bandit
        x = np.argmax( [ b.mean for b in bandit_group])
       # print(x)
    totaltries[x] = totaltries[x] +1
    rewards[x] = rewards[x] + bandit_group[x].pull(m,x)
    #  bandit_groups.append(bandit_group)
    print("m = ", m, "x = ",x ,end=" ")
    for j in range(n):
       print("{:.4f}".format(bandit_group[j].mean), end=" ")
    print("")
   #print("n:",m,bandit_group[0].mean,bandit_group[1].mean, bandit_group[2].mean, bandit_group[3].mean  )
print(rewards)
print(totaltries)
#print(np.argmax(rewards/totaltries))
#print(np.max(rewards/totaltries))
#print("")
#for i in range(n):
   # print(bandit_group[i].mean)

# initialize 10 bandit objects for 10 slot machines
# each have 0 mean to begin with
# we have a simulation dataset based on some prior probabilities
# using epsilon greedy, we need to find out the right combination of exploration:exploitation        
   

      
# epsilon greedy 
# pull a random number, if < eps then do exploration, i.e. pick a             