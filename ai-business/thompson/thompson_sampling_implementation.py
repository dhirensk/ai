# alpha and beta parameters = 1 for round 1
#Note that for the special case of alpha = 
beta = 1, the prior p(theta) is uniform over [0; 1]
# alpha and beta cannot be 0.

# consider a prior probabilty distribution [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]

#P(theta) = BetaD(alpha+1, Beta+1)
# strategy calcuate  P(theta) of all bandits 
# action --  perform the action on the bandit which has max P(theta) using argmax
# reward -- calculate the reward, i.e. update the alpha and beta for the selected bandit

# total 9 bandits
# total 10000 rounds
import numpy as np
import matplotlib.pyplot as plt

PriorProbabilities = [0.05,0.13,0.09,0.16,0.11,0.04,0.20,0.08,0.01]
alphas = [1,1,1,1,1,1,1,1,1]
betas = [1,1,1,1,1,1,1,1,1]
N = 10000
x = 9

# Getting a matrix of simulation as per the initial probability distribution
X = np.zeros((N,x))
for i in range(N):
    for j  in range(x):
        if np.random.rand() <= PriorProbabilities[j]:
            X[i,j] = 1
            
            
# Validation of X matrix
#Total rewards 
actual_rewards_simulation = np.sum(X, axis=0)
total_rewards_simulation = np.sum(X)
rewards_prob = actual_rewards_simulation/total_rewards_simulation
rewards_prob = np.round(rewards_prob,2)
# [0.06, 0.15, 0.1 , 0.19, 0.13, 0.05, 0.23, 0.09, 0.01] this is probability distribution of simulation data 
# which is close enough to our prior probability assumption

#lets start with the ts
# regret curve is plot at each interval what is total difference between rewards accumulated by best strategy 
# minus the totat rewards accumulated by thomposon sampling
regret_curve = []
best_rewards = [0]* x
total_rewards_ts = [0]* x
best_rewards_upto_n_rounds = 0
total_rewards_ts_upto_n_rounds = 0
for i in range(N):
    max_beta = 0
    P_x = [0] * x

    for j  in range(x):
        # we need to compare beta distribution for each bandit and get the one which has highest prior
        P_x[j] = np.random.beta( alphas[j], betas[j])
        best_rewards[j]+= X[i,j]
        #for regret curve
    best_rewards_upto_n_rounds = max( best_rewards)    
    argmax = np.argmax(P_x)
    #Selecting the bandit leads to either reward 0 or 1 and it is captured in simulation matrix
    reward = X[i,argmax] 
    if reward == 1:
        alphas[argmax] += 1
        total_rewards_ts_upto_n_rounds += 1

    else:
        betas[argmax] += 1

    regret_curve.append(best_rewards_upto_n_rounds - total_rewards_ts_upto_n_rounds)  

plt.close
plt.plot(regret_curve)
plt.xlabel("rounds")
plt.ylabel('regret')
plt.show()