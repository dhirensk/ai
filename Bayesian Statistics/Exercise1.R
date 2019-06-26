#success rate Theta ~ Beta(5,3)
# odds of success = theta/1-theta
#posterior mean E(theta/1-theta) = 1/m * sum ( theta*/1-theta*)



# stimulate m = 10000 samples from the Beta distribution

m = 10000
theta <- rbeta(n = m, shape1 = 5.0, shape2 = 3.0)

# Simulate odds of success 

odds <- theta/(1- theta)


# Expection i.e. Posterior mean for Odds of success by MC Estimation

E_odds = 1/m* sum(odds)

# approx 2.4607

#Question 6
#Indicator function for P( E_odds > 1)
ind = odds > 1

P_E_odds_greaterthan1 = mean(ind)

#Question 7

#Use a (large) Monte Carlo sample to approximate the 0.3 quantile of the standard normal distribution N(0,1),
# the number such that the probability of being less than it is 0.3.

#Use the quantile function in R. You can of course check your answer using the qnorm function.
m = 10000
theta = rnorm(n = m,mean = 0,sd = 1)
# montecarlo estimate for quantile such that p(q) < 0.3


q = quantile(theta, probs = 0.3)

#using R

qr = qnorm(0.3, mean = 0, sd = 1)


#Question 8

var_theta = 5.2
stderr = sqrt(5.2/5000)
