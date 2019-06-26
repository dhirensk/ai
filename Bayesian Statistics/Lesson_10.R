#Question 1
b0 = 1.5
b1 = -0.3
b2 = 1.0

x1 = 0.8
x2 = 1.2

llam_hat = 1.5 -0.3* 0.8 + 1.0* 1.2
lam_hat = exp(llam_hat)
lam_hat
# 11.7

library("COUNT")
data("badhealth")

library("rjags")

mod_string = "model {
  for ( i in 1: length(numvisit)){
    numvisit[i] ~ dpois(lam[i])
    log(lam[i]) = int + b_badh * badh[i] + b_age* age[i] + b_intx*age[i]*badh[i]
  }
  
  int ~ dnorm(0.0, 1/1e6)
  b_badh ~ dnorm(0.0, 1/1e4)
  b_age ~ dnorm(0.0, 1/1e4)
  b_intx ~ dnorm(0.0, 1/1e4)
}"

set.seed(102)



data_jags = as.list(badhealth)
str(data_jags)
mod = jags.model(textConnection(mod_string), data = data_jags, n.chains = 3)

update(mod, n.iter = 1e3)
parameters = c("int","b_badh", "b_age", "b_intx")
mod_sim  = coda.samples(mod, variable.names = parameters, n.iter = 5e3)
summary(mod_sim)
mod_csim = as.mcmc(do.call(rbind, mod_sim))
#Compute DIC Deviance Information Criterion

dic = dic.samples(mod, 1e3)
dic
# 5633
# removing interaction term and using simpler additive model


mod1_string = "model {
  for ( i in 1: length(numvisit)){
    numvisit[i] ~ dpois(lam[i])
    log(lam[i]) = int + b_badh * badh[i] + b_age* age[i] 
  }
  
  int ~ dnorm(0.0, 1/1e6)
  b_badh ~ dnorm(0.0, 1/1e4)
  b_age ~ dnorm(0.0, 1/1e4)
}"

set.seed(102)



data_jags1 = as.list(badhealth)
mod1 = jags.model(textConnection(mod1_string), data = data_jags1, n.chains = 3)

update(mod1, n.iter = 1e3)
parameters1 = c("int","b_badh", "b_age")
mod1_sim  = coda.samples(mod1, variable.names = parameters1, n.iter = 5e3)
summary(mod1_sim)
mod1_csim = as.mcmc(do.call(rbind, mod1_sim))
#Compute DIC Deviance Information Criterion

dic1 = dic.samples(mod1, 1e3)
dic1
# 5639


# Question 4
#Suppose that a retail store receives an average of 15 customer calls per hour, 
#and that the calls approximately follow a Poisson process. If we monitor calls for two hours, 
#what is the probability that there will be fewer than 22 calls in this time period? Round your answer to two decimal places

lam_call = 15  # per hour
# for 2 hrs
lam_call_2hrs = 2* 15 

P_22 = ppois(21, 30)
# 0.05
curve(dpois(x,lambda = 30),from = 0, to = 100)

#Question 5

dat = read.csv(file="callers.csv", header=TRUE)
head(dat)

par(mfrow=c(2,2))

boxplot(calls ~ isgroup2, data = dat)
boxplot(calls/days_active ~ isgroup2, data = dat)

#Poisson Model

mod2_string = "model {
   for(i in 1: length(calls)){
     calls[i] ~ dpois(days_active[i] * lam[i])
     log(lam[i]) = b0 + b[1]*age[i] + b[2]* isgroup2[i]
   }
   b0 ~ dnorm(0.0, 1.0/100)
   b[1] ~ dnorm(0.0, 1.0/100)
   b[2] ~ dnorm(0.0, 1.0/100)
}"

data2_jags = as.list(dat)
mod2 = jags.model(textConnection(mod2_string), data = data2_jags, n.chains = 3)

update(mod2, n.iter = 1e3)
parameters2 = c("b0","b")

mod2_sim = coda.samples(mod2, variable.names = parameters2,n.iter = 10e3 )

# Convergence Diagnoistic
gelman.diag(mod2_sim)
plot(mod2_sim)
autocorr.diag(mod2_sim)
autocorr.plot(mod2_sim,lag.max = 500)
effectiveSize(mod2_sim)
summary(mod2_sim)

mod2_csim = as.mcmc(do.call(rbind, mod2_sim))
raftery.diag(mod2_csim)


# calculating residuals

#posterior mean of parameters
mod2_params = colMeans(mod2_csim)

#loglambda
llam_hat = mod2_params["b0"] +  mod2_params["b[1]"] * dat$age + mod2_params["b[2]"] * dat$isgroup2

#MCMC lambda per unit time

lam_hat = exp(llam_hat)

# for active days

# calls[i] ~ dpois(days_active[i] * lam[i])
len = length(dat$calls)
calls_hat = rpois(n = len,lambda = lam_hat * dat$days_active)
resid = dat$calls - calls_hat 
plot(resid)

#What is the posterior probability that b[2] , the coefficient for the indicator isgroup2 is greater than 0? 
head(mod2_csim)

Pb2 = mean(mod2_csim[,"b[2]"]>0)

# 1