library("rjags")
dat = read.csv("callers.csv", header = TRUE)

#Poisson model
# Re-fit the model and use your posterior samples to simulate predictions of the number of calls 
# by a new 29 year old customer from Group 2 whose account is active for 30 days. 
# What is the probability that this new customer calls at least three times during this period? Round your answer to two decimal places.

head(dat)
# calls days_active isgroup2 age

#calls/days_active is better indication for lambda  # so calls ~ lambda * days_active

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

mod2_csim = as.mcmc(do.call(rbind, mod2_sim))

n_sim = nrow(mod2_csim)
llam = mod2_csim[,"b0"] + mod2_csim[,"b[1]"] * 29 + mod2_csim[,"b[2]"] * 1
lam = exp(llam)

calls = rpois(n_sim, lam * 30)

P_calls_3 = mean(calls>=3)
# 0.23