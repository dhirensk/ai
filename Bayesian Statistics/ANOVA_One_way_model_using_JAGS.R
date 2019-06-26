data("PlantGrowth")
boxplot(weight ~ group, data = PlantGrowth)


#non informative flat prior model

lmod = lm(weight ~ group, data = PlantGrowth)
summary(lmod)

anova(lmod)
#In this case, it gives a p value of 0.016 which is marginally significant.

#Cell means model
mod_cm = lm(weight ~ -1 + group, data=PlantGrowth)
summary(mod_cm)

#Bayesian Model

#Cell means model, each group gets its own mean

## weight[i]  ~iid N(mu[grp[i]], sig2)
## y[i] ~iid N( µ[grp[i]], sig2)
## prec = 1/sig2
library("rjags")
mod_string = "model{
  for(i in 1: length(y)){
    y[i] ~ dnorm(mu[grp[i]], prec)
  }
  
  for(j in 1:3){
    mu[j] ~ dnorm(0.0,1/1e6)
  }
  #Inverse Gamma on sig2 = Gamma on prec
  n_0 = 5.0 # prior effective sample size for sig2
  s2_0 = 1.0 # prior point estimate for sig2
  nu_0 = n_0 / 2.0 # prior parameter for inverse-gamma  shape
  beta_0 = n_0 * s2_0 / 2.0 # prior parameter for inverse-gamma  rate
  prec ~ dgamma(nu_0, beta_0)
  sig = 1/ sqrt(prec)
}"

inits = function(){
  inits = list("mu" = rnorm(n = 3, mean = 0, sd = 100), "prec"= rgamma(n = 1, shape = 1, rate = 1))
}

as.numeric(PlantGrowth$group)
data_jags = list(y = PlantGrowth$weight, grp = as.numeric(PlantGrowth$group))

mod = jags.model(textConnection(mod_string), data = data_jags, inits = inits, n.chains = 3)

update(mod, n.iter = 1000)

parameters = c("mu", "sig")
mod_sim = coda.samples(mod,variable.names = parameters, n.iter = 5000)

plot(mod_sim)
summary(mod_sim)

#Combine Samples

mod_csim = as.mcmc(do.call(rbind,mod_sim))

#Gelman Rubin Diag

gelman.diag(mod_sim)


#Auto-Correlation 
autocorr.diag(mod_sim)


#Effective Size

effectiveSize(mod_sim)


#Posterior Mean

PM_params = colMeans(mod_csim)
PM_params


#Comparing with non-informative linear model

coefficients(lmod)


# Residuals

data_jags$grp

# order means as per the observation groups
yhat = PM_params[1:3][data_jags$grp]

resid = data_jags$y - yhat
plot(resid)

plot(y = resid, x = yhat)


#Model Summary
summary(mod_sim)


#High Posterior Density Interval
HPDinterval(mod_csim)

#HPD 0.9
HPDinterval(mod_csim, prob = 0.9)

# Posterior Probability of treatment2 > Control group

mean(mod_csim[,3] > mod_csim[,1])


# Posterior Probability that treatment2 will increase yield by 10%

mean(mod_csim[,3] > 1.1 *mod_csim[,1])


#DIC model1
dic.samples(mod, n.iter = 5e3)
# Mean deviance:  58.96 
# penalty 4.069 
# Penalized deviance: 63.03 


#Use the original model (single variance) to calculate a 95% interval of highest posterior density (HPD) for µ3 - µ1
mod_csim2 = mod_csim[,3] - mod_csim[,1]
HPDinterval(mod_csim2)
#lower    upper
#var1 -0.1502806 1.128454