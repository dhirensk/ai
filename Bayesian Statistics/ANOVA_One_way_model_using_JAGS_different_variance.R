data("PlantGrowth")
boxplot(weight ~ group, data = PlantGrowth)


#non informative flat prior model

lmod = lm(weight ~ group, data = PlantGrowth)
summary(lmod)

anova(lmod)
#In this case, it gives a p value of 0.016 which is marginally significant.


#Bayesian Model

#Cell means model, each group gets its own mean

## weight[i]  ~iid N(mu[grp[i]], sig2[grp[i]])
## y[i] ~iid N( µ[grp[i]], sig2)
## prec = 1/sig2
library("rjags")
mod2_string = "model{
  for(i in 1: length(y)){
    y[i] ~ dnorm(mu[grp[i]], prec[grp[i]])
  }
  
  for(j in 1:3){
    mu[j] ~ dnorm(0.0,1/1e6)
  }
   #Inverse Gamma on sig2 = Gamma on prec
  n_0 = 5.0 # prior effective sample size for sig2
  s2_0 = 1.0 # prior point estimate for sig2
  nu_0 = n_0 / 2.0 # prior parameter for inverse-gamma  shape
  beta_0 = n_0 * s2_0 / 2.0 # prior parameter for inverse-gamma  rate
  for(k in 1:3){
    prec[k] ~ dgamma(nu_0, beta_0)
  }

  sig = 1/ sqrt(prec)
}"

inits2 = function(){
  inits = list("mu" = rnorm(n = 3, mean = 0, sd = 100), "prec"= rgamma(n = 3, shape = 1, rate = 1))
}

as.numeric(PlantGrowth$group)
data_jags2 = list(y = PlantGrowth$weight, grp = as.numeric(PlantGrowth$group))

mod2 = jags.model(textConnection(mod2_string), data = data_jags2, inits = inits2, n.chains = 3)

update(mod2, n.iter = 1000)

parameters2 = c("mu", "sig")
mod2_sim = coda.samples(mod2,variable.names = parameters2, n.iter = 5000)

plot(mod2_sim)
summary(mod2_sim)

#Combine Samples

mod2_csim = as.mcmc(do.call(rbind,mod2_sim))

#Gelman Rubin Diag

gelman.diag(mod2_sim)


#Auto-Correlation 
autocorr.diag(mod2_sim)


#Effective Size

effectiveSize(mod2_sim)


#Posterior Mean

PM_params2 = colMeans(mod2_csim)
PM_params2


#Comparing with non-informative linear model

coefficients(lmod)


# Residuals

data_jags2$grp

# order means as per the observation groups
yhat2 = PM_params2[1:3][data_jags2$grp]

resid2 = data_jags$y - yhat2
plot(resid2)

plot(y = resid2, x = yhat2)


#Model Summary
summary(mod2_sim)


#High Posterior Density Interval
HPDinterval(mod2_csim)

#HPD 0.9
HPDinterval(mod2_csim, prob = 0.9)

# Posterior Probability of treatment2 > Control group

mean(mod2_csim[,3] > mod2_csim[,1])


# Posterior Probability that treatment2 will increase yield by 10%

mean(mod2_csim[,3] > 1.1 *mod2_csim[,1])


#DIC model2
dic.samples(mod2, n.iter = 5e3)
#Mean deviance:  61.27 
#penalty 5.78 
#Penalized deviance: 67.05 
