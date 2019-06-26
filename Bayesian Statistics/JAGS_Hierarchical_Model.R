dat = read.table("cookies.txt", header = TRUE)
library("rjags")
head(dat)
mod_string = "model {
  
  for(i in 1: length(chips)){
    chips[i] ~ dpois(lam[location[i]])
  }
  
  for(j in 1: max(location)){
    lam[j] ~ dgamma(alpha, beta)
  }
  
  mu ~ dgamma(2.0, 1.0/5.0)
  sig ~ dexp(1.0)
  
  alpha = mu^2/ sig^2
  beta = mu/sig^2
}"

set.seed(113)
data_jags = as.list(dat)

params = c("lam","mu", "sig")

mod = jags.model(textConnection(mod_string),data = data_jags,n.chains = 3 )
update(mod, 1e3)

mod_sim = coda.samples(mod, variable.names = params, n.iter = 5e3)

mod_csim = as.mcmc(do.call(rbind, mod_sim))


plot(mod_sim, ask = TRUE)

dic = dic.samples(mod, n.iter = 1e3)
dic

### Model Checking

pm_params = colMeans(mod_csim)
pm_params


#Observation Residuals

# yhat = rpois( length(dat$chips), lambda = rep(pm_params[1:5], each=30)) why not this


yhat = rep(pm_params[1:5], each=30)

resid = dat$chips - yhat
plot(resid)

plot(jitter(yhat), resid )

var(resid[which(yhat <7)])

var(resid[which(yhat >11)])



# Lambda Residuals

lam_resid = pm_params[1:5] -  pm_params["mu"] 
plot(lam_resid)
abline(h = 0, lty=2)  



#model summary

summary(mod_sim)

#Posterior Predictive Simulation
n_sim = nrow(mod_csim)
n_sim

# Posterior MCMC samples of alpha and beta and lambda
post_alpha = mod_csim[,"mu"]^2/mod_csim[,"sig"]^2
post_beta = mod_csim[,"mu"]/mod_csim[,"sig"]^2
lam_pred = rgamma(n_sim, shape = post_alpha, rate = post_beta)

hist(lam_pred)
mean( lam_pred > 15)

#Posterior Prediction of Samples

y_pred = rpois(n_sim, lambda = lam_pred)

hist(y_pred)
mean(y_pred > 15)

#hist of original data

hist(dat$chips)

# prediction for location1

y_pred1 = rpois(n_sim, lambda = mod_csim[,"lam[1]"])
hist(y_pred1)

mean(y_pred1 < 7)  
