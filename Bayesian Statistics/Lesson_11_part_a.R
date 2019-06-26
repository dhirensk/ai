library("rjags")
dat = read.csv(file="pctgrowth.csv", header=TRUE)
head(dat)
# yi ∣ θgi,σ2 ∼ind N(θgi,σ2),i=1,…,53, gi ∈{1,…,5}
# θg ∣ μ,τ2 ∼ iid N(μ,τ2),g=1,…,5
# μ∼N(0,1e6),
# τ2∼IG(1/2,1*3/2)
# σ2∼IG(2/2,2*1/2)

mod_string = "model {
  for(i in 1: length(y)){
    y[i] ~ dnorm(theta_g[grp[i]], 1/sig2)
  }
  
  for(j in 1: max(grp)){
    theta_g[j] ~ dnorm(mu, 1/tau2)
  }
  
  mu ~ dnorm(0.0, 1.0/1e6)
  tau2 ~ dgamma(1/2, 1*3/2)
  tau = sqrt(tau2)
  sig2 ~ dgamma(2/2, 2*1/2)
  sig = sqrt(sig2)
}"

data_jags = as.list(dat)

mod = jags.model(textConnection(mod_string), data = data_jags, n.chains = 3)

params = c("theta_g","mu","tau","sig")

update(mod, n.iter = 1e3)

mod_sim = coda.samples(mod,variable.names = params, n.iter = 5e3)
mod_csim = as.mcmc(do.call(rbind, mod_sim))
plot(mod_sim, ask = TRUE)

# Convergence Diagnostic

gelman.diag(mod_sim)
autocorr.diag(mod_sim)
effectiveSize(mod_sim)

dic = dic.samples(mod, n.iter = 1e3)

raftery.diag(mod_csim)

pm_params = coefficients(mod)
means_theta = unlist(pm_params["theta_g"])
means_theta
#   theta_g1    theta_g2    theta_g3    theta_g4    theta_g5 
#0.38105768 -1.35487077 -1.27639715  0.08709312 -0.22465705 

#using non-informative prior
means_anova = tapply(dat$y, INDEX=dat$grp, FUN=mean)
## dat is the data read from pctgrowth.csv
means_anova
#1          2          3          4          5 
#0.9900000 -1.6166667 -1.1625000  0.2533333 -0.3714286 


plot(means_anova)
points(means_theta, col="red") ## where means_theta are the posterior point estimates for the industry means.


#making posterior predictive distribution n_sim simulations for new company in group 1.
head(mod_csim)
n_sim = nrow(mod_csim)
y_pred1 = rnorm(n_sim, mean =  mod_csim[,"theta_g[1]"], sd = mod_csim[,"sig"])

hist(y_pred1)

#making posterior predictive simulations for new company not observed so far
# we sample two times, 1st from the normal distribution of theta, then from the normal distribution of observations

theta_pred = rnorm(n_sim, mean = mod_csim[,"mu"], sd = mod_csim[,"tau"])  #15000 sims of theta

#simulate observations

y_pred2 = rnorm(n_sim, mean = theta_pred, sd = mod_csim[,"sig"])
hist(y_pred2)
