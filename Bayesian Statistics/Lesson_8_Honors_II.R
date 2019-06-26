data("warpbreaks")
?warpbreaks
head(warpbreaks)
table(warpbreaks$wool, warpbreaks$tension)
boxplot(breaks ~ wool + tension, data=warpbreaks)
boxplot(log(breaks) ~ wool + tension, data=warpbreaks)


# Re-fit the model with a separate variance for each group. For each variance, use an Inverse-Gamma(1/2, 1/2) prior,
# corresponding to prior sample size 1 and prior guess 1 for each variance.
#Inverse Gamma on sig2 = Gamma on prec
#n_0 = 1.0 # prior effective sample size for sig2
#s2_0 = 1.0 # prior point estimate for sig2
#nu_0 = n_0 / 2.0 # prior parameter for inverse-gamma  shape
#beta_0 = n_0 * s2_0 / 2.0 # prior parameter for inverse-gamma  rate
#prec ~ dgamma(nu_0, beta_0)

mod3_string = " model {
    for( i in 1:length(y)) {
        y[i] ~ dnorm(mu[woolGrp[i], tensGrp[i]], prec[woolGrp[i], tensGrp[i]])
    }
    
    for (j in 1:max(woolGrp)) {
        for (k in 1:max(tensGrp)) {
            mu[j,k] ~ dnorm(0.0, 1.0/1.0e6)
            prec[j,k] ~ dgamma(0.5, 0.5)
            sig[j,k] = sqrt(1.0 / prec[j,k])
        }
    }
    
    
} "
str(warpbreaks)
data3_jags = list(y=log(warpbreaks$breaks), woolGrp=as.numeric(warpbreaks$wool), tensGrp=as.numeric(warpbreaks$tension))
params3 = c("mu", "sig")
mod3 = jags.model(textConnection(mod3_string), data=data3_jags, n.chains=3)
update(mod3, 1e3)
mod3_sim = coda.samples(model=mod3,
                        variable.names=params3,
                        n.iter=5e3)
mod3_csim = as.mcmc(do.call(rbind, mod3_sim))
plot(mod3_sim, ask=TRUE)
## convergence diagnostics
gelman.diag(mod3_sim)
autocorr.diag(mod3_sim)
effectiveSize(mod3_sim)
raftery.diag(mod3_sim)
#Let’s compute the DIC and compare with our previous models.
dic3 = dic.samples(mod3, n.iter=1e3)

#74.62