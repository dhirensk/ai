#install.packages("car")
library("car")
head(Leinhardt)
str(Leinhardt)
pairs(Leinhardt)

plot(infant ~ income, data = Leinhardt)
hist(Leinhardt$income)
hist(Leinhardt$infant)

Leinhardt$logInfant = log(Leinhardt$infant)
Leinhardt$logIncome = log(Leinhardt$income)


plot(logInfant ~ logIncome, data = Leinhardt)

###Modelling
# Using non-informative prior

lmod = lm(logInfant ~ logIncome, data = Leinhardt)
summary(lmod)

dat = na.omit(Leinhardt)

############################ MODEL1 ####################################

mod1_string = "model{
  #likelihood
  for(i in 1:n){
    # prec = 1/sig2
    y[i] ~ dnorm( mu[i], prec)
    mu[i] = b[1]+ b[2]* logIncome[i]
  }
  #  normal priors for b
  for(j in 1:2){
    b[j] ~ dnorm(0.0, 1.0/1.0e6)
  }
  
  #Inverse Gamma on sig2 = Gamma on prec
  n_0 = 5.0 # prior effective sample size for sig2
  s2_0 = 10.0 # prior point estimate for sig2
  nu_0 = n_0 / 2.0 # prior parameter for inverse-gamma
  beta_0 = n_0 * s2_0 / 2.0 # prior parameter for inverse-gamma
  #
  prec ~ dgamma(nu_0, beta_0) 
  sig2 = 1/prec
  sig = sqrt(sig2)
}"

set.seed(72)
data1_jags = list( y=dat$logInfant, n = nrow(dat), logIncome = dat$logIncome)

#parameters we want to monitor
params1 = c("b","sig")

#parameters of likelihood
Inits1 = function(){
  Inits = list("b" = rnorm(n=2,mean = 0.0, sd = 100.0 ), "prec" = rgamma(n = 1, shape = 1.0, rate = 1.0) )
}

#compile

mod1 = jags.model(textConnection(mod1_string),data = data1_jags, inits = Inits1, n.chains = 3)


#Burn in period for 1000 samples

update(mod1,n.iter = 1000)


mod1_sim = coda.samples(mod1, variable.names = params1, n.iter = 5e3)

plot(mod1_sim)



# Combining the multiple chains (combining matrix) into single chain

mod1_csim = as.mcmc(do.call(rbind,mod1_sim))

str(mod1_csim)


# Gelman Rubin Disagnostic

gelman.diag(mod1_sim)


# Autocorrelation between samples

autocorr.diag(mod1_sim)

# Effective Sample Size

effectiveSize(mod1_sim)


# Posterior Summary
summary(mod1_sim)


# Residuals from flat prior model
lmod0 = lm(infant ~ income, data = dat)
plot( resid(lmod0))

plot(  x = predict(lmod0), y = resid(lmod0))


# Normal QQ Plots

qqnorm(y = resid(lmod0))


### Residuals from Bayesian Model

# Design Matrix
X = cbind( rep( 1.0, data1_jags$n), dat$logIncome)

head(X)

#Posterior Means of Parameters
PM_params1 = colMeans(mod1_csim)
PM_params1[1:2]

yhat1 = X %*% PM_params1[1:2]
yhat1
yhat1 = drop(yhat1)


# residuals from bayesian model
resid1 = dat$logInfant - yhat1
plot(resid1)

plot( yhat1, resid1)

# Normal QQ plot

qqnorm(resid1)
############################### END MODEL1 #########################

#Using JAGS

library("rjags")
mod2_string = "model{
  #likelihood
  for(i in 1:n){
    # prec = 1/sig2
    y[i] ~ dnorm( mu[i], prec)
    mu[i] = b[1]+ b[2]* logIncome[i] + b[3]* is_oil[i] 
  }
  #  normal priors for b
  for(j in 1:3){
    b[j] ~ dnorm(0.0, 1.0/1.0e6)
  }
  
  #Inverse Gamma on sig2 = Gamma on prec
  n_0 = 5.0 # prior effective sample size for sig2
  s2_0 = 10.0 # prior point estimate for sig2
  nu_0 = n_0 / 2.0 # prior parameter for inverse-gamma
  beta_0 = n_0 * s2_0 / 2.0 # prior parameter for inverse-gamma
  #
  prec ~ dgamma(nu_0, beta_0) 
  sig2 = 1/prec
  sig = sqrt(sig2)
}"

set.seed(72)
# as.number(condition) gives 1/0
data2_jags = list( y=dat$logInfant, n = nrow(dat), logIncome = dat$logIncome, is_oil = as.numeric(dat$oil=="yes"))

#parameters we want to monitor
  params2 = c("b","sig")

#parameters of likelihood
Inits2 = function(){
  Inits = list("b" = rnorm(n=3,mean = 0.0, sd = 100.0 ), "prec" = rgamma(n = 1, shape = 1.0, rate = 1.0) )
}

#compile

mod2 = jags.model(textConnection(mod2_string),data = data2_jags, inits = Inits2, n.chains = 3)


#Burn in period for 1000 samples

update(mod2,n.iter = 1000)


mod2_sim = coda.samples(mod2, variable.names = params2, n.iter = 5e3)

plot(mod2_sim)



# Combining the multiple chains (combining matrix) into single chain

mod2_csim = as.mcmc(do.call(rbind,mod2_sim))

str(mod2_csim)


# Gelman Rubin Disagnostic

gelman.diag(mod2_sim)


# Autocorrelation between samples

autocorr.diag(mod2_sim)

# Effective Sample Size

effectiveSize(mod2_sim)


# Posterior Summary
summary(mod2_sim)


# Residuals from flat prior model
lmod0 = lm(infant ~ income+ is_oil, data = data2_jags)
plot( resid(lmod0))

plot(  x = predict(lmod0), y = resid(lmod0))


# Normal QQ Plots

qqnorm(y = resid(lmod0))


### Residuals from Bayesian Model

# Design Matrix
X2 = cbind( rep( 1.0, data2_jags$n), data2_jags$logIncome, data2_jags$is_oil)

head(X2)

#Posterior Means of Parameters
PM_params2 = colMeans(mod2_csim)
PM_params2[1:3]

yhat2 = X2 %*% PM_params1[1:3]
yhat2
yhat2 = drop(yhat2)  #covert matrix to vector


# residuals from bayesian model

resid2 = dat$logInfant - yhat2
par(mfrow=c(2,1))
#plot(resid2)

plot( yhat2, resid2)
plot(yhat1, resid1)

sd(resid2)

# compare t and normal distribution
curve(dnorm(x),from = -5, to = 5)
curve(dt(x,df =1), from = -5, to = 5, add = TRUE, col="red")

# Normal QQ plot

qqnorm(resid2)


# Outliers/ Highest Residuals

rownames(dat)
str(dat)
# get rownames for the outliers by decreasing sort of residuals
head(rownames(dat)[order(resid2,decreasing = TRUE)])


#Deviance Information Criterion DIC

dic.samples(mod1, n.iter = 1e3)
dic.samples(mod2, n.iter = 1e3)
