#install.packages("COUNT")
library("COUNT")
data("badhealth")
?badhealth

any(is.na(badhealth)) # return true or false if any data is missing


#Historgram

hist(badhealth$numvisit , breaks = 20)

# check any 0 visits

min(badhealth$numvisit)
#0

sum(badhealth$numvisit == 0)
# 360

plot(jitter(log(numvisit + 0.1)) ~ jitter(age), data = badhealth, subset=badh==0&numvisit>0, xlab ="age", ylab = "log(visitis)")

points(jitter(log(numvisit)) ~ jitter(age), data = badhealth, subset=badh==1&numvisit>0 , col="red")

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


#Convergence Diagnostics

par(mar=c(1,1,1,1))
plot(mod_sim)

gelman.diag(mod_sim)
autocorr.diag(mod_csim)
autocorr.plot(mod_csim)
effectiveSize(mod_sim)
raftery.diag(mod_csim)
#Compute DIC Deviance Information Criterion

dic = dic.samples(mod, 1e3)
dic


#Residuals

X = as.matrix(badhealth[,-1])
X = cbind( X, with(badhealth, badh*age))
head(X)

#pmed_coef = apply(mod_csim,MARGIN = 2, FUN = median)
pmed_coef = apply(mod_csim,MARGIN = 2, FUN = mean)
pmed_coef

#log of lamdaHat
llam_hat = pmed_coef["int"] + X %*% pmed_coef[c("b_badh", "b_age","b_intx")]

lam_hat = exp(llam_hat)
resid = badhealth$numvisit - lam_hat
plot(resid)

plot(lam_hat[which(badhealth$badh==0)], resid[which(badhealth$badh==0)], xlim = c(0,8), ylim = range(resid))

points(lam_hat[which(badhealth$badh==1)], resid[which(badhealth$badh==1)], xlim = c(0,8), col ="red")

var(resid[which(badhealth$badh==0)])
#7.022

var(resid[which(badhealth$badh==1)])
#41.19

summary(mod_sim)


#Let's say, we have two people age 35. One person is in good health and the other person is in poor health.
#What is the posterior probability that the individual with poor health will have more doctor visits?

#badhealth
c("b_badh", "b_age","b_intx")
x1 = c(0,35,0)

#good health
x2 = c(1,35,35)

head(mod_csim)

loglam1 = mod_csim[,"int"] + mod_csim[, c(2,1,3)] %*% x1
loglam2 = mod_csim[,"int"] + mod_csim[, c(2,1,3)] %*% x2


lam1 = exp(loglam1)
lam2 = exp(loglam2)

plot(density(lam1))
plot(density(lam2))

n_sim = length(lam1)
# 15000

y1 = rpois(n_sim, lam1)
y2 = rpois(n_sim, lam2)


table(factor(y1, levels = 0:18))
table(factor(y1, levels = 0:18))/ n_sim
plot( table(factor(y1, levels = 0:18))/ n_sim)

points( table((y2+ 0.1))/ n_sim, col="red")

mean(y2 > y1)

