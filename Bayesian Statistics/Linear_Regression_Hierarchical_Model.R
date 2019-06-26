library("car")
data("Leinhardt")
head(Leinhardt)
dat = na.omit(Leinhardt)
dat$logincome = log(dat$income)
dat$loginfant = log(dat$infant)

str(dat)

#JAGS Linear Hierarchical Model

mod_string = "model{
  for( i in 1:length(y)){
    y[i] ~ dnorm(mu[i], prec)
    mu[i] = a[region[i]] + b[1]* log_income[i] + b[2]* is_oil[i]
  }
  for(j in 1 :max(region)){
    a[j] ~ dnorm(a0, prec_a)
  }
  
  a0 ~ dnorm(0.0, 1.0/1.0e6)
  prec_a ~ dgamma(1/2.0, 1*10.0/2.0)
  tau = sqrt(1.0/prec_a)
  
  for( j in 1:2){
    b[j] ~ dnorm(0.0, 1.0/1.0e6)
  }
  
  prec ~ dgamma(5.0/2.0, 5*10.0/2.0)
  sig = sqrt(1.0/prec)
}"

set.seed(116)
data_jags = list(y=dat$loginfant, log_income = dat$logincome, is_oil= as.numeric(dat$oil=="yes"), region = as.numeric(dat$region))

table(data_jags$is_oil, data_jags$region)

params = c("a0","a","b", "sig", "tau")

mod = jags.model(textConnection(mod_string), data = data_jags, n.chains = 3)
update(mod, n.iter = 1e3)
mod_sim = coda.samples(mod, variable.names = params, n.iter = 5e3)
mod_csim  = as.mcmc(do.call(rbind, mod_sim))

#Convergance Diagnoistic

plot(mod_sim, ask = TRUE)

gelman.diag(mod_sim)
autocorr.diag(mod_sim)
autocorr.plot(mod_sim)
effectiveSize(mod_sim)

### DIC
dic.samples(mod, 1e3)
summary(mod_sim)
