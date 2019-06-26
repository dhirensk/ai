library("boot")
data("urine")
head(urine)

#removing rows containing null
dat = na.omit(urine)

# dimensions of dataset
dim(dat)

pairs(dat)

#Feature Scaling, remove first column y
X = scale(dat[,-1], center = TRUE, scale = TRUE)
colMeans(X)
apply(X, MARGIN = 2, FUN = sd)

#Bayesian Logistic Model based on logit
library("rjags")
mod1_string = "model{
  for ( i in 1: length(y)){
    y[i] ~ dbern(p[i])
    
    logit(p[i]) = int + b[1]* gravity[i]+ b[2]*ph[i] + b[3]* osmo[i] + b[4]*cond[i] + b[5]* urea[i] + b[6]* calc[i]
  }
  
  int ~ dnorm( 0.0, 1.0/25.0)
   for (j in 1:6){
     b[j] ~ ddexp(0.0, sqrt(2.0))  # has variance 1.0
   }
}"


set.seed(92)
data_jags = list(y = dat$r, gravity = X[,"gravity"], ph = X[,"ph"], osmo = X[,"osmo"], cond= X[,"cond"], urea = X[,"urea"], calc = X[,"calc"])

params = c("int", "b")


mod1 = jags.model(textConnection(mod1_string), data = data_jags, n.chains = 3)

update(mod1, n.iter = 1e3)

mod1_sim = coda.samples(mod1, variable.names = params, n.iter = 5e3)

mod1_csim = as.mcmc(do.call(rbind, mod1_sim))
#convergence Diagnostics

plot(mod1_sim, ask = TRUE)
gelman.diag(mod1_sim)
autocorr.diag(mod1_sim)
autocorr.plot(mod1_sim)
effectiveSize(mod1_sim)

##Calculate DIC
dic1 =  dic.samples(mod1, n.iter = 1e3)

#Posterior
par(mfrow = c(3,2))
densplot(mod1_csim[,1:6], xlim = c(-3,3))
colnames(X)


# we keep the coefficients for gravity, conductivity and calcium concentration as others have almost 0 mean b[5] also highy correlated with b[0]

mod2_string  = "model{
  for ( i in 1: length(y)){
    y[i] ~ dbern(p[i])
    
    logit(p[i]) = int + b[1]* gravity[i]+ b[2]*cond[i] +  b[3]* calc[i]
  }
  
  int ~ dnorm( 0.0, 1.0/25.0)
   for (j in 1:3){
     b[j] ~ dnorm( 0.0, 1.0/25.0)   # non-informative for logistic regression
   }
}"

mod2 = jags.model(textConnection(mod2_string), data = data_jags, n.chains = 3)

update(mod2, n.iter = 1e3)

mod2_sim = coda.samples(mod2, variable.names = params, n.iter = 5e3)

mod2_csim = as.mcmc(do.call(rbind, mod2_sim))

dic2 = dic.samples(mod2, n.iter = 1e3)

summary(mod2_sim)


colMeans(mod1_csim)
pm_coeff = colMeans(mod2_csim)
# b[1]      b[2]      b[3]       int 
# 1.413340 -1.352343  1.896415 -0.147686 

1/(1 + exp(0.15))

# Prediction for specific gravity is at the average.
# conductivity is one standard deviation below the mean. 
# calcium concentration is one standard deviation above the mean.

1/(1 + exp(0.15 -1.42*0.0 - -1.36 * (-1) -1.88 * (1.0)))
# 0.9564784

# matrix multiply X with Beta co-efficients and add intercept
pm_Xb = pm_coeff["int"] + X[,c(1,4,6)] %*% pm_coeff[1:3]

#prediction
phat = 1/(1 + exp(-pm_Xb))


#plot the pred vs data

plot( phat, dat$r)
plot( phat, jitter(dat$r))


tab0.5 = table(phat > 0.5, dat$r)
tab0.5


acc1 = sum(diag(tab0.5)/sum(tab0.5))



tab0.3 = table(phat > 0.3, dat$r)
tab0.3

#accuracy

acc2 = sum(diag(tab0.3)/sum(tab0.3))
acc2
