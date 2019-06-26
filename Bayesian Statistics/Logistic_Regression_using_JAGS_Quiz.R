library("MASS")
data("OME")
?OME # background on the data
head(OME)

any(is.na(OME)) # check for missing values
dat = subset(OME, OME != "N/A") # manually remove OME missing values identified with "N/A"
dat$OME = factor(dat$OME)
str(dat)

par(mfrow=c(2,1))

plot(dat$Age, dat$Correct / dat$Trials )
plot(dat$OME, dat$Correct / dat$Trials )
plot(dat$Loud, dat$Correct / dat$Trials )
plot(dat$Noise, dat$Correct / dat$Trials )

# reference logistic regression model with noninformative prior in R

mod_glm = glm( Correct/Trials ~ Age+ OME + Loud + Noise, data = dat, weights = Trials, family = "binomial" )
summary(mod_glm)


# To get an idea of how the model fits, we can create residual 
# (using a special type of residual for non-normal likelihoods) and in-sample prediction plots.

plot(residuals(mod_glm, type="deviance"))
plot(fitted(mod_glm), dat$Correct/dat$Trials)


#Posterior Mode estimate for OMElow
# same as mean

X = model.matrix(mod_glm)[,-1] # -1 removes the column of 1s for the intercept
head(X)

# Fitting a JAGS model 
mod_string = " model {
	for (i in 1:length(y)) {
		y[i] ~ dbin(phi[i], n[i])
		logit(phi[i]) = b0 + b[1]*Age[i] + b[2]*OMElow[i] + b[3]*Loud[i] + b[4]*Noiseincoherent[i]
	}
	
	b0 ~ dnorm(0.0, 1.0/5.0^2)
	for (j in 1:4) {
		b[j] ~ dnorm(0.0, 1.0/4.0^2)
	}
	
} "

data_jags = as.list(as.data.frame(X))
data_jags$y = dat$Correct # this will not work if there are missing values in dat (because they would be ignored by model.matrix).
#Always make sure that the data are accurately pre-processed for JAGS.
data_jags$n = dat$Trials
str(data_jags) # make sure that all variables have the same number of observations (712).

mod = jags.model(textConnection(mod_string), data = data_jags, n.chains = 3)
update(mod, n.iter = 1e3)
params = c("b0", "b")
mod_sim = coda.samples(mod,variable.names = params, n.iter = 5e3)

# 95% confidence interval for Posterior Mean
raftery.diag(mod_sim)

#Highest Posterior Density Interval
HPDinterval(mod_sim, prob = 0.95)


coefficients(mod)
#  b0 -7.485962 b  0.01779275 -0.16630198  0.17467620  1.66585087


#Point Estimate
# Age = 60, OME = high, Noise = coherent , Loud = 50
b0 = -7.485962
b1 = 0.01779275
b2 = -0.16630198
b3 = 0.17467620
b4 = 1.66585087

# logit(phi[i]) = b0 + b[1]*Age[i] + b[2]*OMElow[i] + b[3]*Loud[i] + b[4]*Noiseincoherent[i]
xb =  b0 + b1 * 60 + b2 * 0 + b3* 50 + b4* 0 # since OMElow = 0 if OME=high and Noiseincoherent = 0 if Noise = coherent

# reversing link function
phat = 1/(1 + exp(-xb))
# 0.9101  

#predicting on entire data set
Xb = b0 + b1 * dat$Age + b2 * (as.numeric(dat$OME)-1) + b3* dat$Loud + b4 * (as.numeric(dat$Noise)-1)

#as numberic gives 2 for low and 1 for high. so make than 1 and 0 we subtract 1. also same for Noise

Phat = 1/(1 + exp(-Xb))

(tab0.7 = table(Phat > 0.7, (dat$Correct / dat$Trials) > 0.7))
sum(diag(tab0.7)) / sum(tab0.7)  
# 0.84
  
  