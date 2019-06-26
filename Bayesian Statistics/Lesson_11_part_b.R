library("rjags")
library("MASS")
data("OME")
#### OLD MODEL  ####
# yi∣ϕi ∼ ind Binomial(ni,ϕi),i=1,…,712,
# logit(ϕi)= β0 + β1 * Age[i]+ β2 I(OME[i]=low)+ β3 Loud[i]+ β4 I(Noise[i]=incoherent) 
# β0 ∼ N(0,52)
# βk∼iidN(0,42),k=1,2,3.

#NEW MODEL

#### OLD MODEL  ####
# yi∣ϕi ∼ ind Binomial(ni,ϕi),i=1,…,712,
# logit(ϕi)= a[ID[i]] + β1 * Age[i]+ β2 I(OME[i]=low)+ β3 Loud[i]+ β4 I(Noise[i]=incoherent) 
# a[j] ∼ N(mu, sig2)
# βk∼iidN(0,42),k=1,2,3.


dat = subset(OME, OME != "N/A")
dat$OME = factor(dat$OME) # relabel OME
dat$ID = as.numeric(factor(dat$ID)) # relabel ID so there are no gaps in numbers (they now go from 1 to 63)

## Original reference model and covariate matrix
mod_glm = glm(Correct/Trials ~ Age + OME + Loud + Noise, data=dat, weights=Trials, family="binomial")
X = model.matrix(mod_glm)[,-1]
X
## Original model (that needs to be extended)
mod_string =  "model {
	for (i in 1:length(y)) {
		y[i] ~ dbin(phi[i], n[i])
		logit(phi[i]) = a[ID[i]] + b[1]*Age[i] + b[2]*OMElow[i] + b[3]*Loud[i] + b[4]*Noiseincoherent[i]
	}
	
	for( j in 1:63){
		a[j] ~ dnorm(mu, prec)
	}
	mu ~ dnorm(0, 1/1e2)
	prec   ~ dgamma(1/2, 1/2)
	sig = sqrt(1/prec)

	for (j in 1:4) {
		b[j] ~ dnorm(0.0, 1.0/4.0^2)
	}
	
} "

data_jags = as.list(as.data.frame(X))
data_jags$y = dat$Correct
data_jags$n = dat$Trials
data_jags$ID = dat$ID

params = c("a", "b" ,"mu","sig")

mod = jags.model(textConnection(mod_string), data = data_jags, n.chains = 3)
update(mod, 1e3)


mod_sim = coda.samples(mod, variable.names = params, n.iter = 5e3)

mod_csim = as.mcmc(do.call(rbind, mod_sim))
# Divergence Diagnostics
#plot(mod_sim, ask = TRUE)
#gelman.diag(mod_sim)
#autocorr.diag(mod_sim)
#effectiveSize(mod_sim)


dic = dic.samples(mod, 1e3)
dic
# 1267
# The DIC value for the original model fit in the quiz for logistic regression is about 1264. 

# The actual number of parameters in this hierarchical model is 69 
#(63 random intercepts, four regression coefficients, and two hyperparameters). 
# What is the effective number of parameters? 
# 27.44

coefficients(mod)
