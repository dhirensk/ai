library("car")  # load the 'car' package
data("Anscombe")  # load the data set
?Anscombe  # read a description of the data
head(Anscombe)  # look at the first few lines of the data
pairs(Anscombe)  # scatter plots for each pair of variables

#linear model non-informative
lmod = lm( Anscombe$education ~ income + young + urban, data = Anscombe )
plot(lmod)

resid(lmod)
plot(resid(lmod))
qqnorm(y = resid(lmod))
str(lmod)
summary(lmod)

##coefficients : Named num [1:4] -286.8388 0.0807 0.8173 -0.1058

# Bayesian Model

library("rjags")

mod_string = " model {
    for (i in 1:length(education)) {
        education[i] ~ dnorm(mu[i], prec)
        mu[i] = b0 + b[1]*income[i] + b[2]*young[i] + b[3]*urban[i]
    }
    
    b0 ~ dnorm(0.0, 1.0/1.0e6)
    for (i in 1:3) {
        b[i] ~ dnorm(0.0, 1.0/1.0e6)
    }
    
    prec ~ dgamma(1.0/2.0, 1.0*1500.0/2.0)
    	## Initial guess of variance based on overall
    	## variance of education variable. Uses low prior
    	## effective sample size. Technically, this is not
    	## a true 'prior', but it is not very informative.
    sig2 = 1.0 / prec
    sig = sqrt(sig2)
} "

data_jags = as.list(Anscombe)

inits = function(){
  init = list("b0"=0.0, "b"= rnorm(n = 3, mean = 0, sd = 1.0e6), "prec"= rgamma(n = 1, shape = 1.0, rate = 1.0))
}

lmod1 = jags.model(textConnection(mod_string),data = data_jags,n.chains = 3, inits = inits)

update(lmod1, n.iter = 1000)
params = c("b0","b","sig")

lmod1_sim = coda.samples(lmod1, variable.names =params, n.iter = 1e5 )
plot(lmod1_sim )


#check gelman rubin diag

gelman.diag(lmod1_sim)
# values above 1.0 means converged  

# check autocorrelation

autocorr.diag(lmod1_sim)

autocorr.plot(lmod1_sim, lag.max = 1000)

# posterior means
summary(lmod1_sim)
# coefficients :  -284.1802 0.0805 0.8110 -0.1057
# from lm
##coefficients :  -286.8388 0.0807 0.8173 -0.1058

