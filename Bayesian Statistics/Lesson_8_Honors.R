library("car")  # load the 'car' package
data("Anscombe")  # load the data set
#?Anscombe  # read a description of the data
#head(Anscombe)  # look at the first few lines of the data
#pairs(Anscombe)  # scatter plots for each pair of variables

# Bayesian Model

library("rjags")

mod_string = " model {
    for (i in 1:length(education)) {
        education[i] ~ dnorm(mu[i], prec)
        mu[i] = b0 + b[1]*income[i] + b[2]*young[i] + b[3]*urban[i]
    }
    
    b0 ~ dnorm(0.0, 1.0/1.0e6)
  
    for (j in 1:3){
    b[j] ~ ddexp(0.0, sqrt(2.0))  # has variance 1.0     t= sqrt(2/variance)
   }
  #Inverse Gamma on sig2 = Gamma on prec
  n_0 = 1.0 # prior effective sample size for sig2
  s2_0 = 1.0 # prior point estimate for sig2
  nu_0 = n_0 / 2.0 # prior parameter for inverse-gamma  shape
  beta_0 = n_0 * s2_0 / 2.0 # prior parameter for inverse-gamma  rate
  prec ~ dgamma(nu_0, beta_0)
    sig2 = 1.0 / prec
    sig = sqrt(sig2)
} "

Xc = scale(Anscombe, center=TRUE, scale=TRUE)
str(Xc)

data_jags = as.list(data.frame(Xc))
#data_jags = as.list(Anscombe)

#inits = function(){
#  init = list("b0"=0.0, "b"= rnorm(n = 3, mean = 0, sd = 1.0e6), "prec"= rgamma(n = 1, shape = 1.0, rate = 1.0))
#}

lmod1 = jags.model(textConnection(mod_string),data = data_jags,n.chains = 3)

update(lmod1, n.iter = 1000)
params = c("b0","b","sig")

lmod1_sim = coda.samples(lmod1, variable.names =params, n.iter = 1e5 )
plot(lmod1_sim )


#Inference when compared to Quiz 7A
# The inferences are essentially unchanged. The first two coefficients (for income and percentage youth) are significantly positive
#and the percent urban coefficient is still negative.