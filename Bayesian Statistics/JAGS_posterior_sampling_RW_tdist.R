library("rjags")
library("coda")

# create the model containing, likelihood and prior
# y ~iid~ N(mu,1) sd = 1,  tau = 1/sig2 ,dnorm(mu, tau)
# mu ~ t(mu=0, tau=1/scale, degree of freedom = 1)
model_string = " model{
  for ( i in 1:n){
    y[i] ~ dnorm(mu, 1/sig2)
  }
  mu ~ dt(0,1/1.0, 1)
  sig2 = 1.0
} "

# create data for model
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
n = length(y)

#input list
data_jags = list(y=y,n=n)

#parameter
params = c("mu")

#inits function
inits = function(){
  inits = list("mu"=30.0)
}

#compile
model = jags.model(file = textConnection(model_string),data = data_jags,inits = inits )

#RUN MCMC Sampler
update(model, 500)

#Sample posterior
model_sim = coda.samples(model, variable.names = params,n.iter = 1000)
plot(model_sim)
summary(model_sim)
