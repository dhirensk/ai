# log of g function
lg = function(mu, n, ybar) {
  mu2 = mu^2
  n * (ybar * mu - mu2 / 2.0) - log(1 + mu2)
}

#Metropolis Hastings 
mh = function(n, ybar, n_iter, mu_init, cand_sd) {
  ## Random-Walk Metropolis-Hastings algorithm
  
  ## step 1, initialize
  mu_out = numeric(n_iter)
  accpt = 0
  mu_now = mu_init
  lg_now = lg(mu=mu_now, n=n, ybar=ybar)
  
  ## step 2, iterate
  for (i in 1:n_iter) {
    ## step 2a
    mu_cand = rnorm(n=1, mean=mu_now, sd=cand_sd) # draw a candidate
    
    ## step 2b
    lg_cand = lg(mu=mu_cand, n=n, ybar=ybar) # evaluate log of g with the candidate
    lalpha = lg_cand - lg_now # log of acceptance ratio
    alpha = exp(lalpha)
    
    ## step 2c
    u = runif(1) # draw a uniform variable which will be less than alpha with probability min(1, alpha)
    if (u < alpha) { # then accept the candidate
      mu_now = mu_cand
      accpt = accpt + 1 # to keep track of acceptance
      lg_now = lg_cand
    }
    
    ## collect results
    mu_out[i] = mu_now # save this iteration's value of mu
  }
  
  ## return a list of output
  list(mu=mu_out, accpt=accpt/n_iter)
}

#Setup the Data
# y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)  #Lesson 4
y = c(-0.2, -1.5, -5.3, 0.3, -0.8, -2.2)  #quiz
ybar = mean(y)
n = length(y)
#View Data
hist(y, freq=FALSE, xlim=c(-1.0, 3.0)) # histogram of the data
curve(dt(x=x, df=1), lty=2, add=TRUE) # prior for mu
points(y, rep(0,n), pch=1) # individual data points
points(ybar, 0, pch=19) # sample mean

set.seed(43) # set the random seed for reproducibility

#Run the Metropolis Hastings
library("coda")

post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=0.5)
str(post)
summary(as.mcmc(post$mu))
traceplot(as.mcmc(post$mu))

#------sd 1.5 gives acceptance of 0.3 which is between 23-50%
post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=1.5)
post$accpt
summary(as.mcmc(post$mu))
traceplot(as.mcmc(post$mu))

post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=3.0)
post$accpt
traceplot(as.mcmc(post$mu))
summary(as.mcmc(post$mu))

post = mh(n=n, ybar=ybar, n_iter=1e3, mu_init=0.0, cand_sd=4.0)
post$accpt
summary(as.mcmc(post$mu))
traceplot(as.mcmc(post$mu))