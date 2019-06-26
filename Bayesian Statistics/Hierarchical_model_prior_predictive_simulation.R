dat = read.table("cookies.txt",header = TRUE)
head(dat)
boxplot(chips ~location, data = dat)

set.seed(112)
n_sim = 500

alpha_pri = rexp(n = n_sim, rate = 1.0/2.0)
beta_pri = rexp(n = n_sim, rate = 5.0)

# Expection/mean of gamma distribution is alpha/beta
mu_pri = alpha_pri/beta_pri
sig_pri = sqrt( alpha_pri/beta_pri^2)

summary(mu_pri)
summary(sig_pri)

#Simulate lambdas from the alphas and betas
lam_pri = rgamma(n_sim, shape = alpha_pri, rate = beta_pri)
summary(lam_pri)


#Simulating prior observations

y_pri = rpois(n_sim, lambda = lam_pri)
summary(y_pri)
  
#Take 5 lambda draws

lam_pri = lam_pri[1:5]
y_pri = rpois(150, lambda = rep(lam_pri, each = 30))
y_pri

rep(lam_pri, each = 30)
