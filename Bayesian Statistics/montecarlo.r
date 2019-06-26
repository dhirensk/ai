set.seed(32)
m = 10000
a = 2.0
b = 1.0/3.0

theta = rgamma(n=m,shape = a,rate = b )
hist(theta, freq = FALSE)
#curve(dgamma(x,shape = a,rate = b),col="blue", add = TRUE)
curve(dgamma(x,shape = a,rate = b),from = 0, to = 100,col="blue")


MC_mean = sum(theta)/m
MC_mean = mean(theta)
True_mean = a/b
MC_var = var(theta)
True_var = a/b^2

# Probablity of theta < 5
# create a indicator function
ind = theta < 5
MC_P5 = mean(ind)

True_P5 = pgamma(5,shape = a, rate = b)

# find quantile for probability = 0.9
MC_quantile = quantile(theta, probs= 0.9)

True_quantile = qgamma(0.9,shape = a, rate = b)