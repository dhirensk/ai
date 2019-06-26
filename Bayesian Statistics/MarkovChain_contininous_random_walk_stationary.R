# Continous Random Walk Markov Chain with Stationary Distribution
set.seed(38)

n = 1500
x = numeric(n)
phi = -0.6

for (i in 2:n) {
  x[i] = rnorm(1, mean=phi*x[i-1], sd=1.0)
}

# time series
plot.ts(x)

