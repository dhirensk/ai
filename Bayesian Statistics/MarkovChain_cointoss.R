Q = matrix(c(0.0, 0.5, 0.0, 0.0, 0.5,
             0.5, 0.0, 0.5, 0.0, 0.0,
             0.0, 0.5, 0.0, 0.5, 0.0,
             0.0, 0.0, 0.5, 0.0, 0.5,
             0.5, 0.0, 0.0, 0.5, 0.0), 
           nrow=5, byrow=TRUE)

Q %*% Q    # Matrix Multiplication
Q * Q      # Element-wise


c(0.2, 0.2, 0.2, 0.2, 0.2) %*% Q

n = 5000
x = numeric(n)
x[1] = 1 # fix the state as 1 for time 1
for (i in 2:n) {
  x[i] = sample.int(5, size=1, prob=Q[x[i-1],])
  # draw the next state from the intergers 1 to 5 with probabilities from the transition matrix Q, based on the previous value of X.
}

table(x) / n