#1. Simulate phi_i from Beta(2,2)
#2. Simulate y_i from Binom(10, phi_i)

m = 1e5
y = numeric(m)
phi = numeric(m)

# draw joint probability distribution for phi and y for m

for(i in 1:m){
  phi[i] = rbeta(n=1, shape1 = 2.0, shape2 = 2.0 )
  y[i] = rbinom(1, size=10, prob = phi[i])
}

# instead of forloop use Vectors

phi = rbeta( n=m, shape1 = 2.0, shape2 = 2.0 )
y = rbinom(n = m, size = 10, prob = phi)

table(y)
table(y)/m
plot(table(y)/m)

mean(y)

df = data.frame(y,phi)
df_phi1 = subset(df, phi < 0.5 & phi > 0.4)
hist(df_phi1[,1], freq = FALSE)

df_phi2 = subset(df, phi < 0.9 & phi > 0.8)
hist(df_phi2[,1], freq = FALSE)

df_phi3 = subset(df, phi < 0.8 & phi > 0.7)
hist(df_phi3[,1], freq = FALSE)
