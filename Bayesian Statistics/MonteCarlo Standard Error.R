set.seed(32)
m = 10000
a = 2.0
b = 1.0/3.0
theta = rgamma(n = m, shape = a, rate = b)

#standard Error in mean
se = sd(theta)/sqrt(m)

# confidence interval of mean
CI_lower = mean(theta) - 2 * se
CI_upper = mean(theta) + 2 * se

#montecarlo estimate

#P(theta < 5)
ind = theta < 5
MC_p5 = mean(ind)

#True probability 
True_P = pgamma(5,shape = a, rate = b)

#standard error in montecarlo estimate

se_p5 = sd(ind)/sqrt(m)

# confidence interval of P5
MC_p5_lower = MC_p5 - 2* se_p5
MC_p5_higer = MC_p5 + 2 * se_p5

# we can say that True_p5 is within 2 SE of MC_p5

# True_p5 0.4963317 is within 0.01 (2 * se_p5) from MC_p5 0.4974 i.e between (0.5074004, 0.4873996)
