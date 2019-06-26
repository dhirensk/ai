dat = read.table("http://www.stat.ufl.edu/~winner/data/pgalpga2008.dat")
names(dat)  <- c("distance","accuracy","sex")
str(dat)

dat$sex = dat$sex -1
#E(y)= b0 + b1x1 + b2x2

lmod = lm( accuracy ~ distance + sex, data = dat)
summary(lmod)
# E(y) = 147 -0.32 * distance + 8.94 * sex

plot(resid(lmod))
plot( y= resid(lmod), x = fitted(lmod))

