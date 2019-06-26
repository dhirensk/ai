install.packages("car")
library("car")
head(Leinhardt)
str(Leinhardt)
pairs(Leinhardt)

plot(income ~ infant, data = Leinhardt)
hist(Leinhardt$income)
hist(Leinhardt$infant)


Leinhardt$logIncome = log(Leinhardt$income)
Leinhardt$logInfant = log(Leinhardt$infant)

plot(logIncome ~ logInfant, data = Leinhardt)

###Modelling

lmod = lm(logIncome ~ logInfant, data = Leinhardt)
summary(lmod)

dat = na.omit(Leinhardt)
