library(bartMachine)


f = function(x){
  10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]
}
sigma = 1    #y = f(x) + sigma*z , z~N(0,1)
n = 250      #number of observations
set.seed(99)
P <- 10
factor_test <- 10
x=matrix(runif(n*P),n,P)
x_new=matrix(runif(n*P*factor_test),n * factor_test,P)
Ey = f(x)
Ey2 = f(x_new)
y=Ey+sigma*rnorm(n)
lmFit = lm(y~.,data.frame(x,y)) #compare lm fit to BART later
X <- x
Y <- y
X_test = x_new

X <- as.data.frame(X)

X[,10] <- as.factor(sample(1:3, size = nrow(X), replace = TRUE))

debug(build_bart_machine)

bartMachine(X = as.data.frame(X), y = y)