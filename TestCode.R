library(SoftBart)


f = function(x){
  10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]
}
sigma = 1    #y = f(x) + sigma*z , z~N(0,1)
n = 250      #number of observations
set.seed(1235)
P <- 50
factor_test <- 10
x=matrix(runif(n*P),n,P)
x_new=matrix(runif(n*P*factor_test),n * factor_test,P)
Ey = f(x)
Ey2 = f(x_new)
y=Ey+sigma*rnorm(n)
lmFit = lm(y~.,data.frame(x,y)) #compare lm fit to BART later
X <- x^2
Y <- y
X_test = x_new^2

# group <- c(rep(1,2), 2, 3, 4, rep(5, P - 5))
# group <- 0:(ncol(X) - 1
fit <- softbart(x,y,x_new,opts = Opts(num_burn = 2500, num_save = 2500, update_gamma = FALSE, update_beta = FALSE), hypers = Hypers(X,Y, num_tree = 50, alpha = 5))
plot(fit$alpha)
fit$y <- Ey
plot(fit)

par(mfrow=c(3,2))
plot(fit$alpha, type = 'l')
plot(colMeans(fit$s), col = muted("blue", l = 60, c = 80), cex = .5)
plot(fit$sigma, type = 'l', col = muted("green", l = 60, c = 80))
plot(fit$beta, type = 'l', col = muted("orange", l = 80, c = 80))
plot(fit$gamma, type = 'l', col = muted("red", l = 70, c = 50))
# 
# hist(fit$sigma, col = muted("green", l = 60, c = 80))
# abline(v = sigma, lwd = 3, col = muted("blue", 60, 80))
# 
plot(colMeans(fit$var_counts > 0), col = ifelse(colMeans(fit$var_counts > 0) > .5, muted("blue"), muted("green", l = 80)), pch = 20)
# 
ms <- function(x) mean(x^2)
ms(fit$y_hat_test_mean - Ey2)

