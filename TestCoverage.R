library(SoftBart)

set.seed(1234)

data(iris)

X <- iris[,-1]
Y <- iris[,1]
out <- softbart(X = X, Y = Y, X_test = X, hypers = Hypers(X,Y, width = .001), opts = Opts(num_burn = 2500, num_save = 2500))
plot(out)

# ms(out$y_hat_test_mean - iris[,1])
# 
# stat <- function(i, w) {
#   fit <- lm(Sepal.Length ~., data = i[w,])
#   return(summary(fit)$sigma)
# }
# 
# boot(data = iris, statistic = stat, R = 1000)
