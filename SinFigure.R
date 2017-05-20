library(SoftBart)
library(tidyverse)
library(rpart)

SEED <- 100
SIGMA <- 2
N <- 250

f <- function(x) 10 * sin(2 * pi * x)



set.seed(SEED)

X <- as.matrix(runif(N))
Y <- f(X) + SIGMA * rnorm(N)
X_test <- as.matrix(seq(from = 0, to = 1, length = 1000))

fit_bart <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = 1E-10, num_tree = 50), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
fit_softbart <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = .1, num_tree = 50), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
fit_tree <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = 1E-10, num_tree = 1), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
fit_softtree <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = .1, num_tree = 1), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
# fit_part <- rpart(Y ~ ., data = as.data.frame(X))

# mu_part <- predict(fit_part, as.data.frame(X_test))
mu_tree <- fit_tree$y_hat_test_mean
mu_bart <- fit_bart$y_hat_test_mean
mu_softtree <- fit_softtree$y_hat_test_mean
mu_softbart <- fit_softbart$y_hat_test_mean

df_sinusoid <- data.frame(X = rep(X_test, 4), 
                        f = c(mu_tree, mu_bart, mu_softtree, mu_softbart), 
                        method = rep(c("DT", "BART", "SoftDT","SoftBART"), each = nrow(X_test)), 
                        the_fun = rep("Sinusoid", 4 * nrow(X_test)))
set.seed(SEED)

g <- function(x) 10 * x
Y <- g(X) + SIGMA * rnorm(N)

# fit_part <- rpart(Y ~ ., data = as.data.frame(X))
fit_tree <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = 1E-10, num_tree = 1), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
fit_bart <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = 1E-10, num_tree = 50), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
fit_softtree <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = .1, num_tree = 1), opts = Opts(update_beta = FALSE, update_gamma = FALSE))
fit_softbart <- softbart(X = X, Y = Y, X_test = X_test, hypers = Hypers(X,Y, width = .1, num_tree = 50), opts = Opts(update_beta = FALSE, update_gamma = FALSE))

# mu_part <- predict(fit_part, as.data.frame(X_test))
mu_tree <- fit_tree$y_hat_test_mean
mu_bart <- fit_bart$y_hat_test_mean
mu_softtree <- fit_softtree$y_hat_test_mean
mu_softbart <- fit_softbart$y_hat_test_mean

df_linear <- data.frame(X = rep(X_test, 4), 
                        f = c(mu_tree, mu_bart, mu_softtree, mu_softbart), 
                        method = rep(c("DT", "BART", "SoftDT", "SoftBART"), each = nrow(X_test)), 
                        the_fun = rep("Linear", 4 * nrow(X_test)))

save(df_linear, df_sinusoid, f, g, file = "SinFigure.RData")

p1 <- ggplot(df_linear, aes(x = X, y = f)) + geom_line(color = muted("blue", 70, 80), size = 1.1) + facet_wrap(~method) + stat_function(fun = g, linetype = 'dashed', color = 'black') + theme_bw() + xlab("$x$") + ylab("$\\widehat f(x)$")
p2 <- ggplot(df_sinusoid, aes(x = X, y = f)) + geom_line(color = muted("red", 70, 80), size = 1.1) + facet_wrap(~method) + stat_function(fun = f, linetype = 'dashed', color = 'black') + theme_bw() + xlab("$x$") + ylab("$\\widehat f(x)$")

gridExtra::grid.arrange(p1, p2, ncol = 2)


# df_plot <- rbind(df_sinusoid, df_linear)
# 
# ggplot(df_plot, aes(x=X, y=f, color = method)) + geom_line() + facet_wrap(~method+the_fun) + 
#   stat_function(fun = function(x) 10 * x, linetype = "dashed", color = "black")
