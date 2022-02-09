## Load ----

library(SoftBart)

## Functions used to generate fake data ----
f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
  10 * x[,4] + 5 * x[,5]

gen_data <- function(n_train, n_test, P, sigma) {
  weights <- runif(n_train) + 1
  weights_test <- runif(n_test)
  X <- matrix(runif(n_train * P), nrow = n_train)
  mu <- f_fried(X)
  X_test <- matrix(runif(n_test * P), nrow = n_test)
  mu_test <- f_fried(X_test)
  Y <- mu + sigma * rnorm(n_train) / sqrt(weights)
  Y_test <- mu_test + sigma * rnorm(n_test) / sqrt(weights_test)
  
  return(list(X = X, Y = Y, mu = mu, X_test = X_test, Y_test = Y_test, 
              mu_test = mu_test, weights = weights, weights_test = weights_test))
}

## Simiulate dataset ----

SEED <- digest::digest2int("testing_weights")
set.seed(SEED)
sim_data <- gen_data(250, 100, 1000, 1)

## Fit the model ----

set.seed(SEED + 2)

hypers_1 <- Hypers(sim_data$X, sim_data$Y, weights = sim_data$weights)
# hypers_2 <- Hypers(sim_data$X, sim_data$Y, weights = rep(2, nrow(sim_data$X)))
opts <- Opts(num_burn = 10000, num_save = 10000)

fit <- softbart(X = sim_data$X, Y = sim_data$Y, X_test = sim_data$X_test, 
                hypers = hypers_1, opts = opts)


plot(fit$sigma)
rmse <- function(x,y) sqrt(mean((x-y)^2))
rmse(fit$y_hat_test_mean, sim_data$mu_test)
rmse(fit$y_hat_train_mean, sim_data$mu)

## Making forest ----

hypers <- Hypers(sim_data$X, sim_data$Y, weights = sim_data$weights)
opts <- Opts()

w <- sim_data$weights
forest <- MakeForest(hypers, opts)
mu_hat <- forest$do_gibbs_weighted(sim_data$X, sim_data$Y, w, sim_data$X_test, opts$num_burn)
mu_hat <- forest$do_gibbs_weighted(sim_data$X, sim_data$Y, w, sim_data$X_test, opts$num_save)
