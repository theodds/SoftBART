## Load ----

library(SoftBart)

f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
                       10 * x[,4] + 5 * x[,5]

gen_data <- function(n_train, n_test, P, sigma) {
  X <- matrix(runif(n_train * P), nrow = n_train)
  mu <- f_fried(X)
  X_test <- matrix(runif(n_test * P), nrow = n_test)
  mu_test <- f_fried(X_test)
  Y <- mu + sigma * rnorm(n_train)
  Y_test <- mu_test + sigma * rnorm(n_test)
  
  return(list(X = X, Y = Y, mu = mu, X_test = X_test, Y_test = Y_test, 
              mu_test = mu_test))
}

## Simiulate dataset ----

SEED <- digest::digest2int("testing_predict")
set.seed(SEED)
sim_data <- gen_data(250, 100, 1000, 1)

## Fitting ----

hypers <- Hypers(sim_data$X, sim_data$Y)
opts <- Opts(cache_trees = TRUE, num_burn = 100)

forest <- MakeForest(hypers, opts)
mu_hat <- forest$do_gibbs(sim_data$X, sim_data$Y, sim_data$X_test, opts$num_burn)
mu_hat <- forest$do_gibbs(sim_data$X, sim_data$Y, sim_data$X_test, opts$num_save)
