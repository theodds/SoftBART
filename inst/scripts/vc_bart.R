## Load ----

library(SoftBart)
library(tidyverse)
library(zeallot)

## Some defaults ----

set.seed(1112111)
N     <- 500
P     <- 10
sigma <- 1

## Generate Some Data ----

fried <- function(x) 10 * sin(x[,1] * x[,2]) + 20 * (x[,3]-0.5)^2 + 
                       10 * x[,4] + 5 * x[,5]

gen_data <- function(N, P, sigma) {
  X  <- matrix(runif(N*P), nrow = N)
  M  <- rnorm(N)
  mu <- M * fried(X)
  Y  <- mu + sigma * rnorm(N)
  
  out <- list(X = X, M = M, mu = mu, Y = Y)
  return(out)
}

c(X, M, mu, Y) %<-% gen_data(N, P, sigma)

## Make objects ----

mu_y    <- mean(Y)
sd_y    <- sd(Y)
Y_scale <- (Y - mu_y) / sd_y

hypers <- Hypers(X, Y_scale, normalize_Y = FALSE)
opts   <- Opts()

my_forest <- MakeForest(hypers, opts)

run_vc_bart <- function(forest, y, X, M, num_burn, num_save, num_thin) {
  beta_out  <- matrix(NA, nrow = num_save, ncol = nrow(X))
  alpha_out <- numeric(num_save)
  sigma_out <- numeric(num_save)
  
  alpha <- 0
  for(i in 1:num_burn) {
    R <- (y - alpha) / M
    beta <- forest$do_gibbs_weighted(X, R, M^2, X, 1)
    sigma <- forest$get_sigma()
    R <- (y - M * beta)
    alpha <- rnorm(1, mean(R), sigma / sqrt(length(R)))
  }
  for(i in 1:num_save) {
    for(j in 1:num_thin) {
      R <- (y - alpha) / M
      beta <- forest$do_gibbs_weighted(X, R, M^2, X, 1)
      sigma <- forest$get_sigma()
      R <- (y - M * beta)
      alpha <- rnorm(1, mean(R), sigma / sqrt(length(R)))      
    }
    alpha_out[i] <- alpha
    beta_out[i,] <- beta
    sigma_out[i] <- sigma
  }
  return(list(alpha = alpha_out, beta = beta_out, sigma = sigma_out))
}

fitted_vc <- run_vc_bart(my_forest, Y_scale, X, M, 1000, 1000, 1)
rmse <- function(x, y) sqrt(mean((x-y)^2))
rmse(M * colMeans(fitted_vc$beta) * sd_y, mu)
plot(M * colMeans(fitted_vc$beta) * sd_y, mu)
