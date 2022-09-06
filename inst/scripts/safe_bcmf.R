## Load (116) ----

library(SoftBart)
library(tidyverse)
library(zeallot)

## Globals ----

# set.seed(digest::digest2int("safe bcmf"))

N     <- 250
P     <- 10
sigma <- 1
omega <- 1

## Generate data ----

gen_data <- function(N, P, sigma, omega) {
  f <- function(x) 10 * sin(x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
    10 * x[,4] + 5 * x[,5]
  beta <- c(10, 10, 5, 10, 5, rep(0, P - 5))
  
  X  <- matrix(runif(N * P), nrow = N)
  mu <- (1 - omega) * f(X) + omega * X %*% beta %>% as.numeric()
  Y  <- mu + sigma * rnorm(N)
  
  return(list(Y = Y, X = X, mu = mu))
}

## Make Data ----

c(Y, X, mu) %<-% gen_data(N, P, sigma, omega)
my_data <- data.frame(X = X, Y = Y)

safe_bart <- function(formula, data, test_data = NULL, num_burn, num_thin, 
                      num_save, normalize_X = TRUE) {
  X <- model.matrix(formula, data = data)
  if("(Intercept)" %in% colnames(X)) {
    idx <- which("(Intercept)" %in% colnames(X))
    X_noint <- X[,-idx]
  } else {
    X_noint <- X
  }
  ecdfs <- lapply(1:ncol(X_noint), function(i) ecdf(X_noint[,i]))
  X_bart <- X_noint
  if(normalize_X) {
    for(p in 1:ncol(X_noint)) {
      X_bart[,p] <- ecdfs[[p]](X_noint[,p])
    } 
  }
  Y <- model.frame(formula, data = my_data) %>% model.response
  sd_y <- sd(Y)
  mu_y <- mean(Y)
  Y <- (Y - mu_y) / sd_y
  hypers <- Hypers(X_noint, Y, normalize_Y = FALSE)
  # hypers$sigma_mu_hat <- sigma_mu_hat
  opts <- Opts(update_s = FALSE)
  my_forest <- MakeForest(hypers, opts)
  sigma <- hypers$sigma_hat
  
  XtY      <- t(X) %*% Y
  XtXi     <- solve(t(X) %*% X)
  beta_hat <- XtXi %*% XtY
  V        <- sigma^2 * XtXi
  beta     <- beta_hat
  
  beta_out <- matrix(NA, nrow = num_save, ncol = ncol(X))
  f_out    <- matrix(NA, nrow = num_save, ncol = nrow(X))
  mu_out   <- matrix(NA, nrow = num_save, ncol = nrow(X))
  sigma_out <- numeric(num_save)
  sigma_mu_out <- numeric(num_save)
  # f <- my_forest$do_predict(X_bart)
  
  for(i in 1:num_burn) {
    R         <- Y - as.numeric(X %*% beta)
    f         <- my_forest$do_gibbs(X_bart, R, X_bart, 1)
    sigma     <- my_forest$get_sigma()
    R         <- Y - as.numeric(f)
    beta_hat  <- XtXi %*% t(X) %*% R %>% as.numeric()
    V         <- sigma^2 * XtXi
    beta      <- MASS::mvrnorm(n = 1, mu = beta_hat, Sigma = V)
  }
  
  for(i in 1:num_save) {
    for(j in 1:num_thin) {
      R         <- Y - as.numeric(f)
      beta_hat  <- XtXi %*% t(X) %*% R %>% as.numeric()
      V         <- sigma^2 * XtXi
      beta      <- MASS::mvrnorm(n = 1, mu = beta_hat, Sigma = V)
      R         <- Y - as.numeric(X %*% beta)
      f         <- my_forest$do_gibbs(X_bart, R, X_bart, 1)
      sigma     <- my_forest$get_sigma()      
    }
    beta_out[i,] <- beta * sd_y; beta_out[i,1] <- beta_out[i,1] + mu_y
    f_out[i,] <- f * sd_y
    mu_out[i,] <- f_out[i,] + (X %*% beta_out[i,] %>% as.numeric())
    sigma_out[i] <- sigma * sd_y
    sigma_mu_out[i] <- my_forest$get_sigma_mu() * sd_y
  }
  
  fitted_safe_bart <- list(beta = beta_out, f = f_out, mu = mu_out, 
                           sigma = sigma_out, sigma_mu = sigma_mu_out)
  
  return(fitted_safe_bart)
  
}

my_safebart <- safe_bart(Y ~ ., my_data, 10000, 1, 10000)

par(mfrow = c(1,2))
plot(my_safebart$sigma, type = 'l')
plot(my_safebart$sigma_mu, type = 'l')

rmse <- function(x, y) sqrt(mean((x - y)^2))
rmse(colMeans(my_safebart$mu), mu)
