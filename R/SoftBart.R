#' Create hyperparameter object for SoftBart
#'
#' Creates a list which holds all the hyperparameters for use with the softbart
#' command.
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nx1 vector of training data response.
#' @param group For each column of X, gives the associated group
#' @param alpha Positive constant controlling the sparsity level
#' @param beta Parameter penalizing tree depth in the branching process prior
#' @param gamma Parameter penalizing new nodes in the branching process prior
#' @param k Related to the signal-to-noise ratio, sigma_mu = 0.5 / (sqrt(num_tree) * k). BART defaults to k = 2.
#' @param sigma_hat A prior guess at the conditional variance of Y. If not provided, this is estimated empirically by linear regression.
#' @param shape Shape parameter for gating probabilities
#' @param width Bandwidth of gating probabilities
#' @param num_tree Number of trees in the ensemble
#' @param alpha_scale Scale of the prior for alpha; if not provided, defaults to P
#' @param alpha_shape_1 Shape parameter for prior on alpha; if not provided, defaults to 0.5
#' @param alpha_shape_2 Shape parameter for prior on alpha; if not provided, defaults to 1.0
#' @param num_tree_prob Parameter for geometric prior on number of tree
#'
#' @return Returns a list containing the function arguments.
Hypers <- function(X,Y, group = NULL, alpha = 1, beta = 2, gamma = 0.95, k = 2,
                   sigma_hat = NULL, shape = 1, width = 0.1, num_tree = 20,
                   alpha_scale = NULL, alpha_shape_1 = 0.5,
                   alpha_shape_2 = 1, tau_rate = 10, num_tree_prob = NULL,
                   temperature = 1.0, weights = NULL) {

  if(is.null(alpha_scale)) alpha_scale <- ncol(X)
  if(is.null(num_tree_prob)) num_tree_prob <- 2.0 / num_tree
  if(is.null(weights)) weights <- rep(1, length(Y))

  out                                  <- list()
  out$weights                          <- weights
  out$alpha                            <- alpha
  out$beta                             <- beta
  out$gamma                            <- gamma
  out$sigma_mu                         <- 0.5 / (k * sqrt(num_tree))
  out$k                                <- k
  out$num_tree                         <- num_tree
  out$shape                            <- shape
  out$width                            <- width
  if(is.null(group)) {
    out$group                          <- 1:ncol(X) - 1
  } else {
    out$group                          <- group - 1
  }

  Y                                    <- normalize_bart(Y)
  if(is.null(sigma_hat))
    sigma_hat                          <- GetSigma(X,Y, weights = weights)

  out$sigma                            <- sigma_hat
  out$sigma_hat                        <- sigma_hat

  out$alpha_scale                      <- alpha_scale
  out$alpha_shape_1                    <- alpha_shape_1
  out$alpha_shape_2                    <- alpha_shape_2
  out$tau_rate                         <- tau_rate
  out$num_tree_prob                    <- num_tree_prob
  out$temperature                      <- temperature

  return(out)

}

#' MCMC options for SoftBart
#'
#' Creates a list which provides the parameters for running the Markov chain.
#'
#' @param num_burn Number of warmup iterations for the chain.
#' @param num_thin Thinning interval for the chain.
#' @param num_save The number of samples to collect; in total, num_burn + num_save * num_thin iterations are run
#' @param num_print Interval for how often to print the chain's progress
#' @param update_sigma_mu If true, sigma_mu/k are updated, with a half-Cauchy prior on sigma_mu centered at the initial guess
#' @param update_sigma If true, sigma is updated, with a half-Cauchy prior on sigma centered at the initial guess
#' @param update_s If true, s is updated using the Dirichlet prior.
#' @param update_alpha If true, alpha is updated using a scaled beta prime prior
#' @param update_beta If true, beta is updated using a Normal(0,2^2) prior
#' @param update_gamma If true, gamma is updated using a Uniform(0.5, 1) prior
#' @param update_tau If true, tau is updated for each tree
#' @param update_tau_mean If true, the mean of tau is updated
#'
#' @return Returns a list containing the function arguments
Opts <- function(num_burn = 2500, num_thin = 1, num_save = 2500, num_print = 100,
                 update_sigma_mu = TRUE, update_s = TRUE, update_alpha = TRUE,
                 update_beta = FALSE, update_gamma = FALSE, update_tau = TRUE,
                 update_tau_mean = FALSE, update_sigma = TRUE) {
  out <- list()
  out$num_burn        <- num_burn
  out$num_thin        <- num_thin
  out$num_save        <- num_save
  out$num_print       <- num_print
  out$update_sigma_mu <- update_sigma_mu
  out$update_s         <- update_s
  out$update_alpha    <- update_alpha
  out$update_beta     <- update_beta
  out$update_gamma    <- update_gamma
  out$update_tau      <- update_tau
  out$update_tau_mean <- update_tau_mean
  # out$update_num_tree <- update_num_tree
  out$update_num_tree <- FALSE
  out$update_sigma    <- update_sigma

  return(out)

}

normalize_bart <- function(y) {
  a <- min(y)
  b <- max(y)
  y <- (y - a) / (b - a) - 0.5
  return(y)
}

unnormalize_bart <- function(z, a, b) {
  y <- (b - a) * (z + 0.5) + a
  return(y)
}

#' Fits the SoftBart model
#'
#' Runs the Markov chain for the softbart model and collects the output
#'
#' @param X NxP matrix of training data covariates
#' @param Y Nx1 vector of training data responses
#' @param X_test NxP matrix of test data covariates
#' @param hypers List of hyperparameter values obtained from Hypers function
#' @param opts List of MCMC chain settings obtained from Opts function
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item y_hat_train: fit to the training data for each iteration of the chain
#'   \item y_hat_test: fit to the testing data for each iteration of the chain
#'   \item y_hat_train_mean: fit to the training data, averaged over iterations
#'   \item y_hat_test_mean: fit to the testing data, averaged over iterations
#'   \item sigma: posterior samples of the error standard deviations
#'   \item sigma_mu: posterior samples of sigma_mu, the standard deviation of the leaf node parameters
#'   \item s: posterior samples of s
#'   \item alpha: posterior samples of alpha
#'   \item beta: posterior samples of beta
#'   \item gamma: posterior samples of gamma
#'   \item k: posterior samples of k = 0.5 / (sqrt(num_tree) * sigma_mu)
#' }
softbart <- function(X, Y, X_test, hypers = NULL, opts = Opts()) {

  if(is.null(hypers)){
    hypers <- Hypers(X,Y)
  }

  ## Normalize Y
  Z <- normalize_bart(Y)

  ## Quantile normalize X
  n <- nrow(X)
  idx_train <- 1:n
  X_trans <- rbind(X, X_test)

  if(is.data.frame(X_trans)) {
    print("Preprocessing data frame")
    preproc_df <- preprocess_df(X_trans)
    X_trans <- preproc_df$X
    print("Using default grouping; if this is not desired, preprocess data frame manually using preprocess_df before calling.")
    hypers$group
    hypers$group <- preproc_df$group
  }

  X_trans <- quantile_normalize_bart(X_trans)
  X <- X_trans[idx_train,,drop=FALSE]
  X_test <- X_trans[-idx_train,,drop=FALSE]

  fit <- SoftBart(X,Z,X_test,
                  hypers$group,
                  hypers$alpha,
                  hypers$beta,
                  hypers$gamma,
                  hypers$sigma,
                  hypers$shape,
                  hypers$width,
                  hypers$num_tree,
                  hypers$sigma_hat,
                  hypers$k,
                  hypers$alpha_scale,
                  hypers$alpha_shape_1,
                  hypers$alpha_shape_2,
                  hypers$tau_rate,
                  hypers$num_tree_prob,
                  hypers$temperature,
                  hypers$weights,
                  opts$num_burn,
                  opts$num_thin,
                  opts$num_save,
                  opts$num_print,
                  opts$update_sigma_mu,
                  opts$update_s,
                  opts$update_alpha,
                  opts$update_beta,
                  opts$update_gamma,
                  opts$update_tau,
                  opts$update_tau_mean,
                  opts$update_num_tree,
                  opts$update_sigma)


  a <- min(Y)
  b <- max(Y)

  fit$y_hat_train <- unnormalize_bart(fit$y_hat_train, a, b)
  fit$y_hat_test <- unnormalize_bart(fit$y_hat_test, a, b)
  fit$sigma <- (b - a) * fit$sigma
  fit$k <- 0.5 / (sqrt(hypers$num_tree) * fit$sigma_mu)

  fit$y_hat_train_mean <- colMeans(fit$y_hat_train)
  fit$y_hat_test_mean <- colMeans(fit$y_hat_test)

  fit$y <- Y

  class(fit) <- "softbart"

  return(fit)

}

TreeSelect <- function(X,Y, X_test, hypers = NULL, tree_start = 25, opts = Opts()) {

  if(is.null(hypers)){
    hypers <- Hypers(X,Y)
  }

  best <- 0;

  hypers$num_tree <- tree_start
  fit <- softbart(X,Y,X_test,hypers, opts)
  best <- mean(fit$loglik) + hypers$num_tree * log(1 - hypers$num_tree_prob)

  while(TRUE) {
    tree_old <- hypers$num_tree
    tree_new <- 2 * tree_old
    hypers$num_tree <- tree_new
    fit <- softbart(X,Y,X,hypers, opts)
    gof <- mean(fit$loglik) + hypers$num_tree * log(1 - hypers$num_tree_prob)
    if(gof < best) {
      break
    }
    best <- gof
  }

  hypers$num_tree <- tree_old
  hypers$temperature <- 1.0
  fit <- softbart(X,Y,X_test,hypers, opts)

  return(list(num_tree = tree_old, fit = fit))
}

GetSigma <- function(X,Y, weights = NULL) {
  
  if(is.null(weights)) weights <- rep(1, length(Y))
  stopifnot(is.matrix(X) | is.data.frame(X))

  if(is.data.frame(X)) {
    X <- model.matrix(~.-1, data = X)
  }


  fit <- cv.glmnet(x = X, y = Y, weights = weights)
  fitted <- predict(fit, X)
  sigma_hat <- sqrt(mean((fitted - Y)^2))
  # sigma_hat <- 0
  # if(nrow(X) > 2 * ncol(X)) {
  #   fit <- lm(Y ~ X)
  #   sigma_hat <- summary(fit)$sigma
  # } else {
  #   sigma_hat <- sd(Y)
  # }

  return(sigma_hat)

}

MakeForest <- function(hypers, opts) {
  mf <- Module(module = "mod_forest", PACKAGE = "SoftBart")
  return(new(mf$Forest, hypers, opts))
}
