#' Create a list of hyperparameter values
#'
#' Creates a list which holds all the hyperparameters for use with the
#' model-fitting functions and with the \code{MakeForest} functionality.
#'
#' @param X A matrix of training data covariates.
#' @param Y A vector of training data responses.
#' @param group Allows for grouping of covariates with shared splitting proportions, which is useful for categorical dummy variables. For each column of \code{X}, \code{group} gives the associated group.
#' @param alpha Positive constant controlling the sparsity level.
#' @param beta Parameter penalizing tree depth in the branching process prior.
#' @param gamma Parameter penalizing new nodes in the branching process prior.
#' @param k Related to the signal-to-noise ratio, \code{sigma_mu = 0.5 / (sqrt(num_tree) * k)}. BART defaults to \code{k = 2} after applying the max/min normalization to the outcome.
#' @param sigma_hat A prior guess at the conditional variance of \code{Y} given \code{X}. If not provided, this is estimated empirically by linear regression.
#' @param shape Shape parameter for gating probabilities.
#' @param width Bandwidth of gating probabilities.
#' @param num_tree Number of trees in the ensemble.
#' @param alpha_scale Scale of the prior for \code{alpha}; if not provided, defaults to the number of predictors.
#' @param alpha_shape_1 Shape parameter for prior on \code{alpha}; if not provided, defaults to 0.5.
#' @param alpha_shape_2 Shape parameter for prior on \code{alpha}; if not provided, defaults to 1.0.
#' @param tau_rate Rate parameter for the bandwidths of the trees with an exponential prior; defaults to 10.
#' @param num_tree_prob Parameter for geometric prior on number of tree.
#' @param temperature The temperature applied to the posterior distribution; set to 1 unless you know what you are doing.
#' @param weights Only used by the function \code{softbart}, this is a vector of weights to be used in heteroskedastic regression models, with the variance of an observation given by \code{sigma_sq / weight}.
#' @param normalize_Y Do you want to compute \code{sigma_hat} after applying the standard BART max/min normalization to \eqn{(-0.5, 0.5)} for the outcome? If \code{FALSE}, no normalization is applied. This might be useful for fitting custom models where the outcome is normalized by hand.
#'
#' @return Returns a list containing the function arguments.
Hypers <- function(X,Y, group = NULL, alpha = 1, beta = 2, gamma = 0.95, k = 2,
                   sigma_hat = NULL, shape = 1, width = 0.1, num_tree = 20,
                   alpha_scale = NULL, alpha_shape_1 = 0.5,
                   alpha_shape_2 = 1, tau_rate = 10, num_tree_prob = NULL,
                   temperature = 1.0, weights = NULL, normalize_Y = TRUE) {

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

  if(normalize_Y) {
    Y                                  <- normalize_bart(Y)
  }
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
#' Creates a list that provides the parameters for running the Markov chain.
#'
#' @param num_burn Number of warmup iterations for the chain.
#' @param num_thin Thinning interval for the chain.
#' @param num_save The number of samples to collect; in total, \code{num_burn + num_save * num_thin} iterations are run.
#' @param num_print Interval for how often to print the chain's progress.
#' @param update_sigma_mu If \code{TRUE}, \code{sigma_mu} is  updated, with a half-Cauchy prior on \code{sigma_mu} centered at the initial guess.
#' @param update_sigma If \code{TRUE}, \code{sigma} is updated, with a half-Cauchy prior on \code{sigma} centered at the initial guess.
#' @param update_s If \code{TRUE}, \code{s} is updated using the Dirichlet prior \eqn{s \sim D(\alpha / P, \ldots, \alpha / P)} where \eqn{P} is the number of covariates.
#' @param update_alpha If \code{TRUE}, \code{alpha} is updated using a scaled beta prime prior.
#' @param update_beta If \code{TRUE}, \code{beta} is updated using a normal prior with mean 0 and variance 4.
#' @param update_gamma If \code{TRUE}, gamma is updated using a Uniform(0.5, 1) prior.
#' @param update_tau If \code{TRUE}, the bandwidth \code{tau} is updated for each tree
#' @param update_tau_mean If \code{TRUE}, the mean of \code{tau} is updated
#' @param cache_trees If \code{TRUE}, we save the trees for each MCMC iteration when using the MakeForest interface
#'
#' @return Returns a list containing the function arguments.
Opts <- function(num_burn = 2500, num_thin = 1, num_save = 2500, num_print = 100,
                 update_sigma_mu = TRUE, update_s = TRUE, update_alpha = TRUE,
                 update_beta = FALSE, update_gamma = FALSE, update_tau = TRUE,
                 update_tau_mean = FALSE, update_sigma = TRUE,
                 cache_trees = TRUE) {
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
  out$cache_trees     <- cache_trees

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
#' Runs the Markov chain for the semiparametric Gaussian model \deqn{Y = r(X) +
#' \epsilon}{Y = r(X) + epsilon} and collects the output, where \eqn{r(x)}{r(x)}
#' is modeled using a soft BART model.
#'
#' @param X A matrix of training data covariates.
#' @param Y A vector of training data responses.
#' @param X_test A matrix of test data covariates
#' @param hypers A ;ist of hyperparameter values obtained from \code{Hypers} function
#' @param opts A list of MCMC chain settings obtained from \code{Opts} function
#' @param verbose If \code{TRUE}, progress of the chain will be printed to the console.
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item \code{y_hat_train}: predicted values for the training data for each iteration of the chain.
#'   \item \code{y_hat_test}: predicted values for the test data for each iteration of the chain.
#'   \item \code{y_hat_train_mean}: predicted values for the training data, averaged over iterations.
#'   \item \code{y_hat_test_mean}: predicted values for the test data, averaged over iterations.
#'   \item \code{sigma}: posterior samples of the error standard deviations.
#'   \item \code{sigma_mu}: posterior samples of \code{sigma_mu}, the standard deviation of the leaf node parameters.
#'   \item \code{s}: posterior samples of \code{s}.
#'   \item \code{alpha}: posterior samples of \code{alpha}.
#'   \item \code{beta}: posterior samples of \code{beta}.
#'   \item \code{gamma}: posterior samples of \code{gamma}.
#'   \item \code{k}: posterior samples of \code{k = 0.5 / (sqrt(num_tree) * sigma_mu)}
#'   \item \code{num_leaves_final}: the number of leaves for each tree at the final iteration.
#' }
#' 
#' @examples
#' 
#' ## NOTE: SET NUMBER OF BURN IN AND SAMPLE ITERATIONS HIGHER IN PRACTICE
#' 
#' num_burn <- 10 ## Should be ~ 5000
#' num_save <- 10 ## Should be ~ 5000
#' 
#' set.seed(1234)
#' f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
#'   10 * x[,4] + 5 * x[,5]
#' 
#' gen_data <- function(n_train, n_test, P, sigma) {
#'   X <- matrix(runif(n_train * P), nrow = n_train)
#'   mu <- f_fried(X)
#'   X_test <- matrix(runif(n_test * P), nrow = n_test)
#'   mu_test <- f_fried(X_test)
#'   Y <- mu + sigma * rnorm(n_train)
#'   Y_test <- mu_test + sigma * rnorm(n_test)
#'   
#'   return(list(X = X, Y = Y, mu = mu, X_test = X_test, Y_test = Y_test, mu_test = mu_test))
#' }
#' 
#' ## Simiulate dataset
#' sim_data <- gen_data(250, 100, 1000, 1)
#' 
#' ## Fit the model
#' fit <- softbart(X = sim_data$X, Y = sim_data$Y, X_test = sim_data$X_test, 
#'                 hypers = Hypers(sim_data$X, sim_data$Y, num_tree = 50, temperature = 1),
#'                 opts = Opts(num_burn = num_burn, num_save = num_save, update_tau = TRUE))
#' 
#' ## Plot the fit (note: interval estimates are not prediction intervals, 
#' ## so they do not cover the predictions at the nominal rate)
#' plot(fit)
#' 
#' ## Look at posterior model inclusion probabilities for each predictor. 
#' 
#' plot(posterior_probs(fit)[["post_probs"]], 
#'      col = ifelse(posterior_probs(fit)[["post_probs"]] > 0.5, scales::muted("blue"), 
#'                   scales::muted("green")), 
#'      pch = 20)
#' 
#' 
#' rmse(fit$y_hat_test_mean, sim_data$mu_test)
#' rmse(fit$y_hat_train_mean, sim_data$mu)
#' 
softbart <- function(X, Y, X_test, hypers = NULL, opts = Opts(), verbose = TRUE) {

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
  
  if(!verbose) {
    opts$num_print <- .Machine$integer.max
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
  fitted <- predict_glmnet(fit, X)
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
