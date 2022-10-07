#' SoftBart Varying Coefficient Regression
#'
#' Fits a semiparametric varying coefficient regression model with the
#' nonparametric slope and intercept \deqn{Y = \alpha(X) + Z \beta(X) +
#' \epsilon}{Y = alpha + Z * beta(X) + epsilon} using a soft BART model.
#'
#' @param formula A model formula with a numeric variable on the left-hand-side and non-linear predictors on the right-hand-side.
#' @param linear_var_name A string containing the variable in the data that is to be treated linearly.
#' @param data A data frame consisting of the training data.
#' @param test_data A data frame consisting of the testing data.
#' @param num_tree The number of trees in the ensemble to use.
#' @param k Determines the standard deviation of the leaf node parameters, which
#'   is given by \code{3 / k / sqrt(num_tree)} (intercept) and defaults to
#'   \code{1/k/sqrt(num_tree)} (slope). This can be modified for the slope by
#'   specifying your own hyperparameter.
#' @param hypers_intercept A list of hyperparameters constructed from the \code{Hypers()} function (\code{num_tree}, \code{k}, and \code{sigma_mu} are overridden by this function).
#' @param hypers_slope A list of hyperparameters constructed from the \code{Hypers()} function (\code{num_tree} is overridden by this function).
#' @param opts A list of options for running the chain constructed from the \code{Opts()} function (\code{update_sigma} is overridden by this function).
#' @param verbose If \code{TRUE}, progress of the chain will be printed to the console.
#' @param warn If \code{TRUE}, remind the user that they probably don't want the linear term to be included in the formula for the nonlinear part.
#'
#' @return Returns a list with the following components
#' \itemize{
#'   \item \code{sigma_mu_alpha}: samples of the standard deviation of the leaf node parameters for the intercept.
#'   \item \code{sigma_mu_beta}: samples of the standard deviation of the leaf node parameters for the slope.
#'   \item \code{sigma}: samples of the error standard deviation.
#'   \item \code{var_counts_alpha}: a matrix with a column for each predictor group containing the number of times each predictor is used in the ensemble at each iteration for the intercept.
#'   \item \code{var_counts_beta}: a matrix with a column for each predictor group containing the number of times each predictor is used in the ensemble at each iteration for the slope.
#'   \item \code{alpha_train}: samples of the nonparametric intercept evaluated on the training set.
#'   \item \code{alpha_test}: samples of the nonparametric intercept evaluated on the test set.
#'   \item \code{beta_train}: samples of the nonparametric slope evaluated on the training set.
#'   \item \code{beta_test}: samples of the nonparametric slope evaluated on the test set.
#'   \item \code{mu_train}: samples of the predictions evaluated on the training set.
#'   \item \code{mu_test}: samples of the predictions evaluated on the test set.
#'   \item \code{formula}: the formula specified by the user.
#'   \item \code{ecdfs}: empirical distribution functions, used by the \code{predict} function.
#'   \item \code{opts}: the options used when running the chain.
#'   \item \code{mu_Y, sd_Y}: used with the \code{predict} function to transform predictions.
#'   \item \code{alpha_forest}: a forest object for the intercept; see the \code{MakeForest} documentation for more details.
#'   \item \code{beta_forest}: a forest object for the slope; see the \code{MakeForest} documentation for more details.
#' }
#' @export
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
#'   Z <- rnorm(n_train)
#'   r <- f_fried(X)
#'   mu <- Z * r
#'   X_test <- matrix(runif(n_test * P), nrow = n_test)
#'   Z_test <- rnorm(n_test)
#'   r_test <- f_fried(X_test)
#'   mu_test <- Z_test * r_test
#'   Y <- mu + sigma * rnorm(n_train)
#'   Y_test <- mu + sigma * rnorm(n_test)
#'
#'   return(list(X = X, Y = Y, Z = Z, r = r, mu = mu, X_test = X_test, Y_test =
#'               Y_test, Z_test = Z_test, r_test = r_test, mu_test = mu_test))
#' }
#'
#' ## Simiulate dataset
#' sim_data <- gen_data(250, 250, 100, 1)
#'
#' df <- data.frame(X = sim_data$X, Y = sim_data$Y, Z = sim_data$Z)
#' df_test <- data.frame(X = sim_data$X_test, Y = sim_data$Y_test, Z = sim_data$Z_test)
#'
#' ## Fit the model
#'
#' opts <- Opts(num_burn = num_burn, num_save = num_save)
#' fitted_vc <- vc_softbart_regression(Y ~ . -Z, "Z", df, df_test, opts = opts)
#'
#' ## Plot results
#'
#' plot(colMeans(fitted_vc$mu_test), sim_data$mu_test)
#' abline(a = 0, b = 1)
#' 
vc_softbart_regression <- function(formula, linear_var_name, data, test_data,
                                   num_tree = 20, k = 2,
                                   hypers_intercept = NULL, 
                                   hypers_slope = NULL,
                                   opts = NULL, 
                                   verbose = TRUE, warn = TRUE) {

  ## Get design matricies and groups for categorical
  
  if(warn) {
    warning("Remember: you probably don't want your formula to also include the linear variable!")
  }

  dv <- dummyVars(formula, data)
  terms <- attr(dv$terms, "term.labels")
  group <- dummy_assign(dv)
  suppressWarnings({
    X_train <- predict(dv, data)
    X_test  <- predict(dv, test_data)
  })
  Y_train <- model.response(model.frame(formula, data))
  Y_test  <- model.response(model.frame(formula, test_data))
  
  Z_train <- data[[linear_var_name]]
  Z_test <- test_data[[linear_var_name]]
  

  stopifnot(is.numeric(Y_train))
  mu_Y <- mean(Y_train)
  sd_Y <- sd(Y_train)
  Y_train <- (Y_train - mu_Y) / sd_Y
  Y_test  <- (Y_test - mu_Y) / sd_Y

  ## Set up hypers
  if(is.null(hypers_intercept)) {
    hypers_intercept <- Hypers(X = X_train, Y = Y_train, normalize_Y = FALSE)
  }
  
  if(is.null(hypers_slope)) {
    hypers_slope <- hypers_intercept
    hypers_slope$sigma_mu <- 1 / k / sqrt(num_tree)
  }

  hypers_intercept$sigma_mu <- 3 / k / sqrt(num_tree)
  hypers_intercept$num_tree <- num_tree
  hypers_intercept$group <- group
  hypers_slope$group <- group
  hypers_slope$num_tree <- num_tree
  
  ## Set up opts

  if(is.null(opts)) {
    opts <- Opts()
  }
  opts$num_print <- .Machine$integer.max

  ## Normalize!

  make_01_norm <- function(x) {
    a <- min(x)
    b <- max(x)
    return(function(y) (y - a) / (b - a))
  }

  ecdfs   <- list()
  for(i in 1:ncol(X_train)) {
    ecdfs[[i]] <- ecdf(X_train[,i])
    if(length(unique(X_train[,i])) == 1) ecdfs[[i]] <- identity
    if(length(unique(X_train[,i])) == 2) ecdfs[[i]] <- make_01_norm(X_train[,i])
  }
  for(i in 1:ncol(X_train)) {
    X_train[,i] <- ecdfs[[i]](X_train[,i])
    X_test[,i] <- ecdfs[[i]](X_test[,i])
  }

  ## Make forests ----
  alpha_forest <- MakeForest(hypers_intercept, opts, FALSE)
  beta_forest  <- MakeForest(hypers_slope, opts, FALSE)

  ## Initialize output ----
  alpha_train <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  beta_train  <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  mu_train    <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  
  alpha_test <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  beta_test  <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  mu_test    <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  
  sigma_mu_alpha  <- numeric(opts$num_save)
  sigma_mu_beta   <- numeric(opts$num_save)
  sigma_save      <- numeric(opts$num_save)
  varcounts_alpha <- matrix(NA, nrow = opts$num_save, ncol = length(terms))
  varcounts_beta  <- matrix(NA, nrow = opts$num_save, ncol = length(terms))

  alpha <- as.numeric(alpha_forest$do_predict(X_train))
  beta  <- as.numeric(beta_forest$do_predict(X_train))
  sigma <- alpha_forest$get_sigma()
  
  ## Warmup ----

  pb <- progress_bar$new(
    format = "  warming up [:bar] :percent eta: :eta",
    total = opts$num_burn, clear = FALSE, width= 60)

  for(i in 1:opts$num_burn) {
    if(verbose) pb$tick()
    ## Update alpha
    R     <- Y_train - Z_train * beta
    alpha <- as.numeric(alpha_forest$do_gibbs(X_train, R, X_train, 1))
    sigma <- alpha_forest$get_sigma()
    
    ## Update beta
    beta_forest$set_sigma(sigma)
    
    R <- (Y_train - alpha) / Z_train
    beta <- as.numeric(beta_forest$do_gibbs_weighted(X_train, R, Z_train^2, X_train, 1))
  }
  
  ## Save ----

  pb <- progress_bar$new(
    format = "  saving [:bar] :percent eta: :eta",
    total = opts$num_save, clear = FALSE, width= 60)

  for(i in 1:opts$num_save) {
    if(verbose) pb$tick()
    for(j in 1:opts$num_thin) {
      ## Update alpha
      R     <- Y_train - Z_train * beta
      alpha <- as.numeric(alpha_forest$do_gibbs(X_train, R, X_train, 1))
      sigma <- alpha_forest$get_sigma()
      
      ## Update beta
      beta_forest$set_sigma(sigma)
      
      R <- (Y_train - alpha) / Z_train
      beta <- as.numeric(beta_forest$do_gibbs_weighted(X_train, R, Z_train^2, X_train, 1))
    }

    alpha_train[i,] <- alpha * sd_Y + mu_Y
    beta_train[i,]  <- beta * sd_Y
    mu_train[i,] <- alpha_train[i,] + beta_train[i,] * Z_train
    
    alpha_test[i,] <- as.numeric(alpha_forest$do_predict(X_test)) * sd_Y + mu_Y
    beta_test[i,]  <- as.numeric(beta_forest$do_predict(X_test)) * sd_Y
    mu_test[i,]    <- alpha_test[i,] + beta_test[i,] * Z_test
    
    sigma_save[i] <- sigma * sd_Y
    sigma_mu_alpha[i] <- alpha_forest$get_sigma_mu() * sd_Y
    sigma_mu_beta[i] <- beta_forest$get_sigma_mu() * sd_Y
    varcounts_alpha[i,] <- alpha_forest$get_counts()
    varcounts_beta[i,] <- beta_forest$get_counts()
    
  }
  
  colnames(varcounts_alpha) <- terms
  colnames(varcounts_beta) <- terms
  
  out <- list(sigma_mu_alpha = sigma_mu_alpha, sigma_mu_beta = sigma_mu_beta, 
              var_counts_alpha = varcounts_alpha, 
              var_counts_beta = varcounts_beta, sigma = sigma_save,
              alpha_train = alpha_train, alpha_test = alpha_test,
              mu_train = mu_train, mu_test = mu_test,
              beta_train = beta_train, beta_test = beta_test,
              opts = opts, formula = formula, ecdfs = ecdfs, 
              mu_Y = mu_Y, sd_Y = sd_Y, alpha_forest = alpha_forest, 
              beta_forest = beta_forest, 
              dv = dv)

  class(out) <- "vc_softbart_regression"

  return(out)
}
