#' SoftBart Regression
#' 
#' Fits a semiparametric regression model with the nonparametric function
#' modeled using a SoftBart model.
#'
#' @param formula A model formula with a numeric variable on the left-hand-side and predictors on the right-hand-side.
#' @param data A data frame consisting of the training data.
#' @param test_data A data frame consisting of the testing data.
#' @param num_tree The number of trees in the ensemble to use.
#' @param k Determines the standard deviation of the leaf node parameters, which is given by \code{3 / k / sqrt(num_tree)}.
#' @param hypers A list of hyperparameters constructed from the \code{Hypers()} function (\code{num_tree}, \code{k}, and \code{sigma_mu} are overridden by this function).
#' @param opts A list of options for running the chain constructed from the \code{Opts()} function (\code{update_sigma} is overridden by this function).
#' @param verbose If \code{TRUE}, progress of the chain will be printed to the console.
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item \code{sigma_mu}: samples of the standard deviation of the leaf node parameters.
#'   \item \code{sigma}: samples of the error standard deviation.
#'   \item \code{var_counts}: a matrix with a column for each predictor group containing the number of times each predictor is used in the ensemble at each iteration.
#'   \item \code{mu_train}: samples of the nonparametric function evaluated on the training set.
#'   \item \code{mu_test}: samples of the nonparametric function evaluated on the test set.
#'   \item \code{mu_train_mean}: posterior mean of \code{mu_train}.
#'   \item \code{mu_test_mean}: posterior mean of \code{mu_test}.
#'   \item \code{formula}: the formula specified by the user.
#'   \item \code{ecdfs}: empirical distribution functions, used by the \code{predict} function.
#'   \item \code{opts}: the options used when running the chain.
#'   \item \code{mu_Y, sd_Y}: used with the predict function to transform predictions.
#'   \item \code{forest}: a forest object; see the \code{MakeForest} documentation for more details.
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
#'   mu <- f_fried(X)
#'   X_test <- matrix(runif(n_test * P), nrow = n_test)
#'   mu_test <- f_fried(X_test)
#'   Y <- mu + sigma * rnorm(n_train)
#'   Y_test <- mu + sigma * rnorm(n_test)
#'   
#'   return(list(X = X, Y = Y, mu = mu, X_test = X_test, Y_test = Y_test, 
#'               mu_test = mu_test))
#' }
#' 
#' ## Simiulate dataset
#' sim_data <- gen_data(250, 250, 100, 1)
#' 
#' df <- data.frame(X = sim_data$X, Y = sim_data$Y)
#' df_test <- data.frame(X = sim_data$X_test, Y = sim_data$Y_test)
#' 
#' ## Fit the model
#' 
#' opts <- Opts(num_burn = num_burn, num_save = num_save)
#' fitted_reg <- softbart_regression(Y ~ ., df, df_test, opts = opts)
#' 
#' ## Plot results
#' 
#' plot(colMeans(fitted_reg$mu_test), sim_data$mu_test)
#' abline(a = 0, b = 1)
softbart_regression <- function(formula, data, test_data, num_tree = 20, k = 2, 
                                hypers = NULL, opts = NULL, verbose = TRUE) {
  
  ## Get design matricies and groups for categorical

  char_cols <- sapply(data, is.character)
  data[char_cols] <- lapply(data[char_cols], factor)
  char_cols <- sapply(test_data, is.character)
  test_data[char_cols] <- lapply(test_data[char_cols], factor)

  dv <- dummyVars(formula, data)
  terms <- attr(dv$terms, "term.labels")
  group <- dummy_assign(dv)
  suppressWarnings({
    X_train <- predict(dv, data)
    X_test  <- predict(dv, test_data)
  })
  Y_train <- model.response(model.frame(formula, data))
  Y_test  <- model.response(model.frame(formula, test_data))
  
  stopifnot(is.numeric(Y_train))
  mu_Y <- mean(Y_train)
  sd_Y <- sd(Y_train)
  Y_train <- (Y_train - mu_Y) / sd_Y
  Y_test  <- (Y_test - mu_Y) / sd_Y
  
  ## Set up hypers
  if(is.null(hypers)) {
    hypers <- Hypers(X = X_train, Y = Y_train, normalize_Y = FALSE)
  }
  
  hypers$sigma_mu = 3 / k / sqrt(num_tree)
  hypers$num_tree <- num_tree
  hypers$group <- group
  
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
  
  ## Make forest ----
  reg_forest <- MakeForest(hypers, opts, FALSE)
  
  ## Initialize output
  
  mu_train  <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  mu_test   <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  sigma_mu  <- numeric(opts$num_save)
  sigma     <- numeric(opts$num_save)
  varcounts <- matrix(NA, nrow = opts$num_save, ncol = length(terms))  
  
  ## Warmup
  
  pb <- progress_bar$new(
    format = "  warming up [:bar] :percent eta: :eta",
    total = opts$num_burn, clear = FALSE, width= 60)  
  
  for(i in 1:opts$num_burn) {
    if(verbose) pb$tick()
    ## Update R
    mu <- reg_forest$do_gibbs(X_train, Y_train, X_train, 1)
  }
  
  ## Save
  
  pb <- progress_bar$new(
    format = "  saving [:bar] :percent eta: :eta",
    total = opts$num_save, clear = FALSE, width= 60)
  
  for(i in 1:opts$num_save) {
    if(verbose) pb$tick()
    for(j in 1:opts$num_thin) {
      mu <- reg_forest$do_gibbs(X_train, Y_train, X_train, 1)
    }
    
    sigma_mu[i]   <- reg_forest$get_sigma_mu() * sd_Y
    sigma[i]      <- reg_forest$get_sigma() * sd_Y
    varcounts[i,] <- reg_forest$get_counts()
    mu_train[i,]  <- mu * sd_Y + mu_Y
    mu_test[i,]   <- reg_forest$do_predict(X_test) * sd_Y + mu_Y
    
  }
  
  colnames(varcounts) <- terms
  
  out <- list(sigma_mu = sigma_mu, var_counts = varcounts, sigma = sigma, 
              mu_train = mu_train, mu_test = mu_test, 
              mu_train_mean = colMeans(mu_train), 
              mu_test_mean = colMeans(mu_test),
              opts = opts, formula = formula, ecdfs = ecdfs,
              mu_Y = mu_Y, sd_Y = sd_Y,
              forest = reg_forest,
              dv = dv)
  
  class(out) <- "softbart_regression"
  
  return(out)
  
}
