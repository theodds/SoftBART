#' General SoftBart Regression
#'
#' Fits the general (Soft) BART (GBART) model, which combines the BART model
#' with a linear predictor. That is, it fits the semiparametric Gaussian
#' regression model \deqn{Y = r(X) + Z^\top \beta + \epsilon} where the function
#' \eqn{r(x)} is modeled using a BART ensemble.
#'
#' @param formula A model formula with a numeric variable on the left-hand-side and non-linear predictors on the right-hand-side.
#' @param linear_formula A model formula with the linear variables on the right-hand-side (left-hand-side is not used).
#' @param data A data frame consisting of the training data.
#' @param test_data A data frame consisting of the testing data.
#' @param num_tree The number of trees used in the ensemble.
#' @param k Determines the standard deviation of the leaf node parameters, which is given by \code{3 / k / sqrt(num_tree)}.
#' @param hypers A list of hyperparameters constructed from the \code{Hypers()} function (\code{num_tree}, \code{k}, and \code{sigma_mu} are overridden by this function).
#' @param opts A list of options for running the chain constructed from the \code{Opts()} function (\code{update_sigma} is overridden by this function).
#' @param remove_intercept If \code{TRUE} then any intercept term in the linear formula will be removed, with the overall location of the outcome captured by the nonparametric function.
#' @param verbose If \code{TRUE}, progress of the chain will be printed to the console.
#' @param warn If \code{TRUE}, remind the user that they probably don't want the linear predictors to be included in the formula for the nonlinear part.
#'
#' @return Returns a list with the following components
#' \itemize{
#'   \item \code{r_train}: samples of the nonparametric function evaluated on the training set.
#'   \item \code{r_test}: samples of the nonparametric function evaluated on the test set.
#'   \item \code{eta_train}: samples of the linear predictor on the training set.
#'   \item \code{eta_test}: samples of the linear predictor on the test set.
#'   \item \code{mu_train}: samples of the prediction on the training set.
#'   \item \code{mu_test}: samples of the prediction on the test set.
#'   \item \code{beta}: samples of the regression coefficients.
#'   \item \code{sigma}: samples of the error standard deviation.
#'   \item \code{sigma_mu}: samples of the standard deviation of the leaf node parameters.
#'   \item \code{var_counts}: a matrix with a column for each nonparametric predictor containing the number of times that predictor is used in the ensemble at each iteration.
#'   \item \code{opts}: the options used when running the chain.
#'   \item \code{formula}: the formula specified by the user.
#'   \item \code{ecdfs}: empirical distribution functions, used by the predict function.
#'   \item \code{mu_Y, sd_Y}: used with the predict function to transform predictions.
#'   \item \code{forest}: a forest object for the nonlinear part; see the \code{MakeForest()} documentation for more details.
#' }
#' @export
#'
#' @examples
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
#' fitted_reg <- gsoftbart_regression(Y ~ . - X.4 - X.5, ~ X.4 + X.5, df, df_test, opts = opts)
#' 
#' ## Plot results
#' 
#' plot(colMeans(fitted_reg$mu_test), sim_data$mu_test)
#' abline(a = 0, b = 1)
#' plot(fitted_reg$beta[,1])
#' plot(fitted_reg$beta[,2])
gsoftbart_regression <- function(formula, linear_formula, data, test_data,
                                 num_tree = 20, k = 2,
                                 hypers = NULL,
                                 opts = NULL, remove_intercept = TRUE,
                                 verbose = TRUE, warn = TRUE) {

  ## Get design matricies and groups for categorical

  if(warn) {
    warning("Remember: you probably don't want your formula to also include the linear variables!")
  }

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

  Z_train <- model.matrix(linear_formula, data)
  Z_test  <- model.matrix(linear_formula, test_data)
  
  if(remove_intercept) {
    idx <- which(colnames(Z_train) != "(Intercept)")
    Z_train <- Z_train[,idx]
    Z_test <- Z_test[,idx]    
  }

  stopifnot(is.numeric(Y_train))
  mu_Y <- mean(Y_train)
  sd_Y <- sd(Y_train)
  Y_train <- (Y_train - mu_Y) / sd_Y
  Y_test  <- (Y_test - mu_Y) / sd_Y

  ## Set up hypers
  if(is.null(hypers)) {
    hypers <- Hypers(X = cbind(X_train, Z_train), 
                     Y = Y_train, normalize_Y = FALSE)
  }

  hypers$sigma_mu <- 3 / k / sqrt(num_tree)
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
  forest <- MakeForest(hypers, opts, FALSE)
  
  ## Initialize output ----
  r_train      <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  r_test       <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  eta_train    <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  eta_test     <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  mu_train     <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  mu_test      <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  beta_out     <- matrix(NA, nrow = opts$num_burn, ncol = ncol(Z_train))
  sigma_out    <- numeric(opts$num_save)
  sigma_mu_out <- numeric(opts$num_save)
  varcounts    <- matrix(NA, nrow = opts$num_save, ncol = length(terms))

  ## Prepare running the chain ----
  r     <- as.numeric(forest$do_predict(X_train))
  beta  <- numeric(ncol(Z_train))
  sigma <- forest$get_sigma()
  eta   <- numeric(nrow(Z_train))
  
  ## Function for updating beta ----
  ZtZi <- solve(t(Z_train) %*% Z_train)
  
  update_beta <- function(R, Z, V, sigma) {
    beta_hat <- as.numeric(V %*% t(Z) %*% R)
    beta <- MASS::mvrnorm(n = 1, mu = beta_hat, Sigma = sigma^2 * V)
    return(as.numeric(beta))
  }
  
  ## Warmup ----

  pb <- progress_bar$new(
    format = "  warming up [:bar] :percent eta: :eta",
    total = opts$num_burn, clear = FALSE, width= 60)

  for(i in 1:opts$num_burn) {
    if(verbose) pb$tick()
    
    ## Update beta ----
    R <- Y_train - r
    beta <- update_beta(R, Z_train, ZtZi, sigma^2)
    eta <- as.numeric(Z_train %*% beta)

    ## Update forest and sigma ----
    R <- Y_train - eta
    r <- as.numeric(forest$do_gibbs(X_train, R, X_train, 1))
    sigma <- forest$get_sigma()
  }
  
  ## Save ----

  pb <- progress_bar$new(
    format = "  saving [:bar] :percent eta: :eta",
    total = opts$num_save, clear = FALSE, width= 60)

  for(i in 1:opts$num_save) {
    if(verbose) pb$tick()
    for(j in 1:opts$num_thin) {
      ## Update beta ----
      R <- Y_train - r
      beta <- update_beta(R, Z_train, ZtZi, sigma^2)
      eta <- as.numeric(Z_train %*% beta)
      
      ## Update forest and sigma ----
      R <- Y_train - eta
      r <- as.numeric(forest$do_gibbs(X_train, R, X_train, 1))
      sigma <- forest$get_sigma()
    }
    r_train[i,] <- r * sd_Y + mu_Y
    r_test[i,] <- as.numeric(forest$do_predict(X_test)) * sd_Y + mu_Y
    eta_train[i,] <- eta * sd_Y
    eta_test[i,] <- as.numeric(Z_test %*% beta) * sd_Y
    mu_train[i,] <- r + eta
    mu_test[i,] <- r_test[i,] + eta_test[i,]
    beta_out[i,] <- beta * sd_Y
    sigma_out[i] <- sigma * sd_Y
    sigma_mu_out[i] <- forest$get_sigma_mu() * sd_Y
    varcounts[i,] <- as.numeric(forest$get_counts())
  }

  colnames(varcounts) <- terms
  
  out <- list(r_train = r_train, r_test = r_test, eta_train = eta_train, 
              eta_test = eta_test, mu_train = mu_train, mu_test = mu_test, 
              beta = beta_out, sigma = sigma_out, sigma_mu = sigma_mu_out, 
              var_counts = varcounts, opts = opts, formula = formula, 
              ecdfs = ecdfs, mu_Y = mu_Y, sd_Y = sd_Y, forest = forest, dv = dv)

  class(out) <- "gsoftbart_regression"
  return(out)
}