#' Partial Dependence Function for SoftBART Regression
#'
#' Computes the partial dependence function for a given covariate at a given set of covariate values.
#' 
#' @param fit A fitted model of type softbart_regression
#' @param test_data A data set used to form the baseline distribution of covariates for the partial dependence function
#' @param var_str A string giving the variable name of the predictor to compute the partial dependence function for
#' @param grid The values of the predictor to compute the partial dependence function at
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item pred_df: a data.frame containing columns for a MCMC iteration ID (sample), the value on the grid, and the partial dependence function value
#'   \item mu: a matrix containing the same information as pred_df, with the rows corresponding to iterations and columns corresponding to grid values
#'   \item grid: the grid used as input
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
#' sim_data <- gen_data(250, 250, 10, 1)
#' 
#' df <- data.frame(X = sim_data$X, Y = sim_data$Y)
#' df_test <- data.frame(X = sim_data$X_test, Y = sim_data$Y_test)
#' 
#' ## Fit the model
#' 
#' opts <- Opts(num_burn = num_burn, num_save = num_save)
#' fitted_reg <- softbart_regression(Y ~ ., df, df_test, opts = opts)
#' 
#' ## Compute PDP and plot
#' 
#' grid <- seq(from = 0, to = 1, length = 10)
#' pdp_x4 <- partial_dependence_regression(fitted_reg, df_test, "X.4", grid)
#' plot(pdp_x4$grid, colMeans(pdp_x4$mu))
partial_dependence_regression <- function(fit, test_data, var_str, grid) {
  out <- list()
  out_mu <- list()
  for(i in 1:length(grid)) {
    newdata <- test_data
    newdata[[var_str]] <- grid[[i]]
    preds <- predict.softbart_regression(object = fit, newdata = newdata)
    out[[i]] <- data.frame(sample = nrow(preds$mu), 
                           mu = rowMeans(preds$mu))
    out[[i]][[var_str]] <- grid[[i]]
    out_mu[[i]] <- rowMeans(preds$mu)
  }
  
  out_list <- list(pred_df = do.call(rbind, out), mu = do.call(cbind, out_mu), 
                   grid = grid)
  
  return(out_list)
}
