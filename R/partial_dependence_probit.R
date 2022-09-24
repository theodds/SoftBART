#' Partial Dependence Function for SoftBART Probit Regression
#'
#' Computes the partial dependence function for a given covariate at a given set of covariate values for the probit model
#' 
#' @param fit A fitted model of type softbart_probit
#' @param test_data A data set used to form the baseline distribution of covariates for the partial dependence function
#' @param var_str A string giving the variable name of the predictor to compute the partial dependence function for
#' @param grid The values of the predictor to compute the partial dependence function at
#'
#' @return Returns a list with the following components:
#' \itemize{
#'   \item pred_df: a data.frame containing columns for a MCMC iteration ID (sample), the value on the grid, and the partial dependence function value
#'   \item p: a matrix containing the same information as pred_df, with the rows corresponding to iterations and columns corresponding to grid values
#'   \item grid: the grid used as input
#' }
#' @export
partial_dependence_probit <- function(fit, test_data, var_str, grid) {
  out <- list()
  out_mu <- list()
  for(i in 1:length(grid)) {
    newdata <- test_data
    newdata[[var_str]] <- grid[[i]]
    preds <- predict.softbart_probit(object = fit, newdata = newdata)
    out[[i]] <- data.frame(sample = nrow(preds$p),
                           p = rowMeans(preds$p))
    out[[i]][[var_str]] <- grid[[i]]
    out_mu[[i]] <- rowMeans(preds$p)
  }

  out_list <- list(pred_df = do.call(rbind, out), p = do.call(cbind, out_mu),
                   grid = grid)

  return(out_list)
}
