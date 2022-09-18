#' Root mean squared error
#'
#' Computes the root mean-squared error between y and yhat, given by 
#' sqrt(mean((y - yhat)^2)).
#'
#' @param y the realized outcomes
#' @param yhat the predicted outcomes
#'
#' @return Returns the root mean-squared error.
#'
#' @examples
#' 
#' rmse(c(1,1,1), c(1,0,2))
rmse <- function(y, yhat) {
  return(sqrt(mean((y-yhat)^2)))
}