trank <- function(x) {
  x_unique <- unique(x)
  x_ranks <- rank(x_unique, ties.method = "max")
  tx <- x_ranks[match(x,x_unique)] - 1

  tx <- tx / length(unique(tx))
  tx <- tx / max(tx)

  return(tx)
}

# trank <- function(x) {
#   # x_unique <- unique(x)
#   x_ranks <- rank(x, ties.method = "min")
#   tx <- x_ranks - 1
# 
#   tx <- tx / length(tx)
#   tx <- tx / max(tx)
# 
#   return(tx)
# }

#' Quantile normalization for predictors
#' 
#' Performs a quantile normalization to each column of the matrix \code{X}.
#'
#' @param X A design matrix, should not include a column for the intercept.
#'
#' @return A matrix \code{X_norm} such that each column gives the associated
#'   empirical quantile of each observation for each predictor.
#'
#' @examples
#' X <- matrix(rgamma(100 * 10, shape = 2), nrow = 100)
#' X <- quantile_normalize_bart(X)
#' summary(X)
#' 
quantile_normalize_bart <- function(X) {
  apply(X = X, MARGIN = 2, trank)
}

#' Preprocess a dataset for use with SoftBart
#' 
#' Preprocesses a data frame for use with \code{softbart}; not needed with other
#' model fitting functions, but may also be useful when designing custom methods
#' with \code{MakeForest}. Returns a data matrix X that will work with
#' categorical predictors, and a vector of group indicators; this is required to
#' get sensible variable selection for categorical variables, and should be
#' passed in as the group argument to \code{Hypers}.
#'
#' @param X A data frame, possibly containing categorical variables stored as
#'   factors.
#'
#' @return A list containing two elements.
#' \itemize{
#'   \item \code{X}: a matrix consisting of the columns of the input data frame,
#'   with separate columns for the different levels of categorical variables.
#'   \item \code{group}: a vector of group memberships of the predictors in
#'   \code{X}, to be passed as an argument to \code{Hypers}.
#' }
#'
#' @examples
#' data(iris)
#' preprocess_df(iris)
#' 
preprocess_df <- function(X) {
  stopifnot(is.data.frame(X))

  X <- model.matrix(~.-1, data = X)
  group <- attr(X, "assign") - 1

  return(list(X = X, group = group))

}
