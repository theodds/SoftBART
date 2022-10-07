predict.Rcpp_Forest <- function(object, X, include_burnin = FALSE, ...) {
  
}

predict_glmnet <- function (object, newx, s = c("lambda.1se", "lambda.min"), ...) 
{
  if (is.numeric(s)) 
    lambda = s
  else if (is.character(s)) {
    s = match.arg(s)
    lambda = object[[s]]
    names(lambda) = s
  }
  else stop("Invalid form for s")
  predict(object$glmnet.fit, newx, s = lambda, ...)
}

#' Predict for SoftBart Regression
#'
#' Computes predictions from a \code{softbart_regression} object on new data.
#' 
#' @param object A \code{softbart_regression} object obtained as output of the \code{softbart_regression()} function.
#' @param newdata A dataset to construct predictions on.
#' @param iterations The iterations to get predictions on; includes all of iterations including burn-in and thinning iterations. Defaults to the saved iterations, running from \code{(num_burn + num_thin):(num_burn + num_thin * num_save)}.
#' @param ... Other arguments passed to predict.
#'
#' @return A list containing 
#' \itemize{
#'   \item \code{mu}: samples of the predicted value for each observation and iteration.
#'   \item \code{mu_mean}: posterior predicted values for each observation.
#' }
#' @export
#'
predict.softbart_regression <- function(object, newdata, iterations = NULL, ...) {
  stopifnot(class(object) == "softbart_regression")
  stopifnot(object$opts$cache_trees)
  
  form <- object$formula
  opts <- object$opts
  # if(is.null(iterations)) iterations <- (opts$num_burn+1):(opts$num_save+opts$num_burn)
  if(is.null(iterations)) 
    iterations <- seq(from = opts$num_burn+opts$num_thin, 
                      to = opts$num_burn + opts$num_thin * opts$num_save, 
                      by = opts$num_thin)
  
  suppressWarnings({
    X <- predict(object$dv, newdata)
  })
  for(i in 1:ncol(X)) {
    X[,i] <- object$ecdfs[[i]](X[,i])
  }
  
  pi <- function(i) {
    as.numeric(object$forest$predict_iteration(X, i)) * object$sd_Y + object$mu_Y
  }
  
  mu <- t(sapply(iterations, pi))
  mu_hat <- colMeans(mu)
  
  return(list(mu = mu, mu_mean = mu_hat))
}

#' Predict for SoftBart Probit Regression
#'
#' Computes predictions from a \code{softbart_probit} object for new data.
#' 
#' @param object A \code{softbart_probit} object obtained as output of the \code{softbart_probit} function.
#' @param newdata A dataset to construct predictions on.
#' @param iterations The iterations get predictions on; includes all of iterations including burn-in and thinning iterations. Defaults to the saved iterations, running from \code{(num_burn + num_thin):(num_burn + num_thin * num_save)}.
#' @param ... Other arguments passed to predict.
#'
#' @return A list containing 
#' \itemize{
#'   \item \code{mu}: samples of the nonparametric function for each observation, where \code{pnorm(mu)} is the success probability.
#'   \item \code{mu_mean}: posterior mean of mu.
#'   \item \code{p}: samples of the success probability \code{pnorm(mu)} for each observation.
#'   \item \code{p_mean}: posterior mean of \code{p}.
#' }
#' @export
#'
predict.softbart_probit <- function(object, newdata, iterations = NULL, ...) {
  stopifnot(class(object) == "softbart_probit")
  stopifnot(object$opts$cache_trees)
  
  form <- object$formula
  opts <- object$opts
  # if(is.null(iterations)) iterations <- (opts$num_burn+1):(opts$num_save+opts$num_burn)
  if(is.null(iterations)) 
    iterations <- seq(from = opts$num_burn+opts$num_thin, 
                      to = opts$num_burn + opts$num_thin * opts$num_save, 
                      by = opts$num_thin)
  
  suppressWarnings({
    X <- predict(object$dv, newdata)
  })
  for(i in 1:ncol(X)) {
    X[,i] <- object$ecdfs[[i]](X[,i])
  }
  
  pi <- function(i) {
    as.numeric(object$forest$predict_iteration(X, i)) + 
      object$offset
  }
  
  mu <- t(sapply(iterations, pi))
  p <- pnorm(mu)
  
  mu_hat <- colMeans(mu)
  p_hat <- colMeans(p)
  
  return(list(mu = mu, p = p, mu_mean = mu_hat, p_mean = p_hat))
}
