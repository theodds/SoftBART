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