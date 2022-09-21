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

predict.softbart_probit <- function(object, newdata, iterations = NULL) {
  stopifnot(class(object) == "softbart_probit")
  stopifnot(object$opts$cache_trees)
  
  num_iter <- length(object$sigma_mu)
  form <- object$formula
  if(is.null(iterations)) iterations <- 1:num_iter
  
  suppressWarnings({
    X <- predict(dummyVars(form, data = newdata), newdata)
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
  # plot(colMeans(mu), object$mu_test_mean)
}