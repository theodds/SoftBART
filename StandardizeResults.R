standardize_results <- function(x, base) {
  
  f <- function(y) y$results / x[[base]]$results
  normed_results <- sapply(x, f)
  
  geometric_mean <- function(x) exp(mean(log(x)))
  
  out <- list(normed_results = normed_results, RRMSE = apply(normed_results, 2, geometric_mean))
  
  return(out)
  
}
  