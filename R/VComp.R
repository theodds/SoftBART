logsumexp <- function(x) {
  M <- max(x)
  return(M + log(sum(exp(x - M))))
}

calc_log_V <- function(n, k, log_prior, k_max = 1000, eta = 1) {
  d <- k:k_max
  log_terms <- log_prior(d)
  log_terms <- log_terms + lgamma(d + 1) - lgamma(d - k + 1)
  log_terms <- log_terms + lgamma(eta * d) - lgamma(eta * d + n)
  return(logsumexp(log_terms))
}


