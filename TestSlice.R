library(SoftBart)

logsumexp <- function(x) {
  M <- max(x)
  M + log(sum(exp(x - M)))
}

test_it <- function(N = 20000, P = 200, alpha = 1) {
  sigma <- P/alpha - digamma(1)
  Z <- -sigma * rexp(P)
  logs <- Z - logsumexp(Z)
  s <- exp(logs)
  var_counts <- tabulate(sample(1:P, N, TRUE, s), P)

  zeta_0 <- rnorm(P)
  chol_factor <- diag(P)

  s_out <- matrix(NA, 2000, P)

  zeta_1 <- TestElliptical(zeta_0, chol_factor, var_counts, sigma / 2000)
  for(i in 1:2000) {
    zeta_1 <- TestElliptical(zeta_1, chol_factor, var_counts, sigma / (2000 - i + 1))
    Z_1 <- sigma * log(pnorm(zeta_1))
    logs_1 <- Z_1 - logsumexp(Z_1)
    s_out[i,] <- exp(logs_1)
  }

  return(list(s_out = s_out, s_0 = s, s_hat = colMeans(s_out)))
}

system.time(out <- test_it())

which(out$s_hat > .01)
idx <- which(out$s_0 > .01)

round(out$s_hat[idx], 3)
round(out$s_0[idx], 3)
