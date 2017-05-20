plot.softbart <- function(fit, plquants = c(0.05, 0.95), ...) {

  library(scales)
  cols <- c(muted("blue", 60, 80), muted("green"))


  par(mfrow = c(1,2))

  plot(fit$sigma, type = 'l', ylab = 'sigma', ..., col = muted("green", l = 80, c= 60))

  ql <- apply(fit$y_hat_train, 2, quantile, probs = plquants[1])
  qm <- apply(fit$y_hat_train, 2, quantile, probs = 0.5)
  qu <- apply(fit$y_hat_train, 2, quantile, probs = plquants[2])
  plot(fit$y, qm , ylim = range(ql, qu), xlab = 'y',
       ylab = "posterior interval for E(Y|x)", ...)
  for(i in 1:length(qm)) {
    lines(x = c(fit$y[i], fit$y[i]), y = c(ql[i], qu[i]), col = alpha(cols[1], .7))
    abline(0,1, lty = 2, col = cols[2])
  }

  print(mean(ql < fit$y & qu > fit$y))

}
