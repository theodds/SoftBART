partial_dependence_regression <- function(fit, test_data, var_str, grid) {
  out <- list()
  out_mu <- list()
  for(i in 1:length(grid)) {
    newdata <- test_data
    newdata[[var_str]] <- grid[[i]]
    preds <- predict.softbart_regression(object = fit, newdata = newdata)
    out[[i]] <- data.frame(sample = 1:length(preds$mu_mean), 
                           mu = rowMeans(preds$mu))
    out[[i]][[var_str]] <- grid[[i]]
    out_mu <- rowMeans(preds$mu)
  }
  
  out_list <- list(pred_df = do.call(rbind, out), mu = out_mu, grid = grid)
  
  return(out_list)
}