f_softbart <- function(x,y,x_test) {
  softbart(x,y,x_test, 
           Hypers(x,y,num_tree = 50), 
           Opts(num_burn = 2500, num_save = 2500))$y_hat_test_mean
} 

f_bart <- function(x,y,x_test) {
  softbart(x,y,x_test, Hypers(x,y,width = 1E-10, num_tree=50))$y_hat_test_mean
}

f_rf <- function(x,y,x_test) {
  library(randomForest)
  predict(randomForest(x,y),x_test)
}

f_lasso <- function(x,y,x_test) {
  library(glmnet)
  fit <- cv.glmnet(x, y)
  predict(fit, x_test)
}

f_xgb <- function(x,y,x_test) {
  library(xgboost)
  library(caret)
  fit <- train(x,y,method="xgbTree")
  predict(fit, x_test)
}

f_bm <- function(x,y,x_test) {
  library(bartMachine)
  bm <- bartMachine(as.data.frame(x),y, num_burn_in = 2500, 
                    num_iterations_after_burn_in = 2500)
  return(predict(bm, as.data.frame(x_test)))
}

rrmse_results <- function(x, base) {
  
  f <- function(y) y$results / x[[base]]$results
  normed_results <- sapply(x, f)
  
  geometric_mean <- function(x) exp(mean(log(x)))
  
  out <- list(normed_results = normed_results, RRMSE = apply(normed_results, 2, geometric_mean))
  
  return(out)
  
}
