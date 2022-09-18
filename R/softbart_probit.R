softbart_probit <- function(formula, data, test_data, 
                            hypers = NULL, opts = NULL) {
  
  ## Get design matricies
  
  X_train <- predict(dummyVars(formula, data = data))
  X_test  <- predict(dummyVars(formula, data = test_data))
  Y_train <- model.response(model.frame(formula, data))
  Y_test  <- model.response(model.frame(formula, test_data))
  
  stopifnot(is.factor(Y_train))
  stopifnot(length(levels) == 2)
  Y_train <- as.numeric(Y_train) - 1
  Y_test  <- as.numeric(Y_test) - 1
  
  ## Set up hypers
  
  if(is.null(hypers)) {
    hypers <- Hypers(X = X_train, )
  }
  
  ## Normalize!
  
  make_01_norm <- function(x) {
    a <- min(x)
    b <- max(x)
    return(function(y) (x - a) / (b - a))
  }
  
  ecdfs   <- list()
  for(i in 1:ncol(X_train)) {
    ecdfs[[i]] <- ecdf(X_train[,i])
    if(length(unique(X_train[,i])) == 1) ecdfs[[i]] <- identity
    if(length(unique(X_train[,i])) == 2) ecdfs[[i]] <- make_01_norm(X_train[,i])
  }
  for(i in 1:ncol(X_train)) {
    X_train[,i] <- ecdfs[[i]](X_train)
    X_test[,i] <- ecdfs[[i]](X_test)
  }
  
}