dummy_assign <- function(dummy) {
  terms <- attr(dummy$terms, "term.labels")
  group <- list()
  j     <- 0
  for(k in terms) {
    if(k %in% dummy) {
      group[[k]] <- rep(j, length(dv$lvls[[k]]))
    } else {
      group[[k]] <- j
    }
    j <- j + 1
  }
  return(do.call(c, group))
}

softbart_probit <- function(formula, data, test_data, num_tree = 20,
                            k = 2, hypers = NULL, opts = NULL) {
  
  ## Get design matricies and groups for categorical
  
  dv <- dummyVars(formula, data)
  terms <- attr(dv$terms, "term.labels")
  group <- dummy_assign(dv)
  suppressWarnings({
    X_train <- predict(dummyVars(formula, data = data), data)
    X_test  <- predict(dummyVars(formula, data = test_data), test_data)
  })
  Y_train <- model.response(model.frame(formula, data))
  Y_test  <- model.response(model.frame(formula, test_data))
  
  stopifnot(is.factor(Y_train))
  stopifnot(length(levels(Y_train)) == 2)
  Y_train <- as.numeric(Y_train) - 1
  Y_test  <- as.numeric(Y_test) - 1
  
  ## Set up hypers
  
  if(is.null(hypers)) {
    hypers <- Hypers(X = X_train, Y = Y_train)
  }
  hypers$sigma_mu = 3 / k / sqrt(num_tree)
  hypers$sigma <- 1
  hypers$sigma_hat <- 1
  hypers$num_tree <- num_tree
  hypers$group <- group
  
  ## Set up opts
  
  if(is.null(opts)) {
    opts <- Opts()
  }
  opts$update_sigma <- FALSE
  opts$num_print = Inf
  
  ## Normalize!
  
  make_01_norm <- function(x) {
    a <- min(x)
    b <- max(x)
    return(function(y) (y - a) / (b - a))
  }
  
  ecdfs   <- list()
  for(i in 1:ncol(X_train)) {
    ecdfs[[i]] <- ecdf(X_train[,i])
    if(length(unique(X_train[,i])) == 1) ecdfs[[i]] <- identity
    if(length(unique(X_train[,i])) == 2) ecdfs[[i]] <- make_01_norm(X_train[,i])
  }
  for(i in 1:ncol(X_train)) {
    X_train[,i] <- ecdfs[[i]](X_train[,i])
    X_test[,i] <- ecdfs[[i]](X_test[,i])
  }
  
  ## Make forest ----
  
  probit_forest <- MakeForest(hypers, opts)
  
  ## Initialize Z
  
  mu <- probit_forest$do_predict(X_train) %>% as.numeric()
  lower <- ifelse(Y_train == 0, -Inf, 0)
  upper <- ifelse(Y_train == 0, 0, Inf)
  
  ## Initialize output
  
  mu_train  <- matrix(NA, nrow = opts$num_save, ncol = length(Y_train))
  mu_test   <- matrix(NA, nrow = opts$num_save, ncol = length(Y_test))
  sigma_mu  <- numeric(opts$num_save)
  varcounts <- matrix(NA, nrow = opts$num_save, ncol = length(terms))
  
  
  ## Warmup
  
  pb <- progress_bar$new(
    format = "  warming up [:bar] :percent eta: :eta",
    total = opts$num_burn, clear = FALSE, width= 60)
  
  for(i in 1:opts$num_burn) {
    pb$tick()
    ## Sample Z
    Z <- rtruncnorm(n = length(Y_train), a = lower, b = upper, 
                    mean = mu, sd = 1)
    ## Update R
    mu <- probit_forest$do_gibbs(X_train, Z, X_train, 1)
  }
  
  ## Save
  
  pb <- progress_bar$new(
    format = "  saving [:bar] :percent eta: :eta",
    total = opts$num_save, clear = FALSE, width= 60)
  
  for(i in 1:opts$num_save) {
    pb$tick()
    for(j in 1:opts$num_thin) {
      ## Sample Z
      Z <- rtruncnorm(n = length(Y_train), a = lower, b = upper, 
                      mean = mu, sd = 1)
      ## Update R
      mu <- probit_forest$do_gibbs(X_train, Z, X_train, 1)
    }
    
    sigma_mu[i]   <- probit_forest$get_sigma_mu()
    varcounts[i,] <- probit_forest$get_counts()
    mu_train[i,]  <- mu
    mu_test[i,]   <- probit_forest$do_predict(X_test)
    
  }
  
  return(list(sigma_mu = sigma_mu, varcounts = varcounts, mu_train = mu_train, 
              mu_test = mu_test, forest = probit_forest))
  
}
