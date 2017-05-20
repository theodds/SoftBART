library(BayesTree)
library(tidyverse)
library(SoftBart)
library(TMisc)
library(randomForest)

source("FunctionsForCV.R")
h <- function(x) exp(mean(log(x$results)))

SEED <- 1234
R <- 20
set.seed(SEED)
seed <- sample(1:99999, R)

wipp <- read.csv("datasets/WIPP.csv")
bbb <- read.csv("datasets/r_bbb.csv")
triazines <- read.csv("datasets/triazines.data")

f_compare_all <- function(X, Y, seed, K = 5) {
  out <- list()
  R <- length(seed)
  cv_f <- function(f) CV(f = f, x = X, y = Y, K = K, R = R, seed = seed)
  
  out$rf <- cv_f(f_rf)
  out$lasso <- cv_f(f_lasso)
  out$xgb <- cv_f(f_xgb)
  out$bm <- cv_f(f_bm)
  out$softbart <- cv_f(f_softbart)
  out$bart <- cv_f(f_bart)
  
     
  return(out)
}

## Wipp: DONE ----

# X_wipp <- as.matrix(wipp %>% select(-wipp_y))
# Y_wipp <- wipp$wipp_y
# 
# results_wipp <- f_compare_all(X_wipp, Y_wipp, seed = seed, K = 5)
# save(results_wipp, file = "./cache/results_wipp.RData")

## BBB: DONE ---- 
# X_bbb <- as.matrix(bbb %>% dplyr::select(-logBBB))
# Y_bbb <- bbb$logBBB
# 
# results_softbart <- CV(f = f_softbart, x = X_bbb, y = Y_bbb, K = 5, R = 1, seed = 1234)
# results_bbb <- f_compare_all(X_bbb, Y_bbb, seed = seed, K = 5)
# 
# save(results_bbb, file = "./cache/results_bbb.RData")

## Triazines: DONE ----

# clean_data <- function(x) {
#   P <- ncol(x)
#   drops <- which(is.nan(x[1,]))
#   return(x[,-drops])
# }

# X_tri <- as.matrix(triazines[,-61])
# X_tri <- quantile_normalize_bart(X_tri)
# Y_tri <- triazines[,61]
# X_tri <- clean_data(X_tri)
# 
# results_tri <- f_compare_all(X = X_tri, Y = Y_tri, seed = seed, K = 5)
# save(results_tri, file = "./cache/results_tri.RData")

## AIS: DONE ----

# ais <- read.table("./datasets/ais.txt", header = TRUE)
# Y_ais <- log(ais$Ferr)
# X_ais <- preprocess_df(dplyr::select(ais, -Ferr, -Sport, -Sex))
# group_ais <- X_ais$group
# X_ais <- X_ais$X
# 
# results_ais <- f_compare_all(X = X_ais, Y = Y_ais, seed = seed, K = 5)

## Hatco: DONE ----

# hatco <- read.csv("./datasets/HATCO.csv")
# Y_hatco <- dplyr::select(hatco, X10)[,1]
# X_hatco <- as.matrix(dplyr::select(hatco, -X10))
# 
# results_hatco <- f_compare_all(X = X_hatco, Y = Y_hatco, seed = seed, K = 5)
# save(results_hatco, file = "./cache/results_hatco.RData)

## Servo ----

servo <- read.csv("./datasets/servo.data", header = FALSE)
Y_servo <- log(servo$V5)
X_servo <- dplyr::select(servo, -V5)
X_servo <- as.matrix(preprocess_df(X_servo)$X)

results_servo <- f_compare_all(X = X_servo, Y = Y_servo, seed = seed, K = 5)

save(results_servo, file = "./cache/results_servo.RData")

# 
# cvf <- function(f) CV(f = f, x = X_servo, y = Y_servo, seed = seed[1], K = 5)
# results_rf <- cvf(f_rf)
# results_glmnet <- cvf(f_lasso)
# results_bm <- cvf(f_bm)
# results_sb <- cvf(f_softbart)


## Cpu ----
# cpu <- read.csv("./datasets/datasets/r_cpu.csv")
# X_cpu <- as.matrix(dplyr::select(cpu, -ERP, -PRP, -Vendor))
# X_cpu <- matrix(as.numeric(X_cpu), nrow = nrow(X_cpu), ncol = ncol(X_cpu))
# Y_cpu <- log(cpu$PRP)

# 
# results_cpu <- f_compare_all(X = X_cpu, Y = Y_cpu, seed = seed, K = 5)
# save(results_cpu, file = "./cache/results_cpu.RData")

# cvf <- function(f) CV(f = f, x = X_cpu, y = Y_cpu, seed = seed[1], K = 5)
# results_rf <- cvf(f_rf)
# results_bm <- cvf(f_bm)
# results_sb <- cvf(f_softbart)
# results_bart <- cvf(f_bart)

## Abalone ----

# abalone <- read.csv("./datasets/Abalone.csv")
# Y_aba <- log(abalone$Rings)
# X_aba <- dplyr::select(abalone, -Rings)
# X_aba <- preprocess_df(X_aba)$X
# 
# 
# results_abalone <- f_compare_all(X = as.matrix(X_aba), Y = Y_aba, 
#                                  seed = seed, K = 5)
# 
# save(results_abalone, file = "./cache/results_abalone.RData")

# 
# cvf <- function(f) CV(f = f, x = X_aba, y = Y_aba, seed = seed[1], K = 5)
# results_bm <- cvf(f = f_bm)
# results_rf <- cvf(f = f_rf)
# X_aba <- preprocess_df(X_aba)$X
# results_sb <- cvf(f = f_softbart)


## Diamonds: DONE ----

diamonds <- read.csv("./datasets/diamonds.csv")
as_tibble(diamonds)

Y_diamonds <- log(as.numeric(diamonds$Price))
X_diamonds <- dplyr::select(diamonds, -Price)
X_diamonds <- preprocess_df(X_diamonds)$X

# foo <- CV(f = f_bart, x = X_diamonds, y = Y_diamonds, K = 5, R = 1, seed = seed[2])

# 
# results_diamonds <- f_compare_all(X = X_diamonds, 
#                                   Y = Y_diamonds, seed = seed, K = 5)
# 
# save(results_diamonds, file = "./cache/results_diamonds.RData")


## Tecator: DONE ----
# 
# tecator <- read.csv("./datasets/Tecator.csv")
# 
# X_tec <- as.matrix(dplyr::select(tecator, Spect1:PC22))
# Y_tec <- sqrt(tecator$fat)
# 
# results_tec <- f_compare_all(X = X_tec, Y = Y_tec, seed = seed, K = 5)
# save(results_tec, "./cache/results_tec.RData")
# 
# 
# cvf <- function(f) CV(f = f, x = X_tec, y = Y_tec, seed = seed[1], K = 5)
# 
# dat_xgb <- xgb.DMatrix(data = X_tec, label = Y_tec)
# foo <- xgb.cv(data = dat_xgb, nround = 1, nfold = 5)
# 
# results_rf <- cvf(f_rf)
# results_lasso <- cvf(f_lasso)
# results_bart <- cvf(f_bart)
# results_sb <- cvf(f_softbart)
# results_bm <- cvf(f_bm)
# 
# h(results_rf)
# h(results_lasso)
# h(results_bart)
# h(results_sb)


