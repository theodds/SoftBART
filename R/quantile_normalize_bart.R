trank <- function(x) {
  x_unique <- unique(x)
  x_ranks <- rank(x_unique, ties.method = "max")
  tx <- x_ranks[match(x,x_unique)] - 1

  tx <- tx / length(unique(tx))
  tx <- tx / max(tx)

  return(tx)
}

# trank <- function(x) {
#   # x_unique <- unique(x)
#   x_ranks <- rank(x, ties.method = "min")
#   tx <- x_ranks - 1
# 
#   tx <- tx / length(tx)
#   tx <- tx / max(tx)
# 
#   return(tx)
# }

quantile_normalize_bart <- function(X) {
  apply(X = X, MARGIN = 2, trank)
}

preprocess_df <- function(X) {
  stopifnot(is.data.frame(X))

  X <- model.matrix(~.-1, data = X)
  group <- attr(X, "assign") - 1

  return(list(X = X, group = group))

}
