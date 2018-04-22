find_interaction <- function(fit, i,j) {
  detect <- function(z) (i %in% z & j %in% z)
  find_within <- function(w) max(sapply(w, detect))
  return(sapply(fit$interactions, find_within))
}