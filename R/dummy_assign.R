dummy_assign <- function(dummy) {
  terms <- attr(dummy$terms, "term.labels")
  group <- list()
  j     <- 0
  for(k in terms) {
    if(k %in% dummy$facVars) {
      group[[k]] <- rep(j, length(dummy$lvls[[k]]))
    } else {
      group[[k]] <- j
    }
    j <- j + 1
  }
  return(do.call(c, group))
}