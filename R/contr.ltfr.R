#' Create a Full Set of Dummy Variables
#'
#' This function should not be used by the user. This package uses the caret
#' function predict.dummyVars; for technical reasons, in order for this
#' to work properly we need to import the function contr.ltfr from caret and
#' export our own function that calls contr.ltfr from caret.
#'
#' @param ... Arguments passed to caret::contr.ltfr
#'
#' @return A matrix produced by caret::contr.ltfr 
#' @export
contr.ltfr <- function(...) {
  caret::contr.ltfr(...)
}