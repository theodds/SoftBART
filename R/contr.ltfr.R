#' Create a Full Set of Dummy Variables
#'
#'  Used with \code{dummyVars} in the \pkg{caret} package to create a full set
#'  of dummy variables (i.e. less than full rank parameterization).
#'
#' @param ... A list of arguments.
#'
#' @return A matrix produced containing full sets of dummy variables.
#' @export
contr.ltfr <- function(...) {
  caret::contr.ltfr(...)
}