#' Create an Rcpp_Forest Object
#' 
#' Make an object of type \code{Rcpp_Fores}t, which can be used to embed a soft
#' BART model into other models. Some examples are given in the package
#' vignette.
#'
#' @param hypers A list of hyperparameter values obtained from \code{Hypers()} function
#' @param opts A list of MCMC chain settings obtained from \code{Opts()} function
#' @param warn If \code{TRUE}, reminds the user to normalize their design matrix when interacting with a forest object.
#'
#' @return Returns an object of type \code{Rcpp_Forest}. If \code{forest} is an
#'   \code{Rcpp_Forest} object then it has the following methods.
#' \itemize{
#'   \item \code{forest$do_gibbs(X, Y, X_test, i)} runs \code{i} iterations of
#'   the Bayesian backfitting algorithm and predicts on the test set
#'   \code{X_test}. The state of forest is also updated.
#'   \item \code{forest$do_gibbs_weighted(X, Y, weights X_test, i)} runs \code{i}
#'   iterations of the Bayesian backfitting algorithm and predicts on the test
#'   set \code{X_test}; assumes that \code{Y} is heteroskedastic with known weights. The state
#'   of forest is also updated.
#'   \item \code{forest$do_predict(X)} returns the predictions from a matrix \code{X}
#'   of predictors.
#'   \item \code{forest$get_counts()} returns the number of times each variable
#'   has been used in a splitting rule at the current state of \code{forest}.
#'   \item \code{forest$get_s()} returns the splitting probabilities of the
#'   forest.
#'   \item \code{forest$get_sigma()} returns the error standard deviation of the
#'   forest.
#'   \item \code{forest$get_sigma_mu()} returns the standard deviation of the
#'   leaf node parameters.
#'   \item \code{forest$get_tree_counts()} returns a matrix with a row for
#'   each group of predictors and a column for each tree that counts the number of times each
#'   group of predictors is used in each tree at the current state of \code{forest}.
#'   \item \code{forest$predict_iteration(X, i)} returns the predictions from a
#'   matrix \code{X} of predictors at iteration \code{i}. Requires that \code{opts$cache_trees =
#'   TRUE} in \code{MakeForest(hypers, opts)}.
#'   \item \code{forest$set_s(s)} sets the splitting probabilities of the forest
#'   to \code{s}.
#'   \item \code{forest$set_sigma(x)} sets the error standard deviation of the
#'   forest to \code{x}.
#'   \item \code{forest$num_gibbs} returns the number of iterations in total
#'   that the Gibbs sampler has been run.
#' }
#'
#' @examples \donttest{
#' X <- matrix(runif(100 * 10), nrow = 100, ncol = 10)
#' Y <- rowSums(X) + rnorm(100)
#' my_forest <- MakeForest(Hypers(X,Y), Opts())
#' mu_hat <- my_forest$do_gibbs(X,Y,X,200)
#' }
MakeForest <- function(hypers, opts, warn = TRUE) {
  if(warn) {
    warning("Reminder: make sure to normalize the columns of your design matrix to lie between 0 and 1 when running the Bayesian backfitting algorithm or using do_predict(). THIS IS YOUR RESPONSIBILITY, YOU WILL GET NONSENSE ANSWERS IF YOU DON'T DO THIS. Set warn = FALSE to disable this warning.") 
  }
  mf <- Module(module = "mod_forest", PACKAGE = "SoftBart")
  return(new(mf$Forest, hypers, opts))
}