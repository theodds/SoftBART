Soft Bayesian Sum of Trees Models
================

## The SoftBart package

This package implements the methodology described in the paper

  - Linero, A.R. and Yang, Y. (2018). *Bayesian tree ensembles that
    adapt to smoothness and sparsity.* Journal of the Royal Statistical
    Society, Series B.

### New features on MFM Branch\!\!

Some new experimental features have been pushed to the MFM branch. These
include:

  - The Dirichlet prior has been replaced with a Gibbs prior on the
    partition of the branches, or (equivalently), a “Mixture of Finite
    Mixtures” prior. See Miller and Harrison (2018, JASA) for a
    description of these priors in the context of Bayesian nonparametric
    mixture models. In addition to (suspected) better performance in
    theory and practice, the MCMC for the Gibbs prior should (hopefully)
    mix better.

  - New Markov transistions are used, including sampling a tree from the
    prior and using a type of “Perturb” move due to Pratola (2016,
    Bayesian Analysis). This has substantial benefits for the mixing of
    the chain.

I plan to merge these changes into master once they are thoroughly
tested.

### Installation

The package can be installed with the `devtools` package:

``` r
library(devtools)
install_github("theodds/SoftBART")
```

Note that if you are on OSX you may need to install the gfortran library
from <https://cran.r-project.org/bin/macosx/tools/>.

### Usage

The package is designed to mirror the functionality of the `BayesTree`
package. The function `softbart` is the primary function, and is used in
essentially the same manner as the `bart` function in `BayesTree`.

The following is a minimal example on “Friedman’s example”:

``` r
## Load library
library(SoftBart)

## Functions used to generate fake data
set.seed(1234)
f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
                      10 * x[,4] + 5 * x[,5]
    
gen_data <- function(n_train, n_test, P, sigma) {
    X <- matrix(runif(n_train * P), nrow = n_train)
    mu <- f_fried(X)
    X_test <- matrix(runif(n_test * P), nrow = n_test)
    mu_test <- f_fried(X_test)
    Y <- mu + sigma * rnorm(n_train)
    Y_test <- mu_test + sigma * rnorm(n_test)
        
    return(list(X = X, Y = Y, mu = mu, X_test = X_test, Y_test = Y_test, mu_test = mu_test))
}

## Simiulate dataset
sim_data <- gen_data(250, 100, 1000, 1)
    
## Fit the model
fit <- softbart(X = sim_data$X, Y = sim_data$Y, X_test = sim_data$X_test, 
                hypers = Hypers(sim_data$X, sim_data$Y, num_tree = 50, temperature = 1),
                opts = Opts(num_burn = 5000, num_save = 5000, update_tau = TRUE))
    
## Plot the fit (note: interval estimates are not prediction intervals, 
## so they do not cover the predictions at the nominal rate)
plot(fit)

## Look at posterior model inclusion probabilities for each predictor. 

posterior_probs <- function(fit) colMeans(fit$var_counts > 0)
plot(posterior_probs(fit), 
     col = ifelse(posterior_probs(fit) > 0.5, muted("blue"), muted("green")), 
     pch = 20)

rmse <- function(x,y) sqrt(mean((x-y)^2))

rmse(fit$y_hat_test_mean, sim_data$mu_test)
rmse(fit$y_hat_train_mean, sim_data$mu)
```

### Accessing the model from R

In more complex settings, one may wish to incorporate the SoftBART model
as a component within a larger model. In this case, it is possible to
construct a SoftBART object within `R` and do a single Gibbs sampling
update.

``` r
hypers <- Hypers(sim_data$X, sim_data$Y)
opts <- Opts()

forest <- MakeForest(hypers, opts)
mu_hat <- forest$do_gibbs(sim_data$X, sim_data$Y, sim_data$X_test, opts$num_burn)
mu_hat <- forest$do_gibbs(sim_data$X, sim_data$Y, sim_data$X_test, opts$num_save)
```

The `do_gibbs` function takes as input the data used to do the update,
an additional set of points at which to predict, and the number of
iterations to run the sampler. By default, the probability vector `s`
will not be updated until at least `num_burn / 2` iterations have been
run. This can be checked by calling `forest$num_gibbs`. The `s` vector
itself can be obtained by calling `forest$get_s()`, and in the future we
will add features to deal with other components as well. The code above
first burns in and then samples from the posterior, and is essentially
equivalent to using `softbart`. **WARNING**: if you are going to do
this, you need to preprocess `X` and `X_test` by hand, so that all
values lie in \[0,1\]. The `softbart` function, in addition to doing the
sampling, also preprocesses `X` and `X_test` by applying a quantile
transformation.
