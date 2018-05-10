#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <RcppArmadillo.h>

int sample_class(const arma::vec& probs);
int sample_class(int n);
double logit(double x);
double expit(double x);
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision);
arma::mat choll(const arma::mat& Sigma);
double log_sum_exp(const arma::vec& x);
double rlgam(double shape);
arma::vec rdirichlet(const arma::vec& shape);


#endif
