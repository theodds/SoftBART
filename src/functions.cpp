#include "functions.h"

int sample_class(const arma::vec& probs) {
  double U = R::unif_rand();
  double foo = 0.0;
  int K = probs.size();
  
  // Sample
  for(int k = 0; k < K; k++) {
    foo += probs(k);
    if(U < foo) {
      return(k);
    }
  }
  return K - 1;
}

int sample_class(int n) {
  double U = R::unif_rand();
  double p = 1.0 / ((double)n);
  double foo = 0.0;
  
  for(int k = 0; k < n; k++) {
    foo += p;
    if(U < foo) {
      return k;
    }
  }
  return n - 1;
}

double logit(double x) {
  return log(x) - log(1.0-x);
}

double expit(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// [[Rcpp::export]]
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& Precision) {
  arma::vec z = arma::zeros<arma::vec>(mean.size());
  for(int i = 0; i < mean.size(); i++) {
    z(i) = norm_rand();
  }
  arma::mat Sigma = inv_sympd(Precision);
  arma::mat L = chol(Sigma, "lower");
  arma::vec h = mean + L * z;
  /* arma::mat R = chol(Precision); */
    /* arma::vec h = solve(R,z) + mean; */
    return h;
}

// [[Rcpp::export]]
arma::mat choll(const arma::mat& Sigma) {
  return chol(Sigma);
}

double log_sum_exp(const arma::vec& x) {
  double M = x.max();
  return M + log(sum(exp(x - M)));
}
