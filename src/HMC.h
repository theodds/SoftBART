#include <RcppArmadillo.h>

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

struct HMCSampler {

public:

  virtual double calc_likelihood(const arma::vec& theta) {return 0.0;};
  virtual arma::vec calc_gradient(const arma::vec& theta) {return arma::zeros<arma::vec>(theta.size());};
  void do_leapfrog(arma::vec& theta, arma::vec& r, double epsilon_0);
  void find_reasonable_epsilon(const arma::vec& theta);
  arma::vec do_hmc_iteration(const arma::vec& theta);
  arma::vec do_hmc_iteration_dual(const arma::vec& theta);

  double epsilon;
  int num_leapfrog;
  int num_adapt;
  int num_iter;

  // Adaptation parameters
  double mu, epsilon_bar, log_epsilon_bar, H_bar, gamma, t_0, kappa, delta;

  HMCSampler(double eps, int leap, int adapt) : epsilon(eps), num_leapfrog(leap), num_adapt(adapt) {
    mu = log(10 * eps);
    epsilon_bar = 1.0;
    log_epsilon_bar = 0.0;
    H_bar = 0.0;
    gamma = 0.05;
    t_0 = 10.0;
    kappa = 0.75;
    num_iter = 0;
    delta = 0.7;
  }

};

struct HMCLogit : HMCSampler {

  double calc_likelihood(const arma::vec& theta);
  arma::vec calc_gradient(const arma::vec& theta);

  const arma::vec* Y;
  const arma::mat* X;

  HMCLogit(const arma::vec& YY, const arma::mat& XX, double eps, int leap,
           int num_adapt)
    : HMCSampler(eps, leap, num_adapt) {
    Y = &YY;
    X = &XX;
  }

};

struct HMCExpCopula : HMCSampler {

 public:

  double calc_likelihood(const arma::vec& zeta);
  arma::vec calc_gradient(const arma::vec& zeta);

  arma::vec get_Z_dot(const arma::vec& zeta);
  arma::vec zeta_to_s(const arma::vec& zeta);

  const arma::uvec* counts;
  double sigma;
  double N;

 HMCExpCopula(const arma::uvec& countss, double sigmaa, double eps, int leap, int num_adapt)
   : HMCSampler(eps, leap, num_adapt), sigma(sigmaa) {
    counts = &countss;
    N = sum(*counts);
  }

};
