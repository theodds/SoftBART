#include "HMC.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

vec rnorm_vec(int N) {
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) out(i) = norm_rand();
  return out;
}

vec log_pnorm(const arma::vec& zeta) {
  vec retval = zeros<vec>(zeta.size());
  for(int i = 0; i < zeta.size(); i++) {
    retval(i) = R::pnorm5(zeta(i), 0.0, 1.0, 1, 1);
  }
  return retval;
}

vec log_dnorm(const arma::vec& zeta) {
  vec retval = zeros<vec>(zeta.size());
  for(int i = 0; i < zeta.size(); i++) {
    retval(i) = R::dnorm4(zeta(i), 0.0, 1.0, 1);
  }
  return retval;
}

double logsumexp(const arma::vec& x) {
  double M = x.max();
  return M + log(sum(exp(x - M)));
}

void HMCSampler::do_leapfrog(vec& theta, vec& r, double epsilon_0) {
  // double epsilon_0 = -log(unif_rand()) * epsilon;
  r     = r + 0.5 * epsilon_0 * calc_gradient(theta);
  theta = theta + epsilon * r;
  r     = r + 0.5 * epsilon_0 * calc_gradient(theta);
}

arma::vec HMCSampler::do_hmc_iteration(const arma::vec& theta) {

  vec r = rnorm_vec(theta.size());
  vec r_tilde = r;

  // vec theta_new = theta;
  vec theta_tilde = theta;

  for(int i = 0; i < num_leapfrog; i++) {
    double epsilon_0 = -log(unif_rand()) * epsilon;
    do_leapfrog(theta_tilde, r_tilde, epsilon_0);
  }

  double log_alpha = calc_likelihood(theta_tilde) - calc_likelihood(theta)
    - 0.5 * sum(r_tilde % r_tilde) + 0.5 * sum(r % r);

  return log(unif_rand()) < log_alpha ? theta_tilde : theta;

}

arma::vec HMCSampler::do_hmc_iteration_dual(const arma::vec& theta) {

  vec r = rnorm_vec(theta.size());
  vec r_tilde = r;

  // vec theta_new = theta;
  vec theta_tilde = theta;

  for(int i = 0; i < num_leapfrog; i++) {
    double epsilon_0 = -log(unif_rand()) * epsilon;
    do_leapfrog(theta_tilde, r_tilde, epsilon_0);
  }

  double log_alpha = calc_likelihood(theta_tilde) - calc_likelihood(theta)
    - 0.5 * sum(r_tilde % r_tilde) + 0.5 * sum(r % r);
  double alpha = exp(log_alpha);
  alpha = alpha < 1.0 ? alpha : 1.0;

  vec theta_new = log(unif_rand()) < log_alpha ? theta_tilde : theta;
  if(num_iter < num_adapt) {
    int m = num_iter + 1;
    H_bar = (1.0 - 1.0 / (m + t_0)) * H_bar
    + 1.0 / (m + t_0) * (delta - alpha);
    double log_epsilon = mu - sqrt(m) / gamma * H_bar;
    log_epsilon_bar = pow(m, -kappa) * log_epsilon +
      (1.0 - pow(m, -kappa)) * log_epsilon_bar;
    epsilon = exp(log_epsilon);
    epsilon_bar = exp(log_epsilon_bar);
  }
  else {
    epsilon = epsilon_bar;
  }

  num_iter++;

  return theta_new;

}

void HMCSampler::find_reasonable_epsilon(const arma::vec& theta) {

  epsilon = 1.0;
  vec r = rnorm_vec(theta.size());
  vec theta_prime = theta;
  vec r_prime = r;
  do_leapfrog(theta_prime, r_prime, epsilon);

  double log_p_num = calc_likelihood(theta_prime) - 0.5 * sum(r_prime % r_prime);
  double log_p_den = calc_likelihood(theta) - 0.5 * sum(r % r);
  double log_rat = log_p_num - log_p_den;
  double a = log_rat > log(0.5) ? 1.0 : -1.0;

  while(a * log_rat > -a * log(2.0)) {
    epsilon = pow(2.0,a) * epsilon;
    theta_prime = theta;
    r_prime = r;
    do_leapfrog(theta_prime, r_prime, epsilon);
    log_p_num = calc_likelihood(theta_prime) - 0.5 * sum(r_prime % r_prime);
    log_p_den = calc_likelihood(theta) - 0.5 * sum(r % r);
    log_rat = log_p_num - log_p_den;
  }

  mu = log(10 * epsilon);

}

double HMCLogit::calc_likelihood(const arma::vec& theta) {
  vec eta = (*X) * theta;
  int N = (*Y).size();
  double out = 0.0;
  for(int i = 0; i < N; i++) {
    out += (*Y)(i) * eta(i) - log(1.0 + exp(eta(i)));
  }
  return out;
}

vec HMCLogit::calc_gradient(const arma::vec& theta) {
  vec eta = (*X) * theta;
  vec sigma = 1.0 / (1.0 + exp(-eta));
  int N = (*Y).size();
  vec out = zeros<vec>(theta.size());
  for(int i = 0; i < N; i++) {
    vec x_row = trans((*X).row(i));
    out = out + x_row * ((*Y)(i) - sigma(i));
  }
  return out;
}

double HMCExpCopula::calc_likelihood(const arma::vec& zeta) {
  vec z = sigma * log_pnorm(zeta);
  vec logs = z - logsumexp(z);
  return sum(log_dnorm(zeta)) + sum(logs % (*counts));
}

arma::vec HMCExpCopula::calc_gradient(const arma::vec& zeta) {
  vec z = sigma * log_pnorm(zeta);
  vec logs = z - logsumexp(z);
  vec z_dot = get_Z_dot(zeta);
  vec counts_hat = exp(logs) * N;
  return -zeta + z_dot % ((*counts) - counts_hat);
}

arma::vec HMCExpCopula::get_Z_dot(const arma::vec& zeta) {
  vec z_dot = exp(log(sigma) + log_dnorm(zeta) - log_pnorm(zeta));
  return z_dot;
}

arma::vec HMCExpCopula::zeta_to_s(const arma::vec& zeta) {
  vec z = sigma * log_pnorm(zeta);
  return(exp(z - logsumexp(z)));
}

// [[Rcpp::export]]
arma::mat fit_logistic(const arma::mat& X, const arma::vec& Y, int num_iter) {
  int P = X.n_cols;
  vec theta_0 = rnorm_vec(P);
  mat out = zeros<mat>(num_iter, P);

  Rcout << "Init the sampler\n";
  HMCLogit* sampler = new HMCLogit(Y, X, 1.0, 10, 500);

  Rcout << "Find reasonable epsilon" << "\n";
  sampler->find_reasonable_epsilon(theta_0);
  Rcout << "epsilon = " << sampler->epsilon;

  Rcout << "\nStarting loop:";
  for(int i = 0; i < num_iter; i++) {
    theta_0 = sampler->do_hmc_iteration_dual(theta_0);
    out.row(i) = trans(theta_0);
  }
  Rcout << "\nEnding epsilon = " << sampler->epsilon << "\n";

  return out;

}

// [[Rcpp::export]]
arma::mat fit_copula(const arma::uvec& counts, double sigma, int num_iter, int num_leap) {
  int P = counts.size();
  vec zeta_0 = zeros<vec>(P);
  mat out = zeros<mat>(num_iter, P);

  Rcout << "Init the sampler\n";
  HMCExpCopula* sampler = new HMCExpCopula(counts, sigma, 1.0, num_leap, num_iter/2);

  Rcout << "Find reasonable epsilon" << "\n";
  sampler->find_reasonable_epsilon(zeta_0);
  Rcout << "epsilon = " << sampler->epsilon;

  Rcout << "\nStarting loop\n:";
  for(int i = 0; i < num_iter; i++) {
    zeta_0 = sampler->do_hmc_iteration_dual(zeta_0);
    out.row(i) = trans(sampler->zeta_to_s(zeta_0));
    if((i+1) % 10 == 0)
      Rcout << "\rFinishing iteration " << i + 1 << "\t\t\t";
  }
  Rcout << "\nEnding epsilon = " << sampler->epsilon << "\n";

  return out;
}
