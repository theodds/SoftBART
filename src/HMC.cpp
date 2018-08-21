#include "HMC.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp0x")]]

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

  vec p = rnorm_vec(theta.size());
  vec current_p = p;
  double epsilon_0 = R::runif(0.8, 1.2) * epsilon;

  // Make half-step for moment at the beginning
  p = p + 0.5 * epsilon_0 * calc_gradient(theta);

  vec theta_tilde = theta;

  // Alternate full steps for position and momentum
  for(int i = 0; i < num_leapfrog; i++) {
    // Make a full step for the position

    theta_tilde = theta_tilde + epsilon_0 * p;

    // Make a full step for the momentum, except at the end of trajectory
    if(i < num_leapfrog - 1) p = p + epsilon_0 * calc_gradient(theta_tilde);
  }
  // Make a half step for momentum at the end
  p = p + 0.5 * epsilon * calc_gradient(theta_tilde);

  // Negate the momentum at end of trajectory to make the proposal symmetric
  p = -p;

  // Evaluate potential and kenetic energies at start and end of trajectory
  double current_loglik = calc_likelihood(theta);
  double current_momentum = -0.5 * dot(current_p, current_p);
  double proposed_loglik = calc_likelihood(theta_tilde);
  double proposed_momentum = -0.5 * dot(p, p);

  // Accept or reject the state at end of trajectory, returning either the
  // position at the end of the trajectory or the initial position
  double log_accept = proposed_loglik + proposed_momentum - current_loglik - current_momentum;

  return log(unif_rand()) < log_accept ? theta_tilde : theta;

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

double HMCLogitNormal::calc_likelihood(const arma::vec& zetaeta) {

  int P = zetaeta.size() - 1;
  vec zeta = zetaeta.rows(0, P-1);
  double eta = zetaeta(P);
  double nu = exp(eta);
  vec Z = nu * zeta;
  vec logs = Z - log_sum_exp(Z);

  double out = 0.0;

  // First term of loglik
  out += dot(counts, logs);

  // Second term of loglik
  out += -0.5 * dot(zeta, zeta);

  // Graph term of loglik
  int num_edge = i_vec.size();
  for(int n = 0; n < num_edge; n++) {
    int i = i_vec(n) - 1;
    int j = j_vec(n) - 1;
    if(i < j) {
      double delta_sq = std::pow(zeta(i) - zeta(j), 2);
      out += -a * log(b + 0.5 * delta_sq);
    }
  }

  // Prior of eta
  out += eta - nu / tau;

  // Return
  return out;
}

vec HMCLogitNormal::calc_gradient(const arma::vec& zetaeta) {
  
  int P = zetaeta.size() - 1;
  vec zeta = zetaeta.rows(0, P-1);
  double eta = zetaeta(P);
  double nu = exp(eta);
  vec Z = nu * zeta;
  vec logs = Z - log_sum_exp(Z);
  vec s = exp(logs);
  double B = sum(counts);

  vec out = zeros<vec>(P+1);

  // zeta gradient: first term and second term
  for(int j = 0; j < P; j++) {
    out(j) += nu * (counts(j) - B * s(j)) - zeta(j);
  }

  // zeta gradient: graph term
  int num_edge = i_vec.size();
  for(int n = 0; n < num_edge; n++) {
    int i = i_vec(n) - 1;
    int j = j_vec(n) - 1;
    double delta = zeta(i) - zeta(j);
    double delta_sq = std::pow(delta, 2);
    out(i) += -a * delta / (b + 0.5 * delta_sq);
  }

  // eta gradient
  out(P) += nu * (dot(counts, zeta) - B * dot(s, zeta)) + 1 - nu / tau;

  return out;
}

double HMCPoissonOffsetScaled::calc_likelihood(const arma::vec& zeta) {
  vec theta = zeta % scales;
  double out = -0.5 * dot(theta, Sigma_inv * theta)
    - phi * sum(exp(theta))
    + sum(Y % theta);
  return out;
}

arma::vec HMCPoissonOffsetScaled::calc_gradient(const arma::vec& zeta) {
  vec theta = zeta % scales;
  vec out = -scales % (Sigma_inv * theta - Y + phi * exp(theta));
  return out;
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

// [[Rcpp::export]]
Rcpp::List fit_logitnormal_2(arma::vec& counts,
                             arma::vec& mu,
                             arma::sp_mat& Sigma_inv,
                             arma::vec& theta_init,
                             int num_iter,
                             int num_leap) {

  // Constants
  int P = counts.size();
  double N = sum(counts);
  vec theta_0 = theta_init;
  mat warmup = zeros<mat>(num_iter, P);
  mat samps = zeros<mat>(num_iter, P);
  List out;

  // Compute reasonable scales
  vec scales = zeros<vec>(P);
  for(int p = 0; p < P; p++) {
    double as = counts(p) < 1 ? R::trigamma(1.0) : R::trigamma(counts(p));
    double a = 1.0 / as;
    double b = Sigma_inv(p,p);
    scales(p) = 1.0 / sqrt(a + b);
  }

  // Initialize the sampler
  double epsilon = 0.2;
  double phi_0 = 1.0;
  int num_adapt = 1;
  vec zeta_0 = theta_0 / scales;
  HMCPoissonOffsetScaled* sampler =
    new HMCPoissonOffsetScaled(counts, mu, Sigma_inv, scales, phi_0,
                               epsilon, num_leap, num_adapt);
  
  // Collect samples
  for(int i = 0; i < num_iter; i++) {
    sampler->phi = R::rgamma(N, 1.0/sum(exp(theta_0)));
    zeta_0 = sampler->do_hmc_iteration(zeta_0);
    theta_0 = zeta_0 % scales;
    warmup.row(i) = theta_0.t();
  }
  for(int i = 0; i < num_iter; i++) {
    sampler->phi = R::rgamma(N, 1.0/sum(exp(theta_0)));
    zeta_0 = sampler->do_hmc_iteration(zeta_0);
    theta_0 = zeta_0 % scales;
    samps.row(i) = theta_0.t();
  }

  out["warmup"] = warmup;
  out["samps"] = samps;

  return out;

}

void UpdateOmegaInv(arma::sp_mat& Omega_inv,
                    const arma::vec& theta,
                    double nu,
                    double a, double b) {


  // Init all diag to 1
  for(int p = 0; p < theta.size(); p++) {
    Omega_inv(p,p) = 1.0;
  }
  // Loop over the lower triangle
  for(sp_mat::iterator it = Omega_inv.begin(); it != Omega_inv.end(); ++it) {
    int row = it.row();
    int col = it.col();
    if(row < col) {
      double rate_up = b + 0.5 * pow(theta(row) - theta(col), 2) / nu;
      double scale_up = 1.0 / rate_up;
      double omega = R::rgamma(a, scale_up);
      Omega_inv(row, col) = -omega;
      Omega_inv(col, row) = -omega;
      Omega_inv(row, row) = Omega_inv(row,row) + omega;
      Omega_inv(col, col) = Omega_inv(col, col) + omega;
    }
  }
}

double UpdateTau(const arma::vec& theta_0, const arma::sp_mat& Omega_inv) {

  double shape_up = 0.5 * theta_0.size() + 1.0;
  double rate_up = 0.5 * dot(theta_0, Omega_inv * theta_0);
  double scale_up = 1.0 / rate_up;
  return R::rgamma(shape_up, scale_up);

}

void GetReasonableScales(arma::vec& scales, const arma::vec& counts, const arma::sp_mat& Sigma_inv) {
  for(int p = 0; p < scales.size(); p++) {
    double as = counts(p) < 1 ? R::trigamma(1.0) : R::trigamma(counts(p));
    double a = 1.0 / as;
    double b = Sigma_inv(p,p);
    scales(p) = 1.0 / sqrt(a + b);
  }
}

// [[Rcpp::export]]
Rcpp::List fit_logit_laplace(arma::vec& counts,
                             const arma::sp_mat& Graph,
                             const arma::vec& theta_init,
                             int num_iter,
                             int num_leap) {

  // Constants
  int P = counts.size();
  double N = sum(counts);
  double a_omega = 4.0;
  double b_omega = 1.0;
  vec mu = zeros<vec>(P);
  mat theta_out = zeros<mat>(num_iter, P);
  vec nu_out = zeros<vec>(num_iter);


  // Variables
  double tau = 1.0;
  vec theta_0 = theta_init;
  sp_mat Omega_inv = Graph;
  for(int p = 0; p < P; p++) {
    Omega_inv(p,p) = 1.0;
  }
  sp_mat Sigma_inv = Omega_inv * tau;
  double phi_0 = 1.0;
  vec scales = ones<vec>(P);
  vec zeta_0 = theta_0 / scales;

  // Initialize HMC sampler
  double epsilon = 0.2;
  int num_adapt = 1;
  HMCPoissonOffsetScaled* sampler =
    new HMCPoissonOffsetScaled(counts, mu, Omega_inv, scales, phi_0,
                               epsilon, num_leap, num_adapt);

  // Collect Samples
  for(int i = 0; i < num_iter; i++) {

    UpdateOmegaInv(Omega_inv, theta_0, 1.0/tau, a_omega, b_omega); 
    tau = UpdateTau(theta_0, Omega_inv);
    Sigma_inv = tau * Omega_inv;
    GetReasonableScales(scales, counts, Sigma_inv);

    // Do HMC by (1) rescaling zeta, (2) loading scales and Sigma_inv into HMC
    // (3) updating phi, (4) updating zeta, and (5) getting theta from zeta

    zeta_0 = theta_0 / scales;
    sampler->scales = scales;
    sampler->Sigma_inv = Sigma_inv;
    sampler->phi = R::rgamma(N, 1.0 / sum(exp(theta_0)));
    zeta_0 = sampler->do_hmc_iteration(zeta_0);
    theta_0 = zeta_0 % scales;

    theta_out.row(i) = theta_0.t();
    nu_out(i) = 1.0 / tau;

  }

  List out;
  out["nu"] = nu_out;
  out["theta"] = theta_out;

}

// [[Rcpp::export]]
Rcpp::List fit_logitnormal_nu(arma::vec& counts,
                              const arma::sp_mat& Omega_inv,
                              const arma::vec& theta_init,
                              int num_iter,
                              int num_leap) {

  // Constants
  int P = counts.size();
  double N = sum(counts);
  double tau = 1.0;
  sp_mat Sigma_inv = tau * Omega_inv;
  vec theta_0 = theta_init;
  vec nu_out = zeros<vec>(num_iter);
  mat theta_out = zeros<mat>(num_iter, P);
  vec mu = zeros<vec>(P);

  // Compute reasonable Scales
  vec scales = zeros<vec>(P);
  for(int p = 0; p < P; p++) {
    double as = counts(p) < 1 ? R::trigamma(1.0) : R::trigamma(counts(p));
    double a = 1.0 / as;
    double b = Sigma_inv(p,p);
    scales(p) = 1.0 / sqrt(a + b);
  }
  vec zeta_0 = theta_0 / scales;

  // Initialize HMC Sampler
  double epsilon = 0.2;
  double phi_0 = 1.0;
  int num_adapt = 1;
  HMCPoissonOffsetScaled* sampler =
    new HMCPoissonOffsetScaled(counts, mu, Sigma_inv, scales, phi_0,
                               epsilon, num_leap, num_adapt);

  // Collect Samples
  for(int i = 0; i < num_iter; i++) {


    // Compute reasonable scales
    for(int p = 0; p < P; p++) {
      double as = counts(p) < 1 ? R::trigamma(1.0) : R::trigamma(counts(p));
      double a = 1.0 / as;
      double b = Sigma_inv(p,p);
      scales(p) = 1.0 / sqrt(a + b);
    }
    vec zeta_0 = theta_0 / scales;
    sampler->scales = scales;

    sampler->phi = R::rgamma(N, 1.0 / sum(exp(theta_0)));
    zeta_0 = sampler->do_hmc_iteration(zeta_0);
    theta_0 = zeta_0 % scales;

    // Update tau
    double shape_up = 0.5 * P + 1.0;
    double scale_up = 1.0/( 0.5 * dot(theta_0, Omega_inv * theta_0) );
    double tau_prop = R::rgamma(shape_up, scale_up);
    // double xi_prop = sqrt(tau_prop) / (sqrt(tau_prop) + 1.0);
    // double xi_current = sqrt(tau) / (sqrt(tau) + 1.0);
    // double loglik_rat =
    //   R::dbeta(xi_prop, 0.5, 1.0, 1) - R::dbeta(xi_current, 0.5, 1.0, 1)
    //   - 0.5 * log(tau_prop) - log(sqrt(tau_prop) + 1.0)
    //   + 0.5 * log(tau) + log(sqrt(tau) + 1.0);
    // tau = log(unif_rand()) < loglik_rat ? tau_prop : tau;
    tau = tau_prop;
    nu_out(i) = 1.0 / tau;
    theta_out.row(i) = theta_0.t();
    Sigma_inv = tau * Omega_inv;
    sampler->Sigma_inv = Sigma_inv;
  }

  List out;
  out["nu"] = nu_out;
  out["theta"] = theta_out;

  return(out);

}

// [[Rcpp::export]]
Rcpp::List fit_logitnormal(arma::uvec& counts,
                           double tau,
                           arma::uvec& i_vec,
                           arma::uvec& j_vec,
                           int num_iter,
                           int num_leap) {



  int P = counts.size();
  vec zetaeta = zeros<vec>(P+1);
  mat warmup = zeros<mat>(num_iter, P+1);
  mat samps = zeros<mat>(num_iter, P+1);
  List out;

  HMCLogitNormal* sampler = new HMCLogitNormal(counts, tau, i_vec, j_vec, 1.0,
                                               num_leap, num_iter);
  sampler->find_reasonable_epsilon(zetaeta);

  for(int i = 0; i < num_iter; i++) {
    zetaeta = sampler->do_hmc_iteration_dual(zetaeta);
    warmup.row(i) = trans(zetaeta);
  }
  for(int i = 0; i < num_iter; i++) {
    zetaeta = sampler->do_hmc_iteration_dual(zetaeta);
    samps.row(i) = trans(zetaeta);
  }

  out["warmup"] = warmup;
  out["samps"] = samps;
  return(out);

}
