#include "split_merge.h"

using namespace Rcpp;
using namespace arma;

std::vector<unsigned int> GetS(const arma::uvec& Z, int i, int j) {
  std::vector<unsigned int> S;

  for(unsigned int k = 0; k < Z.size(); k++) {
    if((i != k) && (j != k)) {
      if((Z(i) == Z(k)) || (Z(j) == Z(k))) {
        S.push_back(k);
      }
    }
  }
  return S;
}

double PseudoGibbsSweep(memat& Y,
                        arma::uvec& Z_launch,
                        const arma::uvec& Z,
                        std::vector<unsigned int>& S,
                        SuffStats& theta_i,
                        SuffStats& theta_j,
                        int i,
                        int j
                        )
{
  double loglik = 0.0;

  for(int S_idx = 0; S_idx < S.size(); S_idx++) {
    int k = S[S_idx];
    int z = Z_launch(k);
    double marg_i = 0.0;
    double marg_j = 0.0;
    // uvec Y_vec = trans(Y.row(k));
    if(z == Z_launch(i)) {
      marg_i = theta_i.calc_loglik();
      theta_i.DeleteObs(Y[k]);
      marg_i -= theta_i.calc_loglik();

      marg_j = -theta_j.calc_loglik();
      theta_j.AddObs(Y[k]);
      marg_j += theta_j.calc_loglik();
      theta_j.DeleteObs(Y[k]);
    }
    else {

      marg_j = theta_j.calc_loglik();
      theta_j.DeleteObs(Y[k]);
      marg_j -= theta_j.calc_loglik();

      marg_i = -theta_i.calc_loglik();
      theta_i.AddObs(Y[k]);
      marg_i += theta_i.calc_loglik();
      theta_i.DeleteObs(Y[k]);

    }
    vec log_probs = zeros<vec>(2);
    log_probs(0)  = marg_i + log(theta_i.num_obs);
    log_probs(1) = marg_j + log(theta_j.num_obs);
    log_probs = log_probs - log_sum_exp(log_probs);
    if(Z(k) == Z(i)) {
      theta_i.AddObs(Y[k]);
      loglik += log_probs(0);
    }
    else if(Z(k) == Z(j)) {
      theta_j.AddObs(Y[k]);
      loglik += log_probs(1);
    }
    else {
      Rcout << "ERROROROROR" << "\n";
    }
  }
  return loglik;
}

double RestrictedGibbsSweep(memat& Y,
                            arma::uvec& Z,
                            const std::vector<unsigned int>& S,
                            SuffStats& theta_i,
                            SuffStats& theta_j, 
                            int i, 
                            int j) {

  
  double loglik = 0.0;

  // Do restricted Gibbs sweep
  for(int S_idx = 0; S_idx < S.size(); S_idx++) {
    int k = S[S_idx];
    int z = Z(k);
    double marg_i = 0.0;
    double marg_j = 0.0;
    // uvec Y_vec = trans(Y.row(k));
    if(z == Z(i)) {

      marg_i = theta_i.calc_loglik();
      theta_i.DeleteObs(Y[k]);
      marg_i -= theta_i.calc_loglik();

      marg_j = -theta_j.calc_loglik();
      theta_j.AddObs(Y[k]);
      marg_j += theta_j.calc_loglik();
      theta_j.DeleteObs(Y[k]);

    }
    else if(z == Z(j)) {

      marg_j = theta_j.calc_loglik();
      theta_j.DeleteObs(Y[k]);
      marg_j -= theta_j.calc_loglik();

      marg_i = -theta_i.calc_loglik();
      theta_i.AddObs(Y[k]);
      marg_i += theta_i.calc_loglik();
      theta_i.DeleteObs(Y[k]);

    }
    else {
      Rcout << "Z = " << z << "ERRORRORRROR";
      Rcout << " valid values are " << Z(i) << " and " << Z(j) << "\n";
    }
    vec log_probs = zeros<vec>(2);
    log_probs(0) = marg_i + log(theta_i.num_obs);
    log_probs(1) = marg_j + log(theta_j.num_obs);
    log_probs = log_probs - log_sum_exp(log_probs);
    if(log(unif_rand()) < log_probs(0)) {
      Z(k) = Z(i);
      theta_i.AddObs(Y[k]);
      loglik += log_probs(0);
    }
    else {
      Z(k) = Z(j);
      theta_j.AddObs(Y[k]);
      loglik += log_probs(1);
    }
  }
  return loglik;
}


arma::uvec GetInitLaunchState(const arma::uvec& Z,
                              const std::vector<unsigned int>& S,
                              const arma::uvec& open_idx,
                              int i, int j) {
  
  arma::uvec Z_launch = Z;
  if(Z(i) == Z(j)) {
    Z_launch(i) = open_idx(0);
  }
  for(int k = 0; k < S.size(); k++) {
    Z_launch(S[k]) = unif_rand() < 0.5 ? Z_launch(i) : Z_launch(j);
  }

  return Z_launch;
  
}

SuffStats GetSS(memat& Y,
                const arma::uvec& Z,
                const std::vector<unsigned int>& S,
                double alpha,
                int P, int i)
{
  SuffStats theta(P, alpha);
  // uvec Y_vec = trans(Y.row(i));
  theta.AddObs(Y[i]);
  for(int kk = 0; kk < S.size(); kk++) {
    int k = S[kk];
    if(Z(k) == Z(i)) {
      // Y_vec = trans(Y.row(k));
      theta.AddObs(Y[k]);
    }
  }
  return theta;
}

SuffStats GetMergedSS(memat& Y,
                      const arma::uvec& Z,
                      const std::vector<unsigned int>& S,
                      double alpha,
                      int P, int i, int j)
{
  SuffStats theta_ij(P, alpha);
  // uvec Y_i = trans(Y.row(i));
  // uvec Y_j = trans(Y.row(j));
  theta_ij.AddObs(Y[i]);
  theta_ij.AddObs(Y[j]);
  for(int k = 0; k < S.size(); k++) {
    // Y_i = trans(Y.row(S[k]));
    theta_ij.AddObs(Y[k]);
  }

  return theta_ij;
}

arma::uvec Merge(memat& Y,
                 const arma::uvec& Z,
                 int K,
                 const arma::uvec& open_idx,
                 double alpha,
                 double omega,
                 int P,
                 int M,
                 int i,
                 int j)
{
  // Get S
  std::vector<unsigned int> S = GetS(Z, i,j);

  // Get split sufficient statistics
  SuffStats theta_i = GetSS(Y, Z, S, alpha, P, i);
  SuffStats theta_j = GetSS(Y, Z, S, alpha, P, j);
  double loglik_split = theta_i.calc_loglik() + theta_j.calc_loglik() +
    R::lgammafn(omega/K + theta_i.num_obs) + R::lgammafn(omega/K + theta_j.num_obs);


  // Get launch state
  uvec Z_launch = GetInitLaunchState(Z, S, open_idx, i, j);
  theta_i= GetSS(Y,Z_launch,S,alpha,P,i);
  theta_j= GetSS(Y,Z_launch,S,alpha,P,j);

  // Do restricted gibbs sweeps
  for(int m = 0; m < M; m++)
    RestrictedGibbsSweep(Y, Z_launch, S, theta_i, theta_j, i, j);

  // Rcout << theta_i.num_obs << "\n";
  // Rcout << theta_j.num_obs << "\n";

  // One final pseudo-Gibbs sweep
  double log_trans_prob = PseudoGibbsSweep(Y, Z_launch, Z, S, theta_i, theta_j, i, j);



  // Merge the components and compute sufficient statistics
  uvec Z_merged = Z;
  Z_merged(j) = Z_merged(i);
  for(int S_idx = 0; S_idx < S.size(); S_idx++)
    Z_merged(S[S_idx]) = Z_merged(i);
  SuffStats theta_ij = GetMergedSS(Y, Z_merged, S, alpha, P, i, j);

  if(theta_ij.sum_counts < 0) Rcout << "IJ" << theta_ij.sum_counts << " ";

  double loglik_merged = theta_ij.calc_loglik()
    + R::lgammafn(omega/K + theta_ij.num_obs);


  // Rcout << "loglik_merged = " << loglik_merged << "\n";
  // Rcout << "loglik_split = " << loglik_split << "\n";
  // Rcout << "log_trans_prob = " << log_trans_prob << "\n";


  double log_accept_prob = loglik_merged + log_trans_prob
    - loglik_split - log(1.0);


  double accept_prob = exp(log_accept_prob) > 1.0 ? 1.0 : exp(log_accept_prob);
  // Rcout << accept_prob; 

  if(log(unif_rand()) < log_accept_prob) {
    return Z_merged;
  }

  return Z;

}

arma::uvec Split(memat& Y,
                 const arma::uvec& Z,
                 int K,
                 const arma::uvec& open_idx,
                 double alpha,
                 double omega,
                 int P,
                 int M, 
                 int i,
                 int j)
{
  
  if(open_idx.size() == 0)
    return Z;

  // Get S
  std::vector<unsigned int> S = GetS(Z, i,j);

  // Get merged sufficient statistics
  SuffStats theta_ij = GetMergedSS(Y,Z,S,alpha,P,i,j);

  // Get log-likelihood of merged state
  double loglik_merged = R::lgammafn(omega / K + theta_ij.num_obs) + theta_ij.calc_loglik(); 
  // Rcout << "loglik_merged = " << loglik_merged << "\n";

  // Get Launch state
  uvec Z_launch = GetInitLaunchState(Z, S, open_idx, i, j);

  // Get split sufficient statistics
  SuffStats theta_i = GetSS(Y, Z_launch, S, alpha, P, i);
  SuffStats theta_j = GetSS(Y, Z_launch, S, alpha, P, j);

  // Do restricted gibbs sweeps
  for(int m = 0; m < M; m++) {
    RestrictedGibbsSweep(Y, Z_launch, S, theta_i, theta_j, i, j);
  }

  // One final gibbs sweep
  double log_trans_prob = RestrictedGibbsSweep(Y,Z_launch,S,theta_i,theta_j,i,j);
  // Rcout << "log trans prob = " << log_trans_prob << "\n";

  double loglik_split = theta_i.calc_loglik() + theta_j.calc_loglik()
    + R::lgammafn(omega / K + theta_i.num_obs) + R::lgammafn(omega/K + theta_j.num_obs);
  // Rcout << "loglik_split = " << loglik_split << "\n";

  double log_accept_prob = loglik_split + log(1.0)
    - loglik_merged - log_trans_prob;

  double accept_prob = exp(log_accept_prob) > 1.0 ? 1.0 : exp(log_accept_prob);
  // Rcout << accept_prob; 

  if(log(unif_rand()) > log_accept_prob) {
    Z_launch = Z;
  }

  return Z_launch;

}

arma::uvec SplitMergeCpp(memat& Y,
                         const arma::uvec& Z,
                         int K,
                         const arma::uvec& open_idx,
                         double alpha,
                         double omega,
                         int P,
                         int M) {

  // i and j are defined as in the Split-Merge paper
  int i = sample_class(Y.size());
  int j = sample_class(Y.size());

  if(i == j)
    return Z;

  if(Z(i) == Z(j)) {
    // Rcout << "SPLIT";
    return Split(Y, Z, K, open_idx, alpha, omega, P, M, i, j);
  }
  
  // Rcout << "MERGE";
  uvec out = Merge(Y, Z, K, open_idx, alpha, omega, P, M, i, j);
  return out;

}
