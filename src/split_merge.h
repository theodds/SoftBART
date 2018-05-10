#include <RcppArmadillo.h>
#include "functions.h"

// using namespace arma;
// using namespace Rcpp;

typedef std::vector<unsigned int> mevec ;
typedef std::vector<mevec> memat ;

struct SuffStats {
  std::set<unsigned int> hot_idx;
  arma::uvec counts;
  int sum_counts;
  double alpha;
  int P;
  int num_obs;

  void AddObs(mevec& Y) {
    num_obs += 1;
    for(int j = 0; j < Y.size(); j++) {
      counts(Y[j]) = counts(Y[j]) + 1;
      if(counts(Y[j]) == 1) hot_idx.insert(Y[j]);
      sum_counts += 1;
    }
  }

  void DeleteObs(mevec& Y) {
    num_obs -= 1;
    for(int j = 0; j < Y.size(); j++) {
      counts(Y[j]) = counts(Y[j]) - 1;
      // if(counts(Y(j)) < 0) Rcout << counts(Y(j)); 
      if(counts(Y[j]) == 0) hot_idx.erase(Y[j]);
      sum_counts -= 1;
    }
  }

  SuffStats(int PP, double alphaa) : P(PP), alpha(alphaa) {
    counts = arma::zeros<arma::uvec>(P);


    sum_counts = 0;
    num_obs = 0;
  }

  double calc_loglik() {
    double out = R::lgammafn(P * alpha) - hot_idx.size() * R::lgammafn(alpha);
    out -= R::lgammafn(P * alpha + sum_counts);
    std::set<unsigned int>::iterator it;
    for(it = hot_idx.begin(); it != hot_idx.end(); ++it) {
      out += R::lgammafn(alpha + counts(*it));
    }
    return out;
  }
};

arma::uvec SplitMergeCpp(const memat& Y,
                         const arma::uvec& Z, int K, const arma::uvec& open_idx,
                         double alpha,
                         double omega,
                         int P,
                         int M);



/* arma::mat UpdateSCpp(arma::uvec& Z, arma::mat& Y, double alpha, int K, int P) { */

/*   mat s = zeros<mat>(K, P); */
/*   mat alpha_up = alpha * ones<mat>(K,P); */
/*   int N = Y.n_rows; */

/*   for(int i = 0; i < N; i++) { */
/*     for(int p = 0; p < P; p++) { */
/*       alpha_up(Z(i), p) = alpha_up(Z(i),p) + Y(i,p); */
/*     } */
/*   } */

/*   for(int k = 0; k < K; k++) { */
/*     vec alphaa = trans(alpha_up.row(k)); */
/*     vec s_up = rdirichlet(alphaa); */
/*     s.row(k) = trans(s_up); */
/*   } */

/*   return s; */
/* } */

/* arma::uvec UpdateZCpp(arma::mat& Y, arma::mat& logs, arma::vec& logpi) { */
/*   int K = logs.n_rows; */
/*   int N = Y.n_rows; */
/*   int P = logs.n_cols; */

/*   uvec z = zeros<uvec>(Y.n_rows); */
/*   vec loglik = zeros<vec>(logs.n_rows); */

/*   for(int i = 0; i < N; i++) { */
/*     loglik = logpi; */
/*     for(int k = 0; k < K; k++) { */
/*       loglik(k) = loglik(k) + sum(logs.row(k) % Y.row(i)); */
/*     } */
/*     vec prob = exp(loglik - log_sum_exp(loglik)); */
/*     z(i) = sample_class(prob); */
/*   } */

/*   return z; */
/* } */


std::vector<unsigned int> GetS(const arma::uvec& Z, int i, int j);

double PseudoGibbsSweep(const memat& Y,
                        arma::uvec& Z_launch,
                        const arma::uvec& Z,
                        std::vector<unsigned int> S,
                        SuffStats& theta_i,
                        SuffStats& theta_j,
                        int i,
                        int j
                        );

double RestrictedGibbsSweep(const memat& Y,
                            arma::uvec& Z,
                            const std::vector<unsigned int> S,
                            SuffStats& theta_i,
                            SuffStats& theta_j, 
                            int i, 
                            int j);


arma::uvec GetInitLaunchState(const arma::uvec& Z,
                              const std::vector<unsigned int>& S,
                              const arma::uvec& open_idx,
                              int i, int j);

SuffStats GetSS(const memat& Y,
                const arma::uvec& Z,
                const std::vector<unsigned int>& S,
                double alpha,
                int P, int i);

SuffStats GetMergedSS(const memat& Y,
                      const arma::uvec& Z,
                      const std::vector<unsigned int>& S,
                      double alpha,
                      int P, int i, int j);

arma::uvec Merge(const memat& Y,
                 const arma::uvec& Z,
                 int K,
                 const arma::uvec& open_idx,
                 double alpha,
                 double omega,
                 int P,
                 int M,
                 int i,
                 int j);

arma::uvec Split(memat& Y,
                 const arma::uvec& Z,
                 int K,
                 const arma::uvec& open_idx,
                 double alpha,
                 double omega,
                 int P,
                 int M, 
                 int i,
                 int j);

arma::uvec SplitMergeCpp(memat& Y,
                         const arma::uvec& Z,
                         int K,
                         const arma::uvec& open_idx,
                         double alpha,
                         double omega,
                         int P,
                         int M);
