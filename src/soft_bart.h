#ifndef SOFT_BART_H
#define SOFT_BART_H

#include <RcppArmadillo.h>
#include "functions.h"
#include "split_merge.h"

// [[Rcpp::depends(RcppArmadillo)]]

struct Hypers;
struct Node;

struct Hypers {

  double alpha;
  double omega;
  double beta;
  double gamma;
  double sigma;
  double sigma_mu;
  double shape;
  double width;
  double tau_rate;
  double num_tree_prob;
  double alpha_rate;
  double temperature;
  int num_tree;
  int num_clust;
  int num_groups;
  /* arma::vec s; */
  /* arma::vec logs; */
  arma::uvec group;

  arma::vec rho_propose;

  std::vector<std::vector<unsigned int> > group_to_vars;

  // New stuff for itneraction detection
  arma::uvec z;
  arma::mat s;
  arma::vec s_0;
  arma::mat logs;
  arma::vec pi;
  arma::vec log_pi;

  double sigma_hat;
  double sigma_mu_hat;
  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;

  void UpdateSigma(const arma::vec& r);
  void UpdateSigmaMu(const arma::vec& means);
  /* void UpdateAlpha(); */
  void UpdateGamma(std::vector<Node*>& forest);
  void UpdateBeta(std::vector<Node*>& forest);
  void UpdateTauRate(const std::vector<Node*>& forest);

  // For updating tau
  double loglik_tau(double tau,
                    const std::vector<Node*>& forest,
                    const arma::mat& X, const arma::vec& Y);
  void update_tau(std::vector<Node*>& forest,
                  const arma::mat& X, const arma::vec& Y);


  Hypers(Rcpp::List hypers);
  Hypers();

};

struct Node {

  bool is_leaf;
  bool is_root;
  Node* left;
  Node* right;
  Node* parent;

  // Branch parameters
  int var;
  double val;
  double lower;
  double upper;
  double tau;

  // Leaf parameters
  double mu;

  // Data for computing weights
  double current_weight;

  // New stuff for interaction detection
  const Hypers* hypers;
  int tree_number;

  // Functions
  void Root(const Hypers& hypers, int i);
  void GetLimits();
  void AddLeaves();
  void BirthLeaves(const Hypers& hypers);
  bool is_left();
  void GenTree(const Hypers& hypers);
  void GenBelow(const Hypers& hypers);
  void GetW(const arma::mat& X, int i);
  void DeleteLeaves();
  void UpdateMu(const arma::vec& Y, const arma::mat& X, const Hypers& hypers);
  void UpdateTau(const arma::vec& Y, const arma::mat& X, const Hypers& hypers);
  void SetTau(double tau_new);
  double loglik_tau(double tau_new, const arma::mat& X, const arma::vec& Y, const Hypers& hypers);
  int SampleVar() const;

  Node();
  ~Node();

};



struct Opts {
  int num_burn;
  int num_thin;
  int num_save;
  int num_print;

  bool update_sigma_mu;
  bool update_s;
  bool update_alpha;
  bool update_beta;
  bool update_gamma;
  bool update_tau;
  bool update_tau_mean;
  bool update_num_tree;
  bool split_merge;
  bool s_burned;
  double mh_bd;
  double mh_prior;
  bool do_interaction;

Opts() : update_sigma_mu(true), update_s(true), update_alpha(true),
    update_beta(false), update_gamma(false), update_tau(true),
    update_tau_mean(false), update_num_tree(false), s_burned(false) {

  num_burn = 1;
  num_thin = 1;
  num_save = 1;
  num_print = 100;

}

Opts(Rcpp::List opts_) {

  update_sigma_mu = opts_["update_sigma_mu"];
  update_s = opts_["update_s"];
  update_alpha = opts_["update_alpha"];
  update_beta = opts_["update_beta"];
  update_gamma = opts_["update_beta"];
  update_tau = opts_["update_tau"];
  update_tau_mean = opts_["update_tau_mean"];
  update_num_tree = opts_["update_num_tree"];
  num_burn = opts_["num_burn"];
  num_thin = opts_["num_thin"];
  num_save = opts_["num_save"];
  num_print = opts_["num_print"];

}

};

class Forest {

 private:

  std::vector<Node*> trees;
  Hypers hypers;
  Opts opts;

  arma::umat tree_counts;

 public:

  /* Forest(Rcpp::List hypers_); */
  Forest(Rcpp::List hypers_, Rcpp::List opts_);
  ~Forest();
  // arma::vec predict(const arma::mat& X);
  arma::mat do_gibbs(const arma::mat& X,
                     const arma::vec& Y,
                     const arma::mat& X_test, int num_iter);
  arma::vec get_s() {return hypers.s;}
  arma::uvec get_counts();
  arma::umat get_tree_counts();
  void set_s(const arma::vec& s_);
  int num_gibbs;


};


Opts InitOpts(int num_burn, int num_thin, int num_save, int num_print,
              bool update_sigma_mu, bool update_s, bool update_alpha,
              bool update_beta, bool update_gamma, bool update_tau,
              bool update_tau_mean, bool update_num_tree, bool split_merge,
              double mh_bd, double mh_prior, bool do_interaction);


Hypers InitHypers(const arma::mat& X, double sigma_hat, double alpha, double omega, double beta,
                  double gamma, double k, double width, double shape,
                  int num_tree, double alpha_scale, double alpha_shape_1,
                  double alpha_shape_2, double tau_rate, double num_tree_prob,
                  double alpha_rate,
                  double temperature, int num_clust, const arma::vec& s_0);

void GetSuffStats(Node* n, const arma::vec& y,
                  const arma::mat& X, const Hypers& hypers,
                  arma::vec& mu_hat_out, arma::mat& Omega_inv_out);

double LogLT(Node* n, const arma::vec& Y,
             const arma::mat& X, const Hypers& hypers);

double cauchy_jacobian(double tau, double sigma_hat);

double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old,
                    double temperature = 1.0);
arma::vec loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers);
arma::vec predict(const std::vector<Node*>& forest,
                  const arma::mat& X,
                  const Hypers& hypers);

arma::vec predict(Node* node,
                  const arma::mat& X,
                  const Hypers& hypers);

bool is_left(Node* n);

double SplitProb(Node* node, const Hypers& hypers);
int depth(Node* node);
void leaves(Node* x, std::vector<Node*>& leafs);
std::vector<Node*> leaves(Node* x);
arma::vec get_means(std::vector<Node*>& forest);
void get_means(Node* node, std::vector<double>& means);
std::vector<Node*> init_forest(const arma::mat& X, const arma::vec& Y,
                               const Hypers& hypers);

Rcpp::List do_soft_bart(const arma::mat& X,
                        const arma::vec& Y,
                        const arma::mat& X_test,
                        const Hypers& hypers,
                        const Opts& opts);

void IterateGibbsWithS(std::vector<Node*>& forest, arma::vec& Y_hat,
                       Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                       const Opts& opts);
void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat,
                     Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                     const Opts& opts);
void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,
                 Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                 const Opts& opts);
double activation(double x, double c, double tau);
void birth_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const Hypers& hypers);
void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,
                const Hypers& hypers);
void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                const Hypers& hypers);
void change_decision_rule(Node* tree, const arma::mat& X, const arma::vec& Y,
                          const Hypers& hypers);
Node* draw_prior(Node* tree, const arma::mat& X, const arma::vec& Y, Hypers& hypers);
double growth_prior(int leaf_depth, const Hypers& hypers);
Node* birth_node(Node* tree, double* leaf_node_probability);
double probability_node_birth(Node* tree);
Node* death_node(Node* tree, double* p_not_grand);
std::vector<Node*> not_grand_branches(Node* tree);
void not_grand_branches(std::vector<Node*>& ngb, Node* node);
arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers);
void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers);
arma::mat get_var_counts_by_cluster(std::vector<Node*>& forest,
                                     const Hypers& hypers);
void get_var_counts_by_cluster(arma::mat& counts,
                               Node* node,
                               const Hypers& hypers);
double alpha_to_rho(double alpha, double scale);
double rho_to_alpha(double rho, double scale);
double logpdf_beta(double x, double a, double b);
double growth_prior(int node_depth, double gamma, double beta);
double forest_loglik(std::vector<Node*>& forest, double gamma, double beta);
double tree_loglik(Node* node, int node_depth, double gamma, double beta);
Node* rand(std::vector<Node*> ngb);
void UpdateS(std::vector<Node*>& forest, Hypers& hypers);
void UpdateSShared(std::vector<Node*>& forest, Hypers& hypers);
void UpdateZ(std::vector<Node*>& forest, Hypers& hypers);
void ComputeZLoglik(Node* tree, Hypers& hypers, arma::vec& logliks);
void UpdatePi(std::vector<Node*>& forest, Hypers& hypers);
void UpdateOmega(Hypers& hypers);
void UpdateAlpha(Hypers& hypers);
void UpdateAlphaShared(Hypers& hypers);

// Split merge
void split_merge(std::vector<Node*>& forest, Hypers& hypers); 
void get_predictor(Node* node, mevec& predictor);
arma::uvec get_open_idx(const arma::uvec& Z, int K);

// For tau
bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new);
double logprior_tau(double tau, double tau_rate);
double tau_proposal(double tau);
double log_tau_trans(double tau_new);
arma::vec get_tau_vec(const std::vector<Node*>& forest);

// RJMCMC for trees
std::vector<Node*> TreeSwap(std::vector<Node*>& forest);
std::vector<Node*> TreeSwapLast(std::vector<Node*>& forest);
std::vector<Node*> AddTree(std::vector<Node*>& forest,
                           const Hypers& hypers, const Opts& opts);
std::vector<Node*> DeleteTree(std::vector<Node*>& forest);
void update_num_tree(std::vector<Node*>& forest, Hypers& hypers,
                     const Opts& opts,
                     const arma::vec& Y, const arma::vec& res,
                     const arma::mat& X);
double LogLF(const std::vector<Node*>& forest, const Hypers& hypers,
             const arma::vec& Y, const arma::mat& X);
double loglik_normal(const arma::vec& resid, const double& sigma);
void BirthTree(std::vector<Node*>& forest,
               Hypers& hypers,
               const Opts& opts,
               const arma::vec& Y,
               const arma::vec& res,
               const arma::mat& X);
void DeathTree(std::vector<Node*>& forest,
               Hypers& hypers,
               const arma::vec& Y,
               const arma::vec& res,
               const arma::mat& X);
double TPrior(const std::vector<Node*>& forest, const Hypers& hypers);
void RenormAddTree(std::vector<Node*>& forest,
                   std::vector<Node*>& new_forest,
                   Hypers& hypers);
void UnnormAddTree(std::vector<Node*>& forest,
                   std::vector<Node*>& new_forest,
                   Hypers& hypers);
void RenormDeleteTree(std::vector<Node*>& forest,
                      std::vector<Node*>& new_forest,
                      Hypers& hypers);
void UnnormDeleteTree(std::vector<Node*>& forest,
                      std::vector<Node*>& new_forest,
                      Hypers& hypers);

// Slice sampler

struct loglik {
  virtual double operator() (double x) {return 0.0;}
};

/* struct rho_loglik : loglik { */
/*   double mean_log_s; */
/*   double p; */
/*   double alpha_scale; */
/*   double alpha_shape_1; */
/*   double alpha_shape_2; */

/*   double operator() (double rho) { */

/*     double alpha = rho_to_alpha(rho, alpha_scale); */

/*     double loglik = alpha * mean_log_s */
/*       + Rf_lgammafn(alpha) */
/*       - p * Rf_lgammafn(alpha / p) */
/*       + logpdf_beta(rho, alpha_shape_1, alpha_shape_2); */

/*     /\* Rcpp::Rcout << "Term 1: " << alpha * mean_log_s << "\n"; *\/ */
/*     /\* Rcpp::Rcout << "Term 2:" << Rf_lgammafn(alpha) << "\n"; *\/ */
/*     /\* Rcpp::Rcout << "Term 3:" << -p * Rf_lgammafn(alpha / p) << "\n"; *\/ */
/*     /\* Rcpp::Rcout << "Term 4:" << logpdf_beta(rho, alpha_shape_1, alpha_shape_2) << "\n"; *\/ */

/*     return loglik; */

/*   } */
/* }; */

struct rho_loglik : loglik {
  double sum_log_s;
  double p;
  double k;
  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;

  double operator() (double rho) {
    double alpha = rho_to_alpha(rho, alpha_scale);

    double loglik = k * Rf_lgammafn(alpha) - k * p * Rf_lgammafn(alpha/p)
      + alpha / p * sum_log_s + logpdf_beta(rho, alpha_shape_1, alpha_shape_2);

    return loglik;
  }
};

struct alpha_exp_loglik : loglik {
  double sum_log_s;
  double p;
  double k;
  double alpha_rate;

  double operator() (double alpha) {
    double loglik = k * Rf_lgammafn(alpha) - k * p * Rf_lgammafn(alpha/p)
      + alpha / p * sum_log_s - alpha * alpha_rate;

    return loglik;
  }
};

struct omega_loglik : loglik {
  double mean_log_pi;
  double scale_omega;
  double K;

  double operator() (double omega) {
    double out = Rf_lgammafn(omega) - K * Rf_lgammafn(omega / K)
      + omega * mean_log_pi - omega / scale_omega;
    return out;
  }

 omega_loglik(arma::vec& log_pi, double scale_omegaa) : scale_omega(scale_omegaa) {
    mean_log_pi = mean(log_pi);
    K = (double)log_pi.size();
  }
};

double slice_sampler(double x0, loglik& g, double w,
                     double lower, double upper) {


  /* Find the log density at the initial point, if not already known. */
  double gx0 = g(x0);

  /* Determine the slice level, in log terms. */

  double logy = gx0 - exp_rand();

  /* Find the initial interval to sample from */

  double u = w * unif_rand();
  double L = x0 - u;
  double R = x0 + (w-u);

  /* Expand the interval until its ends are outside the slice, or until the
     limit on steps is reached */

  do {

    if(L <= lower) break;
    if(g(L) <= logy) break;
    L = L - w;

  } while(true);

  do {
    if(R >= upper) break;
    if(g(R) <= logy) break;
    R = R + w;
  } while(true);

  // Shrink interval to lower and upper bounds

  if(L < lower) L = lower;
  if(R > upper) R = upper;

  // Sample from the interval, shrinking it on each rejection

  double x1 = 0.0;

  do {

    x1 = (R - L) * unif_rand() + L;
    double gx1 = g(x1);

    if(gx1 >= logy) break;

    if(x1 > x0) {
      R = x1;
    }
    else {
      L = x1;
    }

  } while(true);

  return x1;

}

// NEW INTERACTION STUFF


void get_var_counts_sparse(arma::sp_mat& out, Node* node, const Hypers& hypers);
arma::sp_mat get_var_counts_sparse(Node* tree, const Hypers& hypers);
void get_interactions_leaves(std::vector<int> &var_leaf, Node* n);
std::vector<int> get_interactions_leaves(Node* n);
std::vector<std::vector<int> > get_interactions(std::vector<Node*>& forest, const Hypers& hypers);
std::vector<std::vector<int> > get_unique_interaction(const std::vector<std::vector<std::vector<int> > > &out);
arma::sp_umat get_counts_interaction(const std::vector<std::vector<std::vector<int> > > &out,
                                     const std::vector<std::vector<int> > &unique_interaction);

// PERTURB STUFF
void branches(Node* n, std::vector<Node*>& branch_vec);
std::vector<Node*> branches(Node* root);
double calc_cutpoint_likelihood(Node* node);
std::vector<double> get_perturb_limits(Node* branch);

void perturb_decision_rule(Node* tree,
                           const arma::mat& X,
                           const arma::vec& Y,
                           const Hypers& hypers);

#endif
