#ifndef SOFT_BART_H
#define SOFT_BART_H

#include <RcppArmadillo.h>
#include "functions.h"

struct Hypers;
struct Node;

struct ProbHypers {
  bool use_counts;
  double dirichlet_mass;
  double log_mass;

  double calc_log_v(int n, int t);

  ProbHypers(double d_mass, arma::vec log_p, bool use_c,
             arma::uvec group);
  ProbHypers();

  int SampleVar();
  int ResampleVar(int var);
  void SwitchVar(int v_old, int v_new);
  void LoadGroups(arma::uvec group);

  std::map<std::pair<int,int>,double> log_V;
  arma::sp_uvec counts;
  arma::vec log_prior;

  // Group structure
  arma::uvec group;
  int num_groups;
  std::vector<std::vector<int>> group_to_vars;


};

struct Hypers {

  double beta;
  double gamma;
  double sigma;
  double sigma_mu;
  double shape;
  double width;
  double tau_rate;
  double temperature;
  int num_tree;

  ProbHypers split_hypers;

  double sigma_hat;
  double sigma_mu_hat;

  void UpdateSigma(const arma::vec& r, const arma::vec& weights);
  void UpdateSigmaMu(const arma::vec& means);
  void UpdateGamma(std::vector<Node*>& forest);
  void UpdateBeta(std::vector<Node*>& forest);
  void UpdateTauRate(const std::vector<Node*>& forest);

  // For updating tau
  // double loglik_tau(double tau,
  //                   const std::vector<Node*>& forest,
  //                   const arma::mat& X, const arma::vec& Y);
  // void update_tau(std::vector<Node*>& forest,
  //                 const arma::mat& X, const arma::vec& Y);

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

  // Functions
  void Root(Hypers& hypers);
  void GetLimits();
  void AddLeaves();
  void BirthLeaves(Hypers& hypers);
  bool is_left();
  void GenTree(Hypers& hypers);
  void GenBelow(Hypers& hypers);
  void GetW(const arma::mat& X, int i);
  void DeleteLeaves(Hypers& hypers);
  void UpdateMu(const arma::vec& Y, const arma::vec& weights,
                const arma::mat& X, const Hypers& hypers);
  void UpdateTau(const arma::vec& Y, const arma::vec& weights,
                 const arma::mat& X, const Hypers& hypers);
  void SetTau(double tau_new);
  double loglik_tau(double tau_new, const arma::mat& X, const arma::vec& Y,
                    const arma::vec& weights, const Hypers& hypers);

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
  bool update_beta;
  bool update_gamma;
  bool update_tau;
  bool update_tau_mean;
  bool update_sigma;

Opts() : update_sigma_mu(true), update_s(true),
    update_beta(false), update_gamma(false), update_tau(true),
         update_tau_mean(false), update_sigma(true) {

  num_burn = 1;
  num_thin = 1;
  num_save = 1;
  num_print = 100;

}

Opts(Rcpp::List opts_) {

  update_sigma_mu = opts_["update_sigma_mu"];
  update_s = opts_["update_s"];
  update_beta = opts_["update_beta"];
  update_gamma = opts_["update_beta"];
  update_tau = opts_["update_tau"];
  update_tau_mean = opts_["update_tau_mean"];
  update_sigma = opts_["update_sigma"];
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
  arma::mat do_gibbs_weighted(const arma::mat& X,
                              const arma::vec& Y, const arma::vec& weights,
                              const arma::mat& X_test, int num_iter);
  arma::uvec get_counts();
  arma::umat get_tree_counts();
  void set_sigma(double sigma);
  int num_gibbs;
  arma::vec do_predict(arma::mat& X);
  double get_sigma();


};


Opts InitOpts(int num_burn, int num_thin, int num_save, int num_print,
              bool update_sigma_mu, bool update_s, 
              bool update_beta, bool update_gamma, bool update_tau,
              bool update_tau_mean, bool update_sigma);


Hypers InitHypers(const arma::mat& X, double sigma_hat, double alpha, double beta,
                  double gamma, double k, double width, double shape,
                  int num_tree, double tau_rate, double temperature);

void GetSuffStats(Node* n, const arma::vec& y, const arma::vec& weights,
                  const arma::mat& X, const Hypers& hypers,
                  arma::vec& mu_hat_out, arma::mat& Omega_inv_out);

double LogLT(Node* n, const arma::vec& Y, const arma::vec& weights,
             const arma::mat& X, const Hypers& hypers);

double cauchy_jacobian(double tau, double sigma_hat);

double update_sigma(const arma::vec& r,
                    double sigma_hat, double sigma_old,
                    double temperature = 1.0);
double update_sigma(const arma::vec& r, const arma::vec& weights,
                    double sigma_hat, double sigma_old,
                    double temperature = 1.0);
arma::vec loglik_data(const arma::vec& Y, const arma::vec& weights, const arma::vec& Y_hat, const Hypers& hypers);
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
                        const arma::vec& weights,
                        const arma::mat& X_test,
                        const Hypers& hypers,
                        const Opts& opts);

void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat,
                     const arma::vec& weights,
                     Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                     const Opts& opts);
void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,
                 const arma::vec& weights,
                 Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                 const Opts& opts);
double activation(double x, double c, double tau);
void birth_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const arma::vec& weights, Hypers& hypers);
void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,
                const arma::vec& weights, Hypers& hypers);
void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                const arma::vec& weights, Hypers& hypers);
Node* draw_prior(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const arma::vec& weights, Hypers& hypers);
double growth_prior(int leaf_depth, const Hypers& hypers);
Node* birth_node(Node* tree, double* leaf_node_probability);
double probability_node_birth(Node* tree);
Node* death_node(Node* tree, double* p_not_grand);
std::vector<Node*> not_grand_branches(Node* tree);
void not_grand_branches(std::vector<Node*>& ngb, Node* node);
arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers);
void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers);
arma::vec rdirichlet(const arma::vec& shape);
double logpdf_beta(double x, double a, double b);
double growth_prior(int node_depth, double gamma, double beta);
double forest_loglik(std::vector<Node*>& forest, double gamma, double beta);
double tree_loglik(Node* node, int node_depth, double gamma, double beta);
Node* rand(std::vector<Node*> ngb);
void copy_node(Node* nn, Node* n);
Node* copy_tree(Node* root, Hypers& hypers);
std::vector<Node*> copy_forest(std::vector<Node*> forest);

// For tau
bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new);
double logprior_tau(double tau, double tau_rate);
double tau_proposal(double tau);
double log_tau_trans(double tau_new);
arma::vec get_tau_vec(const std::vector<Node*>& forest);

// PERTURB STUFF
void branches(Node* n, std::vector<Node*>& branch_vec);
std::vector<Node*> branches(Node* root);
double calc_cutpoint_likelihood(Node* node);
std::vector<double> get_perturb_limits(Node* branch);

void perturb_decision_rule(Node* tree,
                           const arma::mat& X,
                           const arma::vec& Y,
                           const arma::vec& weights,
                           Hypers& hypers);

// Gibbs Prior stuff
void AddTreeCounts(ProbHypers& split_hypers, Node* node);
void SubtractTreeCounts(ProbHypers& split_hypers, Node* node);

#endif
