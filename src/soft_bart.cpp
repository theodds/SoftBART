#include "soft_bart.h"

// TODO:
// I AM IN THE MIDDLE OF WRITING UPDATEZ
// AFTER THAT I NEED TO UPDATE PI
// Make it so that omega is properly set as a hyperparameter
// Add an update for omega
// Add appropriate update for alpha
// Add appropriate update for z
// Make it so that number of clusters is fed in as hyperparameter
// Make it so that number of trees is no longer fixed at 50
// Fix the initialization of Hypers, otherwise forest stuff will not work (messed up s, also pi)

using namespace Rcpp;
using namespace arma;

bool RESCALE = true;

Forest::Forest(Rcpp::List hypers_, Rcpp::List opts_) : hypers(hypers_), opts(opts_) {
  trees.resize(hypers.num_tree);
  for(int i = 0; i < hypers.num_tree; i++) {
    trees[i] = new Node();
    trees[i]->Root(hypers,i);
    // trees[i]->GenTree(hypers);
  }
  num_gibbs = 0;
  tree_counts = zeros<umat>(hypers.num_groups, hypers.num_tree);
}

Forest::~Forest() {
  for(int i = 0; i < trees.size(); i++) {
    delete trees[i];
  }
}

Node::Node() {
  is_leaf = true;
  is_root = true;
  left = NULL;
  right = NULL;
  parent = NULL;

  var = 0;
  val = 0.0;
  lower = 0.0;
  upper = 1.0;
  tau = 1.0;
  mu = 0.0;
  current_weight = 0.0;
}

Opts InitOpts(int num_burn, int num_thin, int num_save, int num_print,
              bool update_sigma_mu, bool update_s, bool update_alpha,
              bool update_beta, bool update_gamma, bool update_tau,
              bool update_tau_mean, bool update_num_tree) {

  Opts out;
  out.num_burn = num_burn;
  out.num_thin = num_thin;
  out.num_save = num_save;
  out.num_print = num_print;
  out.update_sigma_mu = update_sigma_mu;
  out.update_s = update_s;
  out.update_alpha = update_alpha;
  out.update_beta = update_beta;
  out.update_gamma = update_gamma;
  out.update_tau = update_tau;
  out.update_tau_mean = update_tau_mean;
  out.update_num_tree = update_num_tree;
  out.s_burned = false;

  return out;

}

arma::vec InitPi(int num_clusters) {
  return ones<vec>(num_clusters) / ((double) num_clusters);
}

Hypers InitHypers(const mat& X, const uvec& group, double sigma_hat,
                  double alpha, double beta,
                  double gamma, double k, double width, double shape,
                  int num_tree, double alpha_scale, double alpha_shape_1,
                  double alpha_shape_2, double tau_rate, double num_tree_prob,
                  double temperature) {

  int GRID_SIZE = 1000;

  num_tree = 50; // REMOVE ME!!!!!!

  Hypers out;
  out.alpha = alpha;
  out.omega = 1.0; // REMOVE ME!!!!!
  out.beta = beta;
  out.gamma = gamma;
  out.sigma = sigma_hat;
  out.sigma_mu = 0.5 / (k * pow(num_tree, 0.5));
  out.shape = shape;
  out.width = width;
  out.num_tree = num_tree;

  out.num_groups = group.max() + 1;

  out.sigma_hat = sigma_hat;
  out.sigma_mu_hat = out.sigma_mu;

  out.alpha_scale = alpha_scale;
  out.alpha_shape_1 = alpha_shape_1;
  out.alpha_shape_2 = alpha_shape_2;
  out.tau_rate = tau_rate;
  out.num_tree_prob = num_tree_prob;
  out.temperature = temperature;

  out.group = group;

  // New stuff for interaction detection
  // out.num_clust = num_tree;
  out.num_clust = 5;
  out.s = ones<mat>(out.num_clust, out.num_groups) / ((double)(out.num_groups));
  out.logs = log(out.s);
  out.z = arma::zeros<arma::uvec>(num_tree);
  // FAKE INITIALIZATION, REMOVE ME
  out.pi = InitPi(out.num_clust);
  out.log_pi = log(out.pi);
  for(int i = 0; i < num_tree; i++) {
    out.z(i) = i % 5;
  }

  // Create mapping of group to variables
  out.group_to_vars.resize(out.num_groups);
  for(int i = 0; i < out.num_groups; i++) {
    out.group_to_vars[i].resize(0);
  }
  int P = group.size();
  for(int p = 0; p < P; p++) {
    int idx = group(p);
    out.group_to_vars[idx].push_back(p);
  }

  out.rho_propose = zeros<vec>(GRID_SIZE - 1);
  for(int i = 0; i < GRID_SIZE - 1; i++) {
    out.rho_propose(i) = (double)(i+1) / (double)(GRID_SIZE);
  }


  return out;
}

int Node::SampleVar() const {

  int cluster = hypers->z(tree_number);
  int group_idx = sample_class(trans(hypers->s.row(cluster)));
  int var_idx = sample_class(hypers->group_to_vars[group_idx].size());
  // int group_idx = sample_class(hypers->s);
  // int var_idx = sample_class(hypers->group_to_vars[group_idx].size());

  return hypers->group_to_vars[group_idx][var_idx];
}

void Node::Root(const Hypers& hypers, int i) {
  is_leaf = true;
  is_root = true;
  left = this;
  right = this;
  parent = this;

  var = 0;
  val = 0.0;
  lower = 0.0;
  upper = 1.0;
  tau = hypers.width;

  mu = 0.0;
  current_weight = 1.0;

  this->hypers = &hypers;
  tree_number = i;
}

// Check
void Node::AddLeaves() {
  left = new Node;
  right = new Node;
  is_leaf = false;

  // Initialize the leaves

  left->is_leaf = true;
  left->parent = this;
  left->right = left;
  left->left = left;
  left->var = 0;
  left->val = 0.0;
  left->is_root = false;
  left->lower = 0.0;
  left->upper = 1.0;
  left->mu = 0.0;
  left->current_weight = 0.0;
  left->tau = tau;
  left->hypers = hypers;
  left->tree_number = tree_number;
  right->is_leaf = true;
  right->parent = this;
  right->right = right;
  right->left = right;
  right->var = 0;
  right->val = 0.0;
  right->is_root = false;
  right->lower = 0.0;
  right->upper = 1.0;
  right->mu = 0.0;
  right->current_weight = 0.0;
  right->tau = tau;
  right->hypers = hypers;
  right->tree_number = tree_number;
}

void Node::BirthLeaves(const Hypers& hypers) {
  if(is_leaf) {
    AddLeaves();
    var = SampleVar();
    GetLimits();
    val = (upper - lower) * unif_rand() + lower;
  }
}

void Node::GenTree(const Hypers& hypers) {
  Root(hypers,1);
  GenBelow(hypers);
}

void Node::GenBelow(const Hypers& hypers) {
  double grow_prob = SplitProb(this, hypers);
  double u = unif_rand();
  if(u < grow_prob) {
    BirthLeaves(hypers);
    left->GenBelow(hypers);
    right->GenBelow(hypers);
  }
}

double SplitProb(Node* node, const Hypers& hypers) {
  double d = (double)(depth(node));
  return hypers.gamma * pow(1.0 + d, -hypers.beta);
}


bool Node::is_left() {
  return (this == this->parent->left);
}

void Node::GetLimits() {
  Node* y = this;
  lower = 0.0;
  upper = 1.0;
  bool my_bool = y->is_root ? false : true;
  while(my_bool) {
    bool is_left = y->is_left();
    y = y->parent;
    my_bool = y->is_root ? false : true;
    if(y->var == var) {
      my_bool = false;
      if(is_left) {
        upper = y->val;
        lower = y->lower;
      }
      else {
        upper = y->upper;
        lower = y->val;
      }
    }
  }
}

/*Computes the sufficient statistics Omega_inv and mu_hat described in the
  paper; mu_hat is the posterior mean of the leaf nodes, Omega_inv is that
  posterior covariance*/
void GetSuffStats(Node* n, const arma::vec& y,
                  const arma::mat& X, const Hypers& hypers,
                  arma::vec& mu_hat_out, arma::mat& Omega_inv_out) {


  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  vec w_i = zeros<vec>(num_leaves);
  vec mu_hat = zeros<vec>(num_leaves);
  mat Lambda = zeros<mat>(num_leaves, num_leaves);

  for(int i = 0; i < X.n_rows; i++) {
    n->GetW(X, i);
    for(int j = 0; j < num_leaves; j++) {
      w_i(j) = leafs[j]->current_weight;
    }
    mu_hat = mu_hat + y(i) * w_i;
    Lambda = Lambda + w_i * trans(w_i);
  }

  Lambda = Lambda / pow(hypers.sigma, 2) * hypers.temperature;
  mu_hat = mu_hat / pow(hypers.sigma, 2) * hypers.temperature;
  Omega_inv_out = Lambda + eye(num_leaves, num_leaves) / pow(hypers.sigma_mu, 2);
  mu_hat_out = solve(Omega_inv_out, mu_hat);

}

double LogLT(Node* n, const arma::vec& Y,
             const arma::mat& X, const Hypers& hypers) {

  // Rcout << "Leaves ";
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();

  // Get sufficient statistics
  arma::vec mu_hat = zeros<vec>(num_leaves);
  arma::mat Omega_inv = zeros<mat>(num_leaves, num_leaves);
  GetSuffStats(n, Y, X, hypers, mu_hat, Omega_inv);

  int N = Y.size();

  // Rcout << "Compute ";
  double out = -0.5 * N * log(M_2PI * pow(hypers.sigma,2)) * hypers.temperature;
  out -= 0.5 * num_leaves * log(M_2PI * pow(hypers.sigma_mu,2));
  double val, sign;
  log_det(val, sign, Omega_inv / M_2PI);
  out -= 0.5 * val;
  out -= 0.5 * dot(Y, Y) / pow(hypers.sigma, 2) * hypers.temperature;
  out += 0.5 * dot(mu_hat, Omega_inv * mu_hat);

  // Rcout << "Done";
  return out;

}

double cauchy_jacobian(double tau, double sigma_hat) {
  double sigma = pow(tau, -0.5);
  int give_log = 1;

  double out = Rf_dcauchy(sigma, 0.0, sigma_hat, give_log);
  out = out - M_LN2 - 3.0 / 2.0 * log(tau);

  return out;

}

double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old,
                    double temperature) {

  double SSE = dot(r,r) * temperature;
  double n = r.size() * temperature;

  double shape = 0.5 * n + 1.0;
  double scale = 2.0 / SSE;
  double sigma_prop = pow(Rf_rgamma(shape, scale), -0.5);

  double tau_prop = pow(sigma_prop, -2.0);
  double tau_old = pow(sigma_old, -2.0);

  double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -
    cauchy_jacobian(tau_old, sigma_hat);

  return log(unif_rand()) < loglik_rat ? sigma_prop : sigma_old;

}

void Hypers::UpdateSigma(const arma::vec& r) {
  sigma = update_sigma(r, sigma_hat, sigma, temperature);
}

void Hypers::UpdateSigmaMu(const arma::vec& means) {
  sigma_mu = update_sigma(means, sigma_mu_hat, sigma_mu);
}

void Node::UpdateMu(const arma::vec& Y, const arma::mat& X, const Hypers& hypers) {

  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();

  // Get mean and covariance
  vec mu_hat = zeros<vec>(num_leaves);
  mat Omega_inv = zeros<mat>(num_leaves, num_leaves);
  GetSuffStats(this, Y, X, hypers, mu_hat, Omega_inv);

  vec mu_samp = rmvnorm(mu_hat, Omega_inv);
  for(int i = 0; i < num_leaves; i++) {
    leafs[i]->mu = mu_samp(i);
  }
}

arma::vec predict(const std::vector<Node*>& forest,
                  const arma::mat& X,
                  const Hypers& hypers) {

  vec out = zeros<vec>(X.n_rows);
  int num_tree = forest.size();

  for(int t = 0; t < num_tree; t++) {
    out = out + predict(forest[t], X, hypers);
  }

  return out;
}

arma::vec predict(Node* n, const arma::mat& X, const Hypers& hypers) {

  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();
  int N = X.n_rows;
  vec out = zeros<vec>(N);

  for(int i = 0; i < N; i++) {
    n->GetW(X,i);
    for(int j = 0; j < num_leaves; j++) {
      out(i) = out(i) + leafs[j]->current_weight * leafs[j]->mu;
    }
  }

  return out;

}

void Node::GetW(const arma::mat& X, int i) {

  if(!is_leaf) {

    double weight = activation(X(i,var), val, tau);
    left->current_weight = weight * current_weight;
    right->current_weight = (1 - weight) * current_weight;

    left->GetW(X,i);
    right->GetW(X,i);

  }
}

bool is_left(Node* n) {
  return n->is_left();
}

void Node::DeleteLeaves() {
  delete left;
  delete right;
  left = this;
  right = this;
  is_leaf = true;
}

int depth(Node* node) {
  return node->is_root ? 0 : 1 + depth(node->parent);
}

void leaves(Node* x, std::vector<Node*>& leafs) {
  if(x->is_leaf) {
    leafs.push_back(x);
  }
  else {
    leaves(x->left, leafs);
    leaves(x->right, leafs);
  }
}

std::vector<Node*> leaves(Node* x) {
  std::vector<Node*> leafs(0);
  leaves(x, leafs);
  return leafs;
}

arma::vec get_means(std::vector<Node*>& forest) {
  std::vector<double> means(0);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_means(forest[t], means);
  }

  // Convert std::vector to armadillo vector, deep copy
  vec out(&(means[0]), means.size());
  return out;
}

void get_means(Node* node, std::vector<double>& means) {

  if(node->is_leaf) {
    means.push_back(node->mu);
  }
  else {
    get_means(node->left, means);
    get_means(node->right, means);
  }
}

std::vector<Node*> init_forest(const arma::mat& X, const arma::vec& Y,
                               const Hypers& hypers) {

  std::vector<Node*> forest(0);
  for(int t = 0; t < hypers.num_tree; t++) {
    Node* n = new Node;
    n->Root(hypers,t);
    forest.push_back(n);
  }
  return forest;
}

Rcpp::List do_soft_bart(const arma::mat& X,
                        const arma::vec& Y,
                        const arma::mat& X_test,
                        Hypers& hypers,
                        Opts& opts) {


  std::vector<Node*> forest = init_forest(X, Y, hypers);

  vec Y_hat = zeros<vec>(X.n_rows);

  // Do burn_in

  for(int i = 0; i < opts.num_burn; i++) {

    // Don't update s for half of the burn-in
    if(i < opts.num_burn / 2) {
      // Rcout << "Iterating Gibbs\n";
      IterateGibbsNoS(forest, Y_hat, hypers, X, Y, opts);
    }
    else {
      IterateGibbsWithS(forest, Y_hat, hypers, X, Y, opts);
    }

    if((i+1) % opts.num_print == 0) {
      // Rcout << "Finishing warmup " << i + 1 << ": tau = " << hypers.width << "\n";
      Rcout << "\rFinishing warmup " << i + 1
            // << " tau_rate = " << hypers.tau_rate
            << " Number of trees = " << hypers.num_tree
            << "\t\t\t"
        ;
    }

  }

  opts.s_burned = true;
  Rcout << std::endl;

  // Make arguments to return
  mat Y_hat_train = zeros<mat>(opts.num_save, X.n_rows);
  mat Y_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  vec sigma = zeros<vec>(opts.num_save);
  vec sigma_mu = zeros<vec>(opts.num_save);
  // vec alpha = zeros<vec>(opts.num_save);
  vec beta = zeros<vec>(opts.num_save);
  vec gamma = zeros<vec>(opts.num_save);
  // mat s = zeros<mat>(opts.num_save, hypers.s.size());
  umat var_counts = zeros<umat>(opts.num_save, hypers.num_groups);
  vec tau_rate = zeros<vec>(opts.num_save);
  uvec num_tree = zeros<uvec>(opts.num_save);
  vec loglik = zeros<vec>(opts.num_save);
  mat loglik_train = zeros<mat>(opts.num_save, Y_hat.size());

  // Do save iterations
  for(int i = 0; i < opts.num_save; i++) {
    for(int b = 0; b < opts.num_thin; b++) {
      IterateGibbsWithS(forest, Y_hat, hypers, X, Y, opts);
      if(i > opts.num_save / 2) {
        UpdatePi(forest, hypers);
      }
    }

    // Save stuff
    Y_hat_train.row(i) = Y_hat.t();
    Y_hat_test.row(i) = trans(predict(forest, X_test, hypers));
    sigma(i) = hypers.sigma;
    sigma_mu(i) = hypers.sigma_mu;
    // s.row(i) = trans(hypers.s);
    var_counts.row(i) = trans(get_var_counts(forest, hypers));
    // alpha(i) = hypers.alpha;
    beta(i) = hypers.beta;
    gamma(i) = hypers.gamma;
    tau_rate(i) = hypers.tau_rate;
    loglik_train.row(i) = trans(loglik_data(Y,Y_hat,hypers));
    loglik(i) = sum(loglik_train.row(i));
    num_tree(i) = hypers.num_tree;

    if((i + 1) % opts.num_print == 0) {
      // Rcout << "Finishing save " << i + 1 << ": tau = " << hypers.width << "\n";
      Rcout << "\rFinishing save " << i + 1 << "\t\t\t";
    }

  }

  Rcout << std::endl;

  Rcout << "Number of leaves at final iterations:\n";
  for(int t = 0; t < hypers.num_tree; t++) {
    Rcout << leaves(forest[t]).size() << " ";
    if((t + 1) % 10 == 0) Rcout << "\n";
  }

  // for(int t = 0; t < hypers.num_tree; t++) {
  //   Rcout << leaves(forest[t]).size() << " ";
  //   if((t + 1) % 10 == 0) Rcout << "\n";
  // }

  List out;
  out["y_hat_train"] = Y_hat_train;
  out["y_hat_test"] = Y_hat_test;
  out["sigma"] = sigma;
  out["sigma_mu"] = sigma_mu;
  // out["s"] = s;
  // out["alpha"] = alpha;
  out["fvarcounts"] = get_var_counts_by_cluster(forest, hypers);
  out["fz"] = hypers.z;
  out["beta"] = beta;
  out["gamma"] = gamma;
  out["var_counts"] = var_counts;
  out["tau_rate"] = tau_rate;
  out["num_tree"] = num_tree;
  out["loglik"] = loglik;
  out["loglik_train"] = loglik_train;


  return out;

}

void IterateGibbsWithS(std::vector<Node*>& forest, arma::vec& Y_hat,
                       Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                       const Opts& opts) {
  IterateGibbsNoS(forest, Y_hat, hypers, X, Y, opts);
  // if(opts.update_s) UpdateS(forest, hypers);
  if(opts.update_s) {
    if(opts.s_burned) {
      UpdateS(forest, hypers);
      UpdateZ(forest, hypers);
    }
    else {
      UpdateSShared(forest, hypers); 
    }
  }
  // if(opts.update_alpha) hypers.UpdateAlpha();
  // if(opts.update_num_tree) update_num_tree(forest, hypers, opts, Y, Y - Y_hat, X);
}

void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat,
                     Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                     const Opts& opts) {


  // Rcout << "Backfitting trees";
  TreeBackfit(forest, Y_hat, hypers, X, Y, opts);
  arma::vec res = Y - Y_hat;
  arma::vec means = get_means(forest);

  // Rcout << "Doing other updates";
  hypers.UpdateSigma(res);
  if(opts.update_sigma_mu) hypers.UpdateSigmaMu(means);
  if(opts.update_beta) hypers.UpdateBeta(forest);
  if(opts.update_gamma) hypers.UpdateGamma(forest);
  if(opts.update_tau_mean) hypers.UpdateTauRate(forest);

  Rcpp::checkUserInterrupt();
}

void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,
                 const Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                 const Opts& opts) {

  double MH_BD = 0.7;

  int num_tree = hypers.num_tree;
  for(int t = 0; t < num_tree; t++) {
    // Rcout << "Getting backfit quantities";
    arma::vec Y_star = Y_hat - predict(forest[t], X, hypers);
    arma::vec res = Y - Y_star;

    if(forest[t]->is_leaf || unif_rand() < MH_BD) {
      // Rcout << "BD step";
      birth_death(forest[t], X, res, hypers);
      // Rcout << "Done";
    }
    else {
      // Rcout << "Change step";
      change_decision_rule(forest[t], X, res, hypers);
      // Rcout << "Done";
    }
    if(opts.update_tau) forest[t]->UpdateTau(res, X, hypers);
    forest[t]->UpdateMu(res, X, hypers);
    Y_hat = Y_star + predict(forest[t], X, hypers);
  }
}

double activation(double x, double c, double tau) {
  return 1.0 - expit((x - c) / tau);
}

void birth_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const Hypers& hypers) {


  double p_birth = probability_node_birth(tree);

  if(unif_rand() < p_birth) {
    node_birth(tree, X, Y, hypers);
  }
  else {
    node_death(tree, X, Y, hypers);
  }
}

void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,
                const Hypers& hypers) {

  // Rcout << "Sample leaf";
  double leaf_probability = 0.0;
  Node* leaf = birth_node(tree, &leaf_probability);

  // Rcout << "Compute prior";
  int leaf_depth = depth(leaf);
  double leaf_prior = growth_prior(leaf_depth, hypers);

  // Get likelihood of current state
  // Rcout << "Current likelihood";
  double ll_before = LogLT(tree, Y, X, hypers);
  ll_before += log(1.0 - leaf_prior);

  // Get transition probability
  // Rcout << "Transistion";
  double p_forward = log(probability_node_birth(tree) * leaf_probability);

  // Birth new leaves
  // Rcout << "Birth";
  leaf->BirthLeaves(hypers);

  // Get likelihood after
  // Rcout << "New Likelihood";
  double ll_after = LogLT(tree, Y, X, hypers);
  ll_after += log(leaf_prior) +
    log(1.0 - growth_prior(leaf_depth + 1, hypers)) +
    log(1.0 - growth_prior(leaf_depth + 1, hypers));

  // Get Probability of reverse transition
  // Rcout << "Reverse";
  std::vector<Node*> ngb = not_grand_branches(tree);
  double p_not_grand = 1.0 / ((double)(ngb.size()));
  double p_backward = log((1.0 - probability_node_birth(tree)) * p_not_grand);

  // Do MH
  double log_trans_prob = ll_after + p_backward - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    leaf->DeleteLeaves();
    leaf->var = 0;
  }
  else {
    // Rcout << "Accept!";
  }
}

void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                const Hypers& hypers) {

  // Select branch to kill Children
  double p_not_grand = 0.0;
  Node* branch = death_node(tree, &p_not_grand);

  // Compute before likelihood
  int leaf_depth = depth(branch->left);
  double leaf_prob = growth_prior(leaf_depth - 1, hypers);
  double left_prior = growth_prior(leaf_depth, hypers);
  double right_prior = growth_prior(leaf_depth, hypers);
  double ll_before = LogLT(tree, Y, X, hypers) +
    log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob);

  // Compute forward transition prob
  double p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)));

  // Save old leafs, do not delete (they are dangling, need to be handled by the end)
  Node* left = branch->left;
  Node* right = branch->right;
  branch->left = branch;
  branch->right = branch;
  branch->is_leaf = true;

  // Compute likelihood after
  double ll_after = LogLT(tree, Y, X, hypers) + log(1.0 - leaf_prob);

  // Compute backwards transition
  std::vector<Node*> leafs = leaves(tree);
  double p_backwards = log(1.0 / ((double)(leafs.size())) * probability_node_birth(tree));

  // Do MH and fix dangles
  double log_trans_prob = ll_after + p_backwards - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    branch->left = left;
    branch->right = right;
    branch->is_leaf = false;
  }
  else {
    delete left;
    delete right;
  }
}

void change_decision_rule(Node* tree, const arma::mat& X, const arma::vec& Y,
                          const Hypers& hypers) {

  std::vector<Node*> ngb = not_grand_branches(tree);
  Node* branch = rand(ngb);

  // Calculate likelihood before proposal
  double ll_before = LogLT(tree, Y, X, hypers);

  // save old split
  int old_feature = branch->var;
  double old_value = branch->val;
  double old_lower = branch->lower;
  double old_upper = branch->upper;

  // Modify the branch
  // branch->var = sample_class(hypers.s);
  branch->var = tree->SampleVar();
  branch->GetLimits();
  branch->val = (branch->upper - branch->lower) * unif_rand() + branch->lower;

  // Calculate likelihood after proposal
  double ll_after = LogLT(tree, Y, X, hypers);

  // Do MH
  double log_trans_prob = ll_after - ll_before;

  if(log(unif_rand()) > log_trans_prob) {
    branch->var = old_feature;
    branch->val = old_value;
    branch->lower = old_lower;
    branch->upper = old_upper;
  }

}

double growth_prior(int leaf_depth, const Hypers& hypers) {
  return hypers.gamma * pow(1.0 + leaf_depth, -hypers.beta);
}

Node* birth_node(Node* tree, double* leaf_node_probability) {
  std::vector<Node*> leafs = leaves(tree);
  Node* leaf = rand(leafs);
  *leaf_node_probability = 1.0 / ((double)leafs.size());

  return leaf;
}

double probability_node_birth(Node* tree) {
  return tree->is_leaf ? 1.0 : 0.5;
}

Node* death_node(Node* tree, double* p_not_grand) {
  std::vector<Node*> ngb = not_grand_branches(tree);
  Node* branch = rand(ngb);
  *p_not_grand = 1.0 / ((double)ngb.size());

  return branch;
}

std::vector<Node*> not_grand_branches(Node* tree) {
  std::vector<Node*> ngb(0);
  not_grand_branches(ngb, tree);
  return ngb;
}

void not_grand_branches(std::vector<Node*>& ngb, Node* node) {
  if(!node->is_leaf) {
    bool left_is_leaf = node->left->is_leaf;
    bool right_is_leaf = node->right->is_leaf;
    if(left_is_leaf && right_is_leaf) {
      ngb.push_back(node);
    }
    else {
      not_grand_branches(ngb, node->left);
      not_grand_branches(ngb, node->right);
    }
  }
}

arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers) {
  arma::uvec counts = zeros<uvec>(hypers.num_groups);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t], hypers);
  }
  return counts;
}

void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers) {
  if(!node->is_leaf) {
    int group_idx = hypers.group(node->var);
    counts(group_idx) = counts(group_idx) + 1;
    get_var_counts(counts, node->left, hypers);
    get_var_counts(counts, node->right, hypers);
  }
}

arma::mat get_var_counts_by_cluster(std::vector<Node*>& forest,
                                     const Hypers& hypers) {

  arma::mat counts = zeros<mat>(hypers.num_clust, hypers.num_groups);
  for(int t = 0; t < forest.size(); t++) {
    get_var_counts_by_cluster(counts, forest[t], hypers);
  }
  return counts;
}

void get_var_counts_by_cluster(arma::mat& counts,
                               Node* node,
                               const Hypers& hypers) {
  if(!node->is_leaf) {
    int group_idx = hypers.group(node->var);
    int z = hypers.z(node->tree_number);
    counts(z, group_idx) = counts(z, group_idx) + 1;
    get_var_counts_by_cluster(counts, node->left, hypers);
    get_var_counts_by_cluster(counts, node->right, hypers);
  }
}

/*Note: Because the shape of the Dirichlet will mostly be small, we sample from
  the Dirichlet distribution by sampling log-gamma random variables using the
  technique of Liu, Martin, and Syring (2017+) and normalizing using the
  log-sum-exp trick */

void UpdateS(std::vector<Node*>& forest, Hypers& hypers) {

  // Get shape vector
  mat shape_up = ones<mat>(hypers.num_clust, hypers.num_groups);
  shape_up = shape_up * hypers.alpha / ((double) hypers.num_groups);
  // Rcout << "\nC";
  shape_up = shape_up + get_var_counts_by_cluster(forest, hypers);
  // Rcout << "\nD";

  // Sample unnormalized s on the log scale
  for(int k = 0; k < hypers.num_clust; k++) {
    vec logs = zeros<vec>(hypers.num_groups);
    for(int p = 0; p < hypers.num_groups; p++) {
      logs(p) = rlgam(shape_up(k,p));
    }
    logs = logs - log_sum_exp(logs);
    hypers.s.row(k) = trans(exp(logs));
    hypers.logs.row(k) = trans(logs);
  }

}

void UpdateSShared(std::vector<Node*>& forest, Hypers& hypers) {

  vec shape_up = ones<vec>(hypers.num_groups)
    * hypers.alpha / ((double) hypers.num_groups);
  shape_up = shape_up + get_var_counts(forest, hypers);

  rowvec logs = zeros<rowvec>(hypers.num_groups);
  for(int i = 0; i < hypers.num_groups; i++) {
    logs(i) = rlgam(shape_up(i));
  }
  logs = logs - log_sum_exp(logs.t());
  rowvec s = exp(logs);
  for(int k = 0; k < hypers.num_clust; k++) {
    hypers.s.row(k) = s;
    hypers.logs.row(k) = logs;
  }

}

void UpdateZ(std::vector<Node*>& forest, Hypers& hypers) {
  for(int t = 0; t < forest.size(); t++) {
    vec logliks = hypers.log_pi;
    Node* tree = forest[t];
    ComputeZLoglik(tree, hypers, logliks);
    vec probs = exp(logliks - log_sum_exp(logliks));
    hypers.z(t) = sample_class(probs);
  }
}

void ComputeZLoglik(Node* tree, Hypers& hypers, arma::vec& logliks) {
  if(!tree->is_leaf) {
    int group_idx = hypers.group(tree->var); 
    for(int k = 0; k < hypers.num_clust; k++) {
      logliks(k) = logliks(k) + hypers.logs(k,group_idx);
    }
    ComputeZLoglik(tree->left, hypers, logliks);
    ComputeZLoglik(tree->right, hypers, logliks);
  }
}

void UpdatePi(std::vector<Node*>& forest, Hypers& hypers) {
  vec shape_up = ones<vec>(hypers.num_clust) * hypers.omega / hypers.num_clust;
  for(int t = 0; t < forest.size(); t++) {
    int z = hypers.z(t);
    shape_up(z) = shape_up(z) + 1;
  }
  vec log_pi_up = zeros<vec>(hypers.num_clust);
  for(int k = 0; k < hypers.num_clust; k++) {
    log_pi_up(k) = rlgam(shape_up(k));
  }
  log_pi_up = log_pi_up - log_sum_exp(log_pi_up);
  hypers.log_pi = log_pi_up;
  hypers.pi = exp(log_pi_up);
}

// THIS IS THE OLD UPDATES
// void UpdateS(std::vector<Node*>& forest, Hypers& hypers) {

//   // Get shape vector
//   vec shape_up = hypers.alpha / ((double)hypers.s.size()) * ones<vec>(hypers.s.size());
//   shape_up = shape_up + get_var_counts(forest, hypers);

//   // Sample unnormalized s on the log scale
//   for(int i = 0; i < shape_up.size(); i++) {
//     hypers.logs(i) = rlgam(shape_up(i));
//   }
//   // Normalize s on the log scale, then exponentiate
//   hypers.logs = hypers.logs - log_sum_exp(hypers.logs);
//   hypers.s = exp(hypers.logs);

// }

// [[Rcpp::export]]
double rlgam(double shape) {
  if(shape >= 0.1) return log(Rf_rgamma(shape, 1.0));

  double a = shape;
  double L = 1.0/a- 1.0;
  double w = exp(-1.0) * a / (1.0 - a);
  double ww = 1.0 / (1.0 + w);
  double z = 0.0;
  do {
    double U = unif_rand();
    if(U <= ww) {
      z = -log(U / ww);
    }
    else {
      z = log(unif_rand()) / L;
    }
    double eta = z >= 0 ? -z : log(w)  + log(L) + L * z;
    double h = -z - exp(-z / a);
    if(h - eta > log(unif_rand())) break;
  } while(true);

  // Rcout << "Sample: " << -z/a << "\n";

  return -z/a;
}

arma::vec rdirichlet(const arma::vec& shape) {
  vec out = zeros<vec>(shape.size());
  for(int i = 0; i < shape.size(); i++) {
    do {
      out(i) = Rf_rgamma(shape(i), 1.0);
    } while(out(i) == 0);
  }
  out = out / sum(out);
  return out;
}

double alpha_to_rho(double alpha, double scale) {
  return alpha / (alpha + scale);
}

double rho_to_alpha(double rho, double scale) {
  return scale * rho / (1.0 - rho);
}

double logpdf_beta(double x, double a, double b) {
  return (a-1.0) * log(x) + (b-1.0) * log(1 - x) - Rf_lbeta(a,b);
}

// void Hypers::UpdateAlpha() {
//   arma::vec logliks = zeros<vec>(rho_propose.size());

//   rho_loglik loglik;
//   loglik.mean_log_s = mean(logs);
//   loglik.p = (double)s.size();
//   loglik.alpha_scale = alpha_scale;
//   loglik.alpha_shape_1 = alpha_shape_1;
//   loglik.alpha_shape_2 = alpha_shape_2;

//   for(int i = 0; i < rho_propose.size(); i++) {
//     logliks(i) = loglik(rho_propose(i));
//   }

//   logliks = exp(logliks - log_sum_exp(logliks));
//   double rho_up = rho_propose(sample_class(logliks));
//   alpha = rho_to_alpha(rho_up, alpha_scale);

// }

// void Hypers::UpdateAlpha() {

//   double rho = alpha_to_rho(alpha, alpha_scale);
//   double psi = mean(log(s));
//   double p = (double)s.size();

//   double loglik = alpha * psi + Rf_lgammafn(alpha) - p * Rf_lgammafn(alpha / p) +
//     logpdf_beta(rho, alpha_shape_1, alpha_shape_2);

//   // 50 MH proposals
//   for(int i = 0; i < 50; i++) {
//     double rho_propose = Rf_rbeta(alpha_shape_1, alpha_shape_2);
//     double alpha_propose = rho_to_alpha(rho_propose, alpha_scale);

//     double loglik_propose = alpha_propose * psi + Rf_lgammafn(alpha_propose) -
//       p * Rf_lgammafn(alpha_propose/p) +
//       logpdf_beta(rho_propose, alpha_shape_1, alpha_shape_2);

//     if(log(unif_rand()) < loglik_propose - loglik) {
//       alpha = alpha_propose;
//       rho = rho_propose;
//       loglik = loglik_propose;
//     }
//   }
// }

// void Hypers::UpdateAlpha() {

//   rho_loglik loglik;
//   loglik.mean_log_s = mean(logs);
//   loglik.p = (double)s.size();
//   loglik.alpha_scale = alpha_scale;
//   loglik.alpha_shape_1 = alpha_shape_1;
//   loglik.alpha_shape_2 = alpha_shape_2;

//   double rho = alpha_to_rho(alpha, alpha_scale);
//   rho = slice_sampler(rho, loglik, 0.1, 0.0 + exp(-10.0), 1.0);
//   alpha = rho_to_alpha(rho, alpha_scale);
// }


double growth_prior(int node_depth, double gamma, double beta) {
  return gamma * pow(1.0 + node_depth, -beta);
}

double forest_loglik(std::vector<Node*>& forest, double gamma, double beta) {
  double out = 0.0;
  for(int t = 0; t < forest.size(); t++) {
    out += tree_loglik(forest[t], 0, gamma, beta);
  }
  return out;
}

double tree_loglik(Node* node, int node_depth, double gamma, double beta) {
  double out = 0.0;
  if(node->is_leaf) {
    out += log(1.0 - growth_prior(node_depth, gamma, beta));
  }
  else {
    out += log(growth_prior(node_depth, gamma, beta));
    out += tree_loglik(node->left, node_depth + 1, gamma, beta);
    out += tree_loglik(node->right, node_depth + 1, gamma, beta);
  }
  return out;
}

void Hypers::UpdateGamma(std::vector<Node*>& forest) {
  double loglik = forest_loglik(forest, gamma, beta);

  for(int i = 0; i < 10; i++) {
    double gamma_prop = 0.5 * unif_rand() + 0.5;
    double loglik_prop = forest_loglik(forest, gamma_prop, beta);
    if(log(unif_rand()) < loglik_prop - loglik) {
      gamma = gamma_prop;
      loglik = loglik_prop;
    }
  }
}

void Hypers::UpdateBeta(std::vector<Node*>& forest) {

  double loglik = forest_loglik(forest, gamma, beta);

  for(int i = 0; i < 10; i++) {
    double beta_prop = fabs(Rf_rnorm(0.0, 2.0));
    double loglik_prop = forest_loglik(forest, gamma, beta_prop);
    if(log(unif_rand()) < loglik_prop - loglik) {
      beta = beta_prop;
      loglik = loglik_prop;
    }
  }
}

Node* rand(std::vector<Node*> ngb) {

  int N = ngb.size();
  arma::vec p = ones<vec>(N) / ((double)(N));
  int i = sample_class(p);
  return ngb[i];
}

// double loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers) {
//   vec res = Y - Y_hat;
//   double out = -0.5 * Y.size() * log(M_2PI * pow(hypers.sigma,2.0)) -
//     dot(res, res) * 0.5 / pow(hypers.sigma,2.0);
//   return out;
// }

arma::vec loglik_data(const arma::vec& Y, const arma::vec& Y_hat, const Hypers& hypers) {
  vec res = Y - Y_hat;
  vec out = zeros<vec>(Y.size());
  for(int i = 0; i < Y.size(); i++) {
    out(i) = -0.5 * log(M_2PI * pow(hypers.sigma,2)) - 0.5 * pow(res(i) / hypers.sigma, 2);
  }
  return out;
}

// [[Rcpp::export]]
List SoftBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test,
              const arma::uvec& group,
              double alpha, double beta, double gamma, double sigma,
              double shape, double width, int num_tree,
              double sigma_hat, double k, double alpha_scale,
              double alpha_shape_1, double alpha_shape_2, double tau_rate,
              double num_tree_prob,
              double temperature,
              int num_burn,
              int num_thin, int num_save, int num_print, bool update_sigma_mu,
              bool update_s, bool update_alpha, bool update_beta, bool update_gamma,
              bool update_tau, bool update_tau_mean, bool update_num_tree) {


  Opts opts = InitOpts(num_burn, num_thin, num_save, num_print, update_sigma_mu,
                       update_s, update_alpha, update_beta, update_gamma,
                       update_tau, update_tau_mean, update_num_tree);

  Hypers hypers = InitHypers(X, group, sigma_hat, alpha, beta, gamma, k, width,
                             shape, num_tree, alpha_scale, alpha_shape_1,
                             alpha_shape_2, tau_rate, num_tree_prob, temperature);

  // Rcout << "Doing soft_bart\n";
  return do_soft_bart(X,Y,X_test,hypers,opts);

}

// [[Rcpp::export]]
bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new) {

  double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;

  return log(unif_rand()) < cutoff ? true : false;

}

// Local tau stuff
void Node::SetTau(double tau_new) {
  tau = tau_new;
  if(!is_leaf) {
    left->SetTau(tau_new);
    right->SetTau(tau_new);
  }
}

double Node::loglik_tau(double tau_new, const arma::mat& X,
                        const arma::vec& Y, const Hypers& hypers) {

  double tau_old = tau;
  SetTau(tau_new);
  double out = LogLT(this, Y, X, hypers);
  SetTau(tau_old);
  return out;

}

void Node::UpdateTau(const arma::vec& Y,
                     const arma::mat& X,
                     const Hypers& hypers) {

  double tau_old = tau;
  double tau_new = tau_proposal(tau);

  double loglik_new = loglik_tau(tau_new, X, Y, hypers) + logprior_tau(tau_new, hypers.tau_rate);
  double loglik_old = loglik_tau(tau_old, X, Y, hypers) + logprior_tau(tau_old, hypers.tau_rate);
  double new_to_old = log_tau_trans(tau_old);
  double old_to_new = log_tau_trans(tau_new);

  bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);

  if(accept_mh) {
    SetTau(tau_new);
  }
  else {
    SetTau(tau_old);
  }

}

Hypers::Hypers() {
  alpha = 1.0;
  omega = 1.0;
  beta = 2.0;
  gamma = 0.95;
}

Hypers::Hypers(Rcpp::List hypers) {
  alpha = hypers["alpha"];
  beta = hypers["beta"];
  gamma = hypers["gamma"];
  omega = 1.0;
  sigma = hypers["sigma"];
  sigma_mu = hypers["sigma_mu"];
  sigma_mu_hat = sigma_mu;
  shape = hypers["shape"];
  width = hypers["width"];
  num_tree = hypers["num_tree"];
  sigma_hat = hypers["sigma_hat"];
  alpha_scale = hypers["alpha_scale"];
  alpha_shape_1 = hypers["alpha_shape_1"];
  alpha_shape_2 = hypers["alpha_shape_2"];
  tau_rate = hypers["tau_rate"];
  num_tree_prob = hypers["num_tree_prob"];
  temperature = hypers["temperature"];

  // Deal with group and num_group
  group = as<arma::uvec>(hypers["group"]);
  num_groups = group.max() + 1;


  // Deal with other stuff

  s = 1.0 / group.size() * arma::ones<arma::vec>(group.size());
  logs = log(s);

  group_to_vars.resize(num_groups);
  for(int i = 0; i < num_groups; i++) {
   group_to_vars[i].resize(0);
  }
  int P = group.size();
  for(int p = 0; p < P; p++) {
   int idx = group(p);
   group_to_vars[idx].push_back(p);
  }

  int GRID_SIZE = 1000;

  rho_propose = arma::zeros<arma::vec>(GRID_SIZE - 1);
  for(int i = 0; i < GRID_SIZE - 1; i++) {
    rho_propose(i) = (double)(i+1) / (double)(GRID_SIZE);
  }

}


// Global tau stuff
double logprior_tau(double tau, double tau_rate) {
  int DO_LOG = 1;
  return Rf_dexp(tau, 1.0 / tau_rate, DO_LOG);
}

double tau_proposal(double tau) {
  double U = 2.0 * unif_rand() - 1;
  return pow(5.0, U) * tau;
  // double w = 0.2 * unif_rand() - 0.1;
  // return tau + w;
}

double Hypers::loglik_tau(double tau,
                          const std::vector<Node*>& forest,
                          const arma::mat& X, const arma::vec& Y) {

  double tau_old = width;
  width = tau;
  vec Y_hat = predict(forest, X, *this);
  double SSE = dot(Y - Y_hat, Y - Y_hat);
  double sigma_sq = pow(sigma, 2);

  double loglik = -0.5 * Y.size() * log(sigma_sq) - 0.5 * SSE / sigma_sq;

  width = tau_old;
  return loglik;

}

double log_tau_trans(double tau_new) {
  return -log(tau_new);
  // return 0.0;
}

void Hypers::UpdateTauRate(const std::vector<Node*>& forest) {

  vec tau_vec = get_tau_vec(forest);
  double shape_up = forest.size() + 1.0;
  double rate_up = sum(tau_vec) + 0.1;
  double scale_up = 1.0 / rate_up;

  tau_rate = Rf_rgamma(shape_up, scale_up);

}

arma::vec get_tau_vec(const std::vector<Node*>& forest) {
  int t = forest.size();
  vec out = zeros<vec>(t);
  for(int i = 0; i < t; i++) {
    out(i) = forest[i]->tau;
  }
  return out;
}


// Reversible jump stuff ----

std::vector<Node*> TreeSwap(std::vector<Node*>& forest) {
  int num_tree = forest.size();
  int idx_1 = sample_class(num_tree);
  int idx_2 = sample_class(num_tree);

  std::vector<Node*> new_forest = forest;

  forest[idx_1] = new_forest[idx_2];
  forest[idx_2] = new_forest[idx_1];

  return forest;

}

std::vector<Node*> TreeSwapLast(std::vector<Node*>& forest) {
  int num_tree = forest.size();
  int idx = sample_class(num_tree);

  // Rcout << "\nSelected tree = " << idx << "\n";

  Node* tree_1 = forest[idx];
  Node* tree_2 = forest[num_tree - 1];
  forest[num_tree-1] = tree_1;
  forest[idx] = tree_2;

  return forest;

}

std::vector<Node*> AddTree(std::vector<Node*>& forest,
                           const Hypers& hypers,
                           const Opts& opts) {
  std::vector<Node*> new_forest = forest;
  Node* new_root = new Node;
  new_root->GenTree(hypers);
  if(opts.update_tau)
    new_root->SetTau(Rf_rgamma(1.0, 1.0 / hypers.tau_rate));

  std::vector<Node*> leafs = leaves(new_root);
  for(int i = 0; i < leafs.size(); i++) {
    leafs[i]->mu = norm_rand() * hypers.sigma_mu;
  }

  new_forest.push_back(new_root);
  return new_forest;

}

std::vector<Node*> DeleteTree(std::vector<Node*>& forest) {

  std::vector<Node*> new_forest = TreeSwapLast(forest);
  new_forest.pop_back();
  return new_forest;

}

void update_num_tree(std::vector<Node*>& forest, Hypers& hypers,
                     const Opts& opts,
                     const arma::vec& Y, const arma::vec& res,
                     const arma::mat& X) {

  double add_or_delete = unif_rand();
  if(add_or_delete <= 0.5 || hypers.num_tree == 1) {
    // Rcout << "Birth step!";
    BirthTree(forest, hypers, opts, Y, res, X);
  }
  else {
    // Rcout << "Death step!";
    DeathTree(forest, hypers, Y, res, X);
  }

}

double LogLF(const std::vector<Node*>& forest, const Hypers& hypers,
             const arma::vec& Y, const arma::mat& X) {
  vec resid = Y - predict(forest, X, hypers);
  return loglik_normal(resid, hypers.sigma);
}

double loglik_normal(const arma::vec& resid, const double& sigma) {
  double N = resid.size();
  double SSE = dot(resid, resid);
  return -0.5 * N * log(M_2PI * pow(sigma, 2)) - 0.5 * SSE / pow(sigma, 2);
}

void BirthTree(std::vector<Node*>& forest,
               Hypers& hypers,
               const Opts& opts,
               const arma::vec& Y,
               const arma::vec& res,
               const arma::mat& X) {

  // Log likelihood of current state
  // Rcout << "1";
  double loglik_old = loglik_normal(res, hypers.sigma);

  // Add tree and modify hypers
  // Rcout << "2";
  std::vector<Node*> new_forest = AddTree(forest, hypers, opts);
  // Rcout << "3";
  RenormAddTree(forest, new_forest, hypers);

  // Calculate new log likelihood
  // Rcout << "4";
  double loglik_new = LogLF(new_forest, hypers, Y, X);

  // Do MH
  // Rcout << "5";
  double accept_ratio = loglik_new - loglik_old + TPrior(new_forest, hypers) - TPrior(forest, hypers);
  if(log(unif_rand()) < accept_ratio) {
    // Rcout << "6";
    forest = new_forest;
  }
  else {
    // Rcout << "7";
    UnnormAddTree(forest, new_forest, hypers);
    // Rcout << "8";
    delete new_forest.back();
  }

}

void DeathTree(std::vector<Node*>& forest,
               Hypers& hypers,
               const arma::vec& Y,
               const arma::vec& res,
               const arma::mat& X) {

  // Log likelihood of current state
  double loglik_old = loglik_normal(res, hypers.sigma);

  // Delete tree and modify hypers
  // Rcout << "Delete tree!";
  std::vector<Node*> new_forest = DeleteTree(forest);
  // Rcout << "Renorm!";
  RenormDeleteTree(forest, new_forest, hypers);

  // Calculate new log likelihood
  double loglik_new = LogLF(new_forest, hypers, Y, X);

  // Do MH
  // Rcout << "Do MH!";
  double accept_ratio = loglik_new - loglik_old + TPrior(new_forest, hypers) - TPrior(forest, hypers);
  if(log(unif_rand()) < accept_ratio) {
    // Rcout << "Accept fix!";
    delete forest.back();
    // Rcout << "Reass!";
    forest = new_forest;
  }
  else {
    // Rcout << "Reject fix!";
    UnnormDeleteTree(forest, new_forest, hypers);
  }

}

double TPrior(const std::vector<Node*>& forest, const Hypers& hypers) {
  int num_tree = forest.size();
  return log(hypers.num_tree_prob) +
    (num_tree - 1.0) * log(1.0 - hypers.num_tree_prob);
}

void RenormAddTree(std::vector<Node*>& forest,
                   std::vector<Node*>& new_forest,
                   Hypers& hypers) {

  int num_tree = forest.size();
  double factor = (double)num_tree / (num_tree + 1.0);
  factor = pow(factor, 0.5);

  // Increase number of trees
  hypers.num_tree = num_tree + 1;

  // Scale sigma_mu
  if(RESCALE) {
    hypers.sigma_mu = hypers.sigma_mu * factor;
    hypers.sigma_mu_hat = hypers.sigma_mu_hat * factor;

    // Scale the leaves
    for(int i = 0; i < new_forest.size(); i++) {
      std::vector<Node*> leafs = leaves(new_forest[i]);
      for(int j = 0; j < leafs.size(); j++) {
        leafs[j]->mu = factor * leafs[j]->mu;
      }
    }
  }
}

void UnnormAddTree(std::vector<Node*>& forest,
                   std::vector<Node*>& new_forest,
                   Hypers& hypers) {


  int num_tree = forest.size();
  double factor = (double)num_tree / (num_tree + 1.0);
  factor = pow(factor, -0.5);

  // Decrease number of trees
  hypers.num_tree = num_tree;

  // Descale sigma_mu
  if(RESCALE) {

    hypers.sigma_mu = hypers.sigma_mu * factor;
    hypers.sigma_mu_hat = hypers.sigma_mu_hat * factor;

    // Descale the leaves
    for(int i = 0; i < new_forest.size(); i++) {
      std::vector<Node*> leafs = leaves(new_forest[i]);
      for(int j = 0; j < leafs.size(); j++) {
        leafs[j]->mu = factor * leafs[j]->mu;
      }
    }
  }
}

void RenormDeleteTree(std::vector<Node*>& forest,
                      std::vector<Node*>& new_forest,
                      Hypers& hypers) {


  // Rcout << "1";
  int num_tree = forest.size();
  double factor = (double)num_tree / (num_tree - 1.0);
  factor = pow(factor, 0.5);

  // Rcout << "2";
  // Decrease number of trees
  hypers.num_tree = num_tree - 1;

  // Rcout << "3";
  // Descale sigma_mu
  if(RESCALE) {

    hypers.sigma_mu = hypers.sigma_mu * factor;
    hypers.sigma_mu_hat = hypers.sigma_mu_hat * factor;

    // Descale the leaves
    // Rcout << "4";
    for(int i = 0; i < new_forest.size(); i++) {
      std::vector<Node*> leafs = leaves(new_forest[i]);
      // Rcout << i;
      for(int j = 0; j < leafs.size(); j++) {
        leafs[j]->mu = factor * leafs[j]->mu;
      }
    }
  }

}

void UnnormDeleteTree(std::vector<Node*>& forest,
                      std::vector<Node*>& new_forest,
                      Hypers& hypers) {

  int num_tree = forest.size();
  double factor = (double)num_tree / (num_tree - 1.0);
  factor = pow(factor, -0.5);

  // Increase the number of trees
  hypers.num_tree = num_tree;

  // Rescale sigma_mu
  if(RESCALE) {

    hypers.sigma_mu = hypers.sigma_mu * factor;
    hypers.sigma_mu_hat = hypers.sigma_mu_hat * factor;

    // Descale the leaves
    for(int i = 0; i < new_forest.size(); i++) {
      std::vector<Node*> leafs = leaves(new_forest[i]);
      for(int j = 0; j < leafs.size(); j++) {
        leafs[j]->mu = factor * leafs[j]->mu;
      }
    }
  }
}

Node::~Node() {
  if(!is_leaf) {
    delete left;
    delete right;
  }
}

arma::mat Forest::do_gibbs(const arma::mat& X, const arma::vec& Y,
                           const arma::mat& X_test, int num_iter) {

  vec Y_hat = predict(trees, X, hypers);
  mat Y_out = zeros<mat>(num_iter, X_test.n_rows);

  int num_warmup = floor(opts.num_burn / 2.0);

  for(int i = 0; i < num_iter; i++) {
    if(opts.update_s && (num_gibbs > num_warmup)) {
      IterateGibbsWithS(trees, Y_hat, hypers, X, Y, opts);
    }
    else {
      IterateGibbsNoS(trees, Y_hat, hypers, X, Y, opts);
    }
    vec tmp = predict(trees, X_test, hypers);
    Y_out.row(i) = tmp.t();
    num_gibbs++;
    if(num_gibbs % opts.num_print == 0) {
      Rcout << "Finishing iteration " << num_gibbs << ": num_trees = " <<
        hypers.num_tree << std::endl;
    }
  }

  return Y_out;

}

void Forest::set_s(const arma::vec& s_) {
  hypers.s = s_;
  hypers.logs = log(s_);
}

arma::uvec Forest::get_counts() {
  return get_var_counts(trees, hypers);
}

arma::umat Forest::get_tree_counts() {
  for(int t = 0; t < hypers.num_tree; t++) {
    std::vector<Node*> tree;
    tree.resize(0);
    tree.push_back(trees[t]);
    tree_counts.col(t) = get_var_counts(tree, hypers);
  }

  return tree_counts;
}

RCPP_MODULE(mod_forest) {

  class_<Forest>("Forest")

    // .constructor<Rcpp::List>()
    .constructor<Rcpp::List, Rcpp::List>()
    .method("do_gibbs", &Forest::do_gibbs)
    .method("get_s", &Forest::get_s)
    .method("get_counts", &Forest::get_counts)
    .method("set_s", &Forest::set_s)
    .method("get_tree_counts", &Forest::get_tree_counts)
    .field("num_gibbs", &Forest::num_gibbs);

}
