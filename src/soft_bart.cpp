#include "soft_bart.h"

using namespace Rcpp;
using namespace arma;




Opts InitOpts(int num_burn, int num_thin, int num_save, int num_print,
              bool update_sigma_mu, bool update_s, bool update_alpha,
              bool update_beta, bool update_gamma) {

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

  return out;

}

Hypers InitHypers(const mat& X, const uvec& group, double sigma_hat,
                  double alpha, double beta,
                  double gamma, double k, double width, double shape,
                  int num_tree, double alpha_scale, double alpha_shape_1,
                  double alpha_shape_2) {

  int GRID_SIZE = 1000;

  Hypers out;
  out.alpha = alpha;
  out.beta = beta;
  out.gamma = gamma;
  out.sigma = sigma_hat;
  out.sigma_mu = 0.5 / (k * pow(num_tree, 0.5));
  out.shape = shape;
  out.width = width;
  out.num_tree = num_tree;

  out.num_groups = group.max() + 1;
  out.s = ones<vec>(out.num_groups) / ((double)(out.num_groups));
  out.logs = log(out.s);

  out.sigma_hat = sigma_hat;
  out.sigma_mu_hat = out.sigma_mu;

  out.alpha_scale = alpha_scale;
  out.alpha_shape_1 = alpha_shape_1;
  out.alpha_shape_2 = alpha_shape_2;

  out.group = group;

  // Create mapping of group to variables
  out.group_to_vars.resize(out.s.size());
  for(int i = 0; i < out.s.size(); i++) {
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

int Hypers::SampleVar() const {

  int group_idx = sample_class(s);
  int var_idx = sample_class(group_to_vars[group_idx].size());

  return group_to_vars[group_idx][var_idx];
}

void Node::Root() {
  is_leaf = true;
  is_root = true;
  left = this;
  right = this;
  parent = this;

  var = 0;
  val = 0.0;
  lower = 0.0;
  upper = 1.0;

  mu = 0.0;
  current_weight = 1.0;
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

}

void Node::BirthLeaves(const Hypers& hypers) {
  if(is_leaf) {
    AddLeaves();
    var = hypers.SampleVar();
    GetLimits();
    val = (upper - lower) * unif_rand() + lower;
  }
}

void Node::GenTree(const Hypers& hypers) {
  Root();
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
    n->GetW(X, i, hypers);
    for(int j = 0; j < num_leaves; j++) {
      w_i(j) = leafs[j]->current_weight;
    }
    mu_hat = mu_hat + y(i) * w_i;
    Lambda = Lambda + w_i * trans(w_i);
  }

  Lambda = Lambda / pow(hypers.sigma, 2);
  mu_hat = mu_hat / pow(hypers.sigma, 2);
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
  double out = -0.5 * N * log(M_2_PI * pow(hypers.sigma,2));
  out -= 0.5 * num_leaves * log(M_2_PI * pow(hypers.sigma_mu,2));
  double val, sign;
  log_det(val, sign, Omega_inv / M_2_PI);
  out -= 0.5 * val;
  out -= 0.5 * dot(Y, Y) / pow(hypers.sigma, 2);
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

double update_sigma(const arma::vec& r, double sigma_hat, double sigma_old) {

  double SSE = dot(r,r);
  int n = r.size();

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
  sigma = update_sigma(r, sigma_hat, sigma);
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
    n->GetW(X,i,hypers);
    for(int j = 0; j < num_leaves; j++) {
      out(i) = out(i) + leafs[j]->current_weight * leafs[j]->mu;
    }
  }

  return out;

}

void Node::GetW(const arma::mat& X, int i, const Hypers& hypers) {

  if(!is_leaf) {

    double weight = activation(X(i,var), val, hypers);
    left->current_weight = weight * current_weight;
    right->current_weight = (1 - weight) * current_weight;

    left->GetW(X,i,hypers);
    right->GetW(X,i,hypers);

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
    n->Root();
    forest.push_back(n);
  }
  return forest;
}

Rcpp::List do_soft_bart(const arma::mat& X,
                        const arma::vec& Y,
                        const arma::mat& X_test,
                        Hypers& hypers,
                        const Opts& opts) {


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
      bool UPDATE_TAU = true;
      if(UPDATE_TAU) hypers.update_tau(forest, X, Y);
    }

    if((i+1) % opts.num_print == 0) {
      Rcout << "Finishing warmup " << i + 1 << ": tau = " << hypers.width << "\n";
    }

  }

  // Make arguments to return
  mat Y_hat_train = zeros<mat>(opts.num_save, X.n_rows);
  mat Y_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  vec sigma = zeros<vec>(opts.num_save);
  vec sigma_mu = zeros<vec>(opts.num_save);
  vec alpha = zeros<vec>(opts.num_save);
  vec beta = zeros<vec>(opts.num_save);
  vec gamma = zeros<vec>(opts.num_save);
  mat s = zeros<mat>(opts.num_save, hypers.s.size());
  umat var_counts = zeros<umat>(opts.num_save, hypers.s.size());

  // Do save iterations
  for(int i = 0; i < opts.num_save; i++) {
    for(int b = 0; b < opts.num_thin; b++) {
      IterateGibbsWithS(forest, Y_hat, hypers, X, Y, opts);
      bool UPDATE_TAU = true;
      if(UPDATE_TAU) hypers.update_tau(forest, X, Y);
    }

    // Save stuff
    Y_hat_train.row(i) = Y_hat.t();
    Y_hat_test.row(i) = trans(predict(forest, X_test, hypers));
    sigma(i) = hypers.sigma;
    sigma_mu(i) = hypers.sigma_mu;
    s.row(i) = trans(hypers.s);
    var_counts.row(i) = trans(get_var_counts(forest, hypers));
    alpha(i) = hypers.alpha;
    beta(i) = hypers.beta;
    gamma(i) = hypers.gamma;


    if((i + 1) % opts.num_print == 0) {
      Rcout << "Finishing save " << i + 1 << ": tau = " << hypers.width << "\n";
    }

  }

  Rcout << "Number of leaves at final iterations:\n";
  for(int t = 0; t < hypers.num_tree; t++) {
    Rcout << leaves(forest[t]).size() << " ";
    if((t + 1) % 10 == 0) Rcout << "\n";
  }

  List out;
  out["y_hat_train"] = Y_hat_train;
  out["y_hat_test"] = Y_hat_test;
  out["sigma"] = sigma;
  out["sigma_mu"] = sigma_mu;
  out["s"] = s;
  out["alpha"] = alpha;
  out["beta"] = beta;
  out["gamma"] = gamma;
  out["var_counts"] = var_counts;


  return out;

}

void IterateGibbsWithS(std::vector<Node*>& forest, arma::vec& Y_hat,
                       Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                       const Opts& opts) {
  IterateGibbsNoS(forest, Y_hat, hypers, X, Y, opts);
  if(opts.update_s) UpdateS(forest, hypers);
  if(opts.update_alpha) hypers.UpdateAlpha();
}

void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat,
                     Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                     const Opts& opts) {


  // Rcout << "Backfitting trees";
  TreeBackfit(forest, Y_hat, hypers, X, Y);
  arma::vec res = Y - Y_hat;
  arma::vec means = get_means(forest);

  // Rcout << "Doing other updates";
  hypers.UpdateSigma(res);
  if(opts.update_sigma_mu) hypers.UpdateSigmaMu(means);
  if(opts.update_beta) hypers.UpdateBeta(forest);
  if(opts.update_gamma) hypers.UpdateGamma(forest);

  Rcpp::checkUserInterrupt();
}

void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,
                 const Hypers& hypers, const arma::mat& X, const arma::vec& Y) {

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
    forest[t]->UpdateMu(res, X, hypers);
    Y_hat = Y_star + predict(forest[t], X, hypers);
  }
}

double activation(double x, double c, const Hypers& hypers) {
  return 1.0 - expit((x - c) / hypers.width);
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
  branch->var = hypers.SampleVar();
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
  // Rcout << "Getting leafs";
  std::vector<Node*> leafs = leaves(tree);
  // Rcout << "Selecting leafs\n";
  // Rcout << "number of leafs" << leafs.size();
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
  arma::uvec counts = zeros<uvec>(hypers.s.size());
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

void UpdateS(std::vector<Node*>& forest, Hypers& hypers) {

  // vec shape_up = hypers.alpha / ((double)hypers.s.size()) * ones<vec>(hypers.s.size()) +
  //   get_var_counts(forest, hypers);
  vec shape_up = zeros<vec>(hypers.s.size());
  double a = hypers.alpha / ((double)hypers.s.size());
  uvec counts = get_var_counts(forest, hypers);

  for(int i = 0; i < shape_up.size(); i++) {
    shape_up(i) = a + counts(i);
    hypers.logs(i) = rlgam(shape_up(i));
  }
  // Rcout << "Logs min" << hypers.logs.min() << "\n";
  // Rcout << "Logs max" << hypers.logs.max() << "\n";
  hypers.logs = hypers.logs - log_sum_exp(hypers.logs);
  hypers.s = exp(hypers.logs);

  // Rcout << "Logs min" << hypers.logs.min() << "\n";
  // Rcout << "Logs max" << hypers.logs.max() << "\n";
  // Rcout << "s min" << hypers.s.min() << "\n";
  // Rcout << "s max" << hypers.s.max() << "\n";
  // Rcout << "Sum of s" << sum(hypers.s) << "\n";
  // hypers.s = rdirichlet(shape_up);
}

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

void Hypers::UpdateAlpha() {
  arma::vec logliks = zeros<vec>(rho_propose.size());

  rho_loglik loglik;
  loglik.mean_log_s = mean(logs);
  loglik.p = (double)s.size();
  loglik.alpha_scale = alpha_scale;
  loglik.alpha_shape_1 = alpha_shape_1;
  loglik.alpha_shape_2 = alpha_shape_2;

  for(int i = 0; i < rho_propose.size(); i++) {
    logliks(i) = loglik(rho_propose(i));
  }

  logliks = exp(logliks - log_sum_exp(logliks));
  double rho_up = rho_propose(sample_class(logliks));
  alpha = rho_to_alpha(rho_up, alpha_scale);

}

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

// [[Rcpp::export]]
List SoftBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test,
              const arma::uvec& group,
              double alpha, double beta, double gamma, double sigma,
              double shape, double width, int num_tree,
              double sigma_hat, double k, double alpha_scale,
              double alpha_shape_1, double alpha_shape_2, int num_burn,
              int num_thin, int num_save, int num_print, bool update_sigma_mu,
              bool update_s, bool update_alpha, bool update_beta, bool update_gamma) {


  Opts opts = InitOpts(num_burn, num_thin, num_save, num_print, update_sigma_mu,
                       update_s, update_alpha, update_beta, update_gamma);

  Hypers hypers = InitHypers(X, group, sigma_hat, alpha, beta, gamma, k, width,
                             shape, num_tree, alpha_scale, alpha_shape_1,
                             alpha_shape_2);

  // Rcout << "Doing soft_bart\n";
  return do_soft_bart(X,Y,X_test,hypers,opts);

}

// [[Rcpp::export]]
bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new) {

  double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;

  return log(unif_rand()) < cutoff ? true : false;

}

double logprior_tau(double tau) {
  int DO_LOG = 1;
  return Rf_dexp(tau, 0.1, DO_LOG);
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

void Hypers::update_tau(std::vector<Node*>& forest,
                           const arma::mat& X, const arma::vec& Y) {

  double tau_old = width;
  double tau_new = tau_proposal(tau_old);

  double loglik_new = loglik_tau(tau_new, forest, X, Y) + logprior_tau(tau_new);
  double new_to_old = log_tau_trans(tau_old);
  double loglik_old = loglik_tau(tau_old, forest, X, Y) + logprior_tau(tau_old);
  double old_to_new = log_tau_trans(tau_new);

  bool accept_mh = do_mh(loglik_new, loglik_old, new_to_old, old_to_new);
  width = accept_mh ? tau_new : tau_old;

}
