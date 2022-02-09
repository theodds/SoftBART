#include "soft_bart.h"

using namespace Rcpp;
using namespace arma;

bool RESCALE = true;

Forest::Forest(Rcpp::List hypers_, Rcpp::List opts_) : hypers(hypers_), opts(opts_) {
  trees.resize(hypers.num_tree);
  for(int i = 0; i < hypers.num_tree; i++) {
    trees[i] = new Node();
    trees[i]->Root(hypers);
    // trees[i]->GenTree(hypers);
  }
  num_gibbs = 0;
  tree_counts = zeros<umat>(hypers.split_hypers.counts.size(), hypers.num_tree);
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
              bool update_sigma_mu, bool update_s,
              bool update_beta, bool update_gamma, bool update_tau,
              bool update_tau_mean, bool update_sigma) {

  Opts out;
  out.num_burn = num_burn;
  out.num_thin = num_thin;
  out.num_save = num_save;
  out.num_print = num_print;
  out.update_sigma_mu = update_sigma_mu;
  out.update_s = update_s;
  out.update_beta = update_beta;
  out.update_gamma = update_gamma;
  out.update_tau = update_tau;
  out.update_tau_mean = update_tau_mean;
  out.update_sigma = update_sigma;

  return out;

}

Hypers InitHypers(const mat& X, const uvec& group, double sigma_hat,
                  double alpha, double beta,
                  double gamma, double k, double width, double shape,
                  int num_tree, double tau_rate, double temperature,
                  arma::vec log_prior) {

  Hypers out;

  out.split_hypers.LoadGroups(group);
  out.split_hypers.log_V.clear();
  out.split_hypers.use_counts = false;
  out.split_hypers.dirichlet_mass = alpha;
  out.split_hypers.log_mass = log(alpha);
  out.split_hypers.log_prior = log_prior;


  out.beta = beta;
  out.gamma = gamma;
  out.sigma = sigma_hat;
  out.sigma_mu = 0.5 / (k * pow(num_tree, 0.5));
  out.shape = shape;
  out.width = width;
  out.num_tree = num_tree;

  out.sigma_hat = sigma_hat;
  out.sigma_mu_hat = out.sigma_mu;

  out.tau_rate = tau_rate;
  out.temperature = temperature;

  return out;
}

int ProbHypers::ResampleVar(int var) {

  counts(group(var)) -= 1;
  int sampled_var = SampleVar();
  counts(group(sampled_var)) += 1;
  return sampled_var;
}

void ProbHypers::SwitchVar(int v_old, int v_new) {
  counts(group(v_old)) -= 1;
  counts(group(v_new)) += 1;
}

int ProbHypers::SampleVar() {

  int group_idx = counts.size() - 1;
  int var_idx;

  if(!use_counts) {
    group_idx = sample_class(counts.size());
    var_idx =  sample_class(group_to_vars[group_idx].size());
    return group_to_vars[group_idx][var_idx];
  }

  double U = R::unif_rand();
  double cumsum = 0.0;
  int K = counts.size();
  int num_branches= sum(counts);
  int num_active = counts.n_nonzero;


  for(int k = 0; k < K; k++) {
    if(counts(k) == 0) {
      double tmp = calc_log_v(num_branches + 1, num_active + 1);
      tmp -= calc_log_v(num_branches, num_active);
      tmp -= log(K - num_active);
      tmp += log_mass;
      cumsum += exp(tmp);
    }
    else {
      double tmp = calc_log_v(num_branches + 1, num_active);
      tmp -= calc_log_v(num_branches, num_active);
      tmp += log(dirichlet_mass + counts(k));
      cumsum += exp(tmp);
    }
    if(U < cumsum) {
      group_idx = k;
      break;
    }
  }

  var_idx = sample_class(group_to_vars[group_idx].size());
  return group_to_vars[group_idx][var_idx];

}

void Node::Root(Hypers& hypers) {
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

}

void Node::BirthLeaves(Hypers& hypers) {
  if(is_leaf) {
    AddLeaves();
    var = hypers.split_hypers.SampleVar();
    hypers.split_hypers.counts(hypers.split_hypers.group(var)) += 1;
    GetLimits();
    val = (upper - lower) * unif_rand() + lower;
  }
}

void Node::GenTree(Hypers& hypers) {
  Root(hypers);
  GenBelow(hypers);
}

void Node::GenBelow(Hypers& hypers) {
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
void GetSuffStats(Node* n, const arma::vec& y, const arma::vec& weights,
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
    mu_hat = mu_hat + y(i) * w_i * weights(i);
    Lambda = Lambda + w_i * trans(w_i) * weights(i);
  }

  Lambda = Lambda / pow(hypers.sigma, 2) * hypers.temperature;
  mu_hat = mu_hat / pow(hypers.sigma, 2) * hypers.temperature;
  Omega_inv_out = Lambda + eye(num_leaves, num_leaves) / pow(hypers.sigma_mu, 2);
  mu_hat_out = solve(Omega_inv_out, mu_hat);

}

double LogLT(Node* n, const arma::vec& Y, const arma::vec& weights,
             const arma::mat& X, const Hypers& hypers) {

  // Rcout << "Leaves ";
  std::vector<Node*> leafs = leaves(n);
  int num_leaves = leafs.size();

  // Get sufficient statistics
  arma::vec mu_hat = zeros<vec>(num_leaves);
  arma::mat Omega_inv = zeros<mat>(num_leaves, num_leaves);
  GetSuffStats(n, Y, weights, X, hypers, mu_hat, Omega_inv);

  int N = Y.size();

  // Rcout << "Compute ";
  // double out = -0.5 * N * log(M_2PI * pow(hypers.sigma,2)) * hypers.temperature;
  double out = 0.5 * sum(log(weights / M_2PI / pow(hypers.sigma,2))) * hypers.temperature;
  out -= 0.5 * num_leaves * log(M_2PI * pow(hypers.sigma_mu,2));
  double val, sign;
  log_det(val, sign, Omega_inv / M_2PI);
  out -= 0.5 * val;
  out -= 0.5 * dot(Y, Y % weights) / pow(hypers.sigma, 2) * hypers.temperature;
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

double update_sigma(const arma::vec& r, const arma::vec&weights,
                    double sigma_hat, double sigma_old,
                    double temperature) {

  double SSE = dot(r,r % weights) * temperature;
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

void Hypers::UpdateSigma(const arma::vec& r, const arma::vec& weights) {
  sigma = update_sigma(r, weights, sigma_hat, sigma, temperature);
}

void Hypers::UpdateSigmaMu(const arma::vec& means) {
  sigma_mu = update_sigma(means, sigma_mu_hat, sigma_mu);
}

void Node::UpdateMu(const arma::vec& Y, const arma::vec& weights,
                    const arma::mat& X, const Hypers& hypers) {

  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();

  // Get mean and covariance
  vec mu_hat = zeros<vec>(num_leaves);
  mat Omega_inv = zeros<mat>(num_leaves, num_leaves);
  GetSuffStats(this, Y, weights, X, hypers, mu_hat, Omega_inv);

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

void Node::DeleteLeaves(Hypers& hypers) {
  hypers.split_hypers.counts(hypers.split_hypers.group(var)) -= 1;
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
                               Hypers& hypers) {

  std::vector<Node*> forest(0);
  for(int t = 0; t < hypers.num_tree; t++) {
    Node* n = new Node;
    n->Root(hypers);
    forest.push_back(n);
  }
  return forest;
}

Rcpp::List do_soft_bart(const arma::mat& X,
                        const arma::vec& Y,
                        const arma::vec& weights,
                        const arma::mat& X_test,
                        Hypers& hypers,
                        const Opts& opts) {


  std::vector<Node*> forest = init_forest(X, Y, hypers);

  vec Y_hat = zeros<vec>(X.n_rows);

  // Do burn_in

  for(int i = 0; i < opts.num_burn; i++) {


    IterateGibbsNoS(forest, Y_hat, weights, hypers, X, Y, opts);

    // Don't update s for half of the burn-in
    if((i == floor(0.5 * opts.num_burn)) && opts.update_s) {
      hypers.split_hypers.use_counts = true;
    }

    if((i+1) % opts.num_print == 0) {
      Rcout << "Finishing warmup " << i + 1
               << " Number of branches = " << sum(hypers.split_hypers.counts)
            << "\n"
      
        ;
    }

  }

  // Make arguments to return
  mat Y_hat_train = zeros<mat>(opts.num_save, X.n_rows);
  mat Y_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  vec sigma = zeros<vec>(opts.num_save);
  vec sigma_mu = zeros<vec>(opts.num_save);
  vec beta = zeros<vec>(opts.num_save);
  vec gamma = zeros<vec>(opts.num_save);
  umat var_counts = zeros<umat>(opts.num_save, hypers.split_hypers.counts.size());
  vec tau_rate = zeros<vec>(opts.num_save);
  vec loglik = zeros<vec>(opts.num_save);
  mat loglik_train = zeros<mat>(opts.num_save, Y_hat.size());

  // Do save iterations
  for(int i = 0; i < opts.num_save; i++) {
    for(int b = 0; b < opts.num_thin; b++) {
      IterateGibbsNoS(forest, Y_hat, weights, hypers, X, Y, opts);
    }

    // Save stuff
    Y_hat_train.row(i) = Y_hat.t();
    Y_hat_test.row(i) = trans(predict(forest, X_test, hypers));
    sigma(i) = hypers.sigma;
    sigma_mu(i) = hypers.sigma_mu;
    var_counts.row(i) = trans(get_var_counts(forest, hypers));
    beta(i) = hypers.beta;
    gamma(i) = hypers.gamma;
    tau_rate(i) = hypers.tau_rate;
    loglik_train.row(i) = trans(loglik_data(Y,weights,Y_hat,hypers));
    loglik(i) = sum(loglik_train.row(i));

    if((i + 1) % opts.num_print == 0) {
      Rcout << "Finishing save " << i + 1 << " Number of branches = "
            << sum(hypers.split_hypers.counts)
            << "\n";
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
  out["beta"] = beta;
  out["gamma"] = gamma;
  out["var_counts"] = var_counts;
  out["tau_rate"] = tau_rate;
  out["loglik"] = loglik;
  out["loglik_train"] = loglik_train;


  return out;

}

void IterateGibbsNoS(std::vector<Node*>& forest, arma::vec& Y_hat,
                     const arma::vec& weights,
                     Hypers& hypers, const arma::mat& X, const arma::vec& Y,
                     const Opts& opts) {


  // Rcout << "Backfitting trees";
  TreeBackfit(forest, Y_hat, weights, hypers, X, Y, opts);
  arma::vec res = Y - Y_hat;
  arma::vec means = get_means(forest);

  // Rcout << "Doing other updates";
  if(opts.update_sigma) hypers.UpdateSigma(res, weights);
  if(opts.update_sigma_mu) hypers.UpdateSigmaMu(means);
  if(opts.update_beta) hypers.UpdateBeta(forest);
  if(opts.update_gamma) hypers.UpdateGamma(forest);
  if(opts.update_tau_mean) hypers.UpdateTauRate(forest);

  Rcpp::checkUserInterrupt();
}

void TreeBackfit(std::vector<Node*>& forest, arma::vec& Y_hat,
                 const arma::vec& weights, Hypers& hypers, const arma::mat& X,
                 const arma::vec& Y,
                 const Opts& opts) {

  double MH_BD = 0.7;
  double MH_PRIOR = 0.4;

  int num_tree = hypers.num_tree;
  for(int t = 0; t < num_tree; t++) {
    // Rcout << "Getting backfit quantities";
    arma::vec Y_star = Y_hat - predict(forest[t], X, hypers);
    arma::vec res = Y - Y_star;

    if(unif_rand() < MH_PRIOR) {
      forest[t] = draw_prior(forest[t], X, res, weights, hypers);
    }
    if(forest[t]->is_leaf || unif_rand() < MH_BD) {
      birth_death(forest[t], X, res, weights, hypers);
    }
    else {
      perturb_decision_rule(forest[t], X, res, weights, hypers);
    }
    if(opts.update_tau) forest[t]->UpdateTau(res, weights, X, hypers);
    forest[t]->UpdateMu(res, weights, X, hypers);
    Y_hat = Y_star + predict(forest[t], X, hypers);
  }
}

double activation(double x, double c, double tau) {
  return 1.0 - expit((x - c) / tau);
}

void birth_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const arma::vec& weights, Hypers& hypers) {


  double p_birth = probability_node_birth(tree);

  if(unif_rand() < p_birth) {
    node_birth(tree, X, Y, weights, hypers);
  }
  else {
    node_death(tree, X, Y, weights, hypers);
  }
}

void node_birth(Node* tree, const arma::mat& X, const arma::vec& Y,
                const arma::vec& weights,
                Hypers& hypers) {

  // Rcout << "Sample leaf";
  double leaf_probability = 0.0;
  Node* leaf = birth_node(tree, &leaf_probability);

  // Rcout << "Compute prior";
  int leaf_depth = depth(leaf);
  double leaf_prior = growth_prior(leaf_depth, hypers);

  // Get likelihood of current state
  // Rcout << "Current likelihood";
  double ll_before = LogLT(tree, Y, weights, X, hypers);
  ll_before += log(1.0 - leaf_prior);

  // Get transition probability
  // Rcout << "Transistion";
  double p_forward = log(probability_node_birth(tree) * leaf_probability);

  // Birth new leaves
  // Rcout << "Birth";
  leaf->BirthLeaves(hypers); // THIS INCREMENTS SPLIT_HYPERS

  // Get likelihood after
  // Rcout << "New Likelihood";
  double ll_after = LogLT(tree, Y, weights, X, hypers);
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
    leaf->DeleteLeaves(hypers); // THIS DECREMENTS SPLIT_HYPERS
    leaf->var = 0;
  }
  else {
    // Rcout << "Accept!";
  }
}

void node_death(Node* tree, const arma::mat& X, const arma::vec& Y,
                const arma::vec& weights,
                Hypers& hypers) {

  // Select branch to kill Children
  double p_not_grand = 0.0;
  Node* branch = death_node(tree, &p_not_grand);

  // Compute before likelihood
  int leaf_depth = depth(branch->left);
  double leaf_prob = growth_prior(leaf_depth - 1, hypers);
  double left_prior = growth_prior(leaf_depth, hypers);
  double right_prior = growth_prior(leaf_depth, hypers);
  double ll_before = LogLT(tree, Y, weights, X, hypers) +
    log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob);

  // Compute forward transition prob
  double p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)));

  // Save old leafs, do not delete (they are dangling, need to be handled by the end)
  SubtractTreeCounts(hypers.split_hypers, branch); // MUST ADD BACK AT END
  Node* left = branch->left;
  Node* right = branch->right;
  branch->left = branch;
  branch->right = branch;
  branch->is_leaf = true;

  // Compute likelihood after
  double ll_after = LogLT(tree, Y, weights, X, hypers) + log(1.0 - leaf_prob);

  // Compute backwards transition
  std::vector<Node*> leafs = leaves(tree);
  double p_backwards = log(1.0 / ((double)(leafs.size())) * probability_node_birth(tree));

  // Do MH and fix dangles
  double log_trans_prob = ll_after + p_backwards - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    branch->left = left;
    branch->right = right;
    branch->is_leaf = false;
    AddTreeCounts(hypers.split_hypers, branch);
  }
  else {
    delete left;
    delete right;
  }
}

void branches(Node* n, std::vector<Node*>& branch_vec) {
  if(!(n->is_leaf)) {
    branch_vec.push_back(n);
    branches(n->left, branch_vec);
    branches(n->right, branch_vec);
  }
}

std::vector<Node*> branches(Node* root) {
  std::vector<Node*> branch_vec;
  branch_vec.resize(0);
  branches(root, branch_vec);
  return branch_vec;
}



double calc_cutpoint_likelihood(Node* node) {
  if(node->is_leaf) return 1;
  
  double out = 1.0/(node->upper - node->lower);
  out = out * calc_cutpoint_likelihood(node->left);
  out = out * calc_cutpoint_likelihood(node->right);
  
  return out;
}

std::vector<double> get_perturb_limits(Node* branch) {
  double min = 0.0;
  double max = 1.0;
  
  Node* n = branch;
  while(!(n->is_root)) {
    if(n->is_left()) {
      n = n->parent;
      if(n->var == branch->var) {
        if(n->val > min) {
          min = n->val;
        }
      }
    }
    else {
      n = n->parent;
      if(n->var == branch->var) {
        if(n->val < max) {
          max = n->val;
        }
      }
    }
  }
  std::vector<Node*> left_branches = branches(n->left);
  std::vector<Node*> right_branches = branches(n->right);
  for(int i = 0; i < left_branches.size(); i++) {
    if(left_branches[i]->var == branch->var) {
      if(left_branches[i]->val > min)
        min = left_branches[i]->val;
    }
  }
  for(int i = 0; i < right_branches.size(); i++) {
    if(right_branches[i]->var == branch->var) {
      if(right_branches[i]->val < max) {
        max = right_branches[i]->val;
      }
    }
  }
  
  std::vector<double> out; out.push_back(min); out.push_back(max);
  return out;
}

void get_limits_below(Node* node) {
  node->GetLimits();
  if(!(node->left->is_leaf)) {
    get_limits_below(node->left);
  }
  if(!(node->right->is_leaf)) {
    get_limits_below(node->right);
  }
}

void perturb_decision_rule(Node* tree,
                           const arma::mat& X,
                           const arma::vec& Y,
                           const arma::vec& weights,
                           Hypers& hypers) {
  
  // Randomly choose a branch; if no branches, we automatically reject
  std::vector<Node*> bbranches = branches(tree);
  if(bbranches.size() == 0)
    return;
  
  // Select the branch
  Node* branch = rand(bbranches);
  
  // Calculuate tree likelihood before proposal
  double ll_before = LogLT(tree, Y, weights, X, hypers);
  
  // Calculate product of all 1/(B - A) here
  double cutpoint_likelihood = calc_cutpoint_likelihood(tree);
  
  // Calculate backward transition density
  std::vector<double> lims = get_perturb_limits(branch);
  double backward_trans = 1.0/(lims[1] - lims[0]);
  
  // save old split
  int old_feature = branch->var;
  double old_value = branch->val;
  double old_lower = branch->lower;
  double old_upper = branch->upper;
  
  // Modify the branch
  // branch->var = hypers.split_hypers.SampleVar();
  branch->var = hypers.split_hypers.ResampleVar(branch->var);
  // branch->GetLimits();
  lims = get_perturb_limits(branch);
  branch->val = lims[0] + (lims[1] - lims[0]) * unif_rand();
  get_limits_below(branch);
  
  // Calculate likelihood after proposal
  double ll_after = LogLT(tree, Y, weights, X, hypers);
  
  // Calculate product of all 1/(B-A)
  double cutpoint_likelihood_after = calc_cutpoint_likelihood(tree);
  
  // Calculate forward transition density
  double forward_trans = 1.0/(lims[1] - lims[0]);
  
  // Do MH
  double log_trans_prob =
    ll_after + log(cutpoint_likelihood_after) + log(backward_trans)
    - ll_before - log(cutpoint_likelihood) - log(forward_trans);
  
  if(log(unif_rand()) > log_trans_prob) {

    // Reset branch vars
    hypers.split_hypers.SwitchVar(branch->var, old_feature);

    branch->var = old_feature;
    branch->val = old_value;
    branch->lower = old_lower;
    branch->upper = old_upper;
    get_limits_below(branch);
  }
}

Node* draw_prior(Node* tree, const arma::mat& X, const arma::vec& Y,
                 const arma::vec& weights, Hypers& hypers) {
  
  // Compute loglik before
  Node* tree_0 = tree;
  double loglik_before = LogLT(tree_0, Y, weights, X, hypers);
  
  // GIBBS: NEED TO REMOVE THE TREE COUNTS AND ADD BACK
  SubtractTreeCounts(hypers.split_hypers, tree);

  // Make new tree and compute loglik after
  Node* tree_1 = new Node;
  tree_1->Root(hypers);
  tree_1->GenBelow(hypers);
  double loglik_after = LogLT(tree_1, Y, weights, X, hypers);
  
  // Do MH
  if(log(unif_rand()) < loglik_after - loglik_before) {

    delete tree_0;
    tree = tree_1;
  }
  else {
    
    // If rejected , need to add the old counts to the tree
    SubtractTreeCounts(hypers.split_hypers, tree_1);
    AddTreeCounts(hypers.split_hypers, tree_0);
    delete tree_1;

  }
  return tree;
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
  arma::uvec counts = zeros<uvec>(hypers.split_hypers.counts.size());
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t], hypers);
  }
  return counts;
}

void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers) {
  if(!node->is_leaf) {
    int group_idx = hypers.split_hypers.group(node->var);
    counts(group_idx) = counts(group_idx) + 1;
    get_var_counts(counts, node->left, hypers);
    get_var_counts(counts, node->right, hypers);
  }
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

double logpdf_beta(double x, double a, double b) {
  return (a-1.0) * log(x) + (b-1.0) * log(1 - x) - Rf_lbeta(a,b);
}

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

arma::vec loglik_data(const arma::vec& Y, const arma::vec& weights,
                      const arma::vec& Y_hat, const Hypers& hypers) {
  vec res = Y - Y_hat;
  vec out = zeros<vec>(Y.size());
  for(int i = 0; i < Y.size(); i++) {
    out(i) = -0.5 * log(M_2PI * pow(hypers.sigma,2) / weights(i))
      - 0.5 * weights(i) * pow(res(i) / hypers.sigma, 2);
  }
  return out;
}

// [[Rcpp::export]]
List SoftBart(const arma::mat& X, const arma::vec& Y, const arma::mat& X_test,
              const arma::uvec& group,
              double alpha, double beta, double gamma, double sigma,
              double shape, double width, int num_tree,
              double sigma_hat, double k, double tau_rate,
              double temperature, const arma::vec& weights,
              int num_burn,
              int num_thin, int num_save, int num_print, bool update_sigma_mu,
              bool update_s, bool update_beta, bool update_gamma,
              bool update_tau, bool update_tau_mean, bool update_sigma,
              arma::vec log_prior) {


  Opts opts = InitOpts(num_burn, num_thin, num_save, num_print, update_sigma_mu,
                       update_s, update_beta, update_gamma,
                       update_tau, update_tau_mean, update_sigma);

  Hypers hypers = InitHypers(X, group, sigma_hat, alpha, beta, gamma, k, width,
                             shape, num_tree, tau_rate, temperature, log_prior);

  // Rcout << "Doing soft_bart\n";
  return do_soft_bart(X, Y, weights, X_test, hypers, opts);

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
                        const arma::vec& Y, const arma::vec& weights,
                        const Hypers& hypers) {

  double tau_old = tau;
  SetTau(tau_new);
  double out = LogLT(this, Y, weights, X, hypers);
  SetTau(tau_old);
  return out;

}

void Node::UpdateTau(const arma::vec& Y,
                     const arma::vec& weights,
                     const arma::mat& X,
                     const Hypers& hypers) {

  double tau_old = tau;
  double tau_new = tau_proposal(tau);

  double loglik_new = loglik_tau(tau_new, X, Y, weights, hypers) + logprior_tau(tau_new, hypers.tau_rate);
  double loglik_old = loglik_tau(tau_old, X, Y, weights, hypers) + logprior_tau(tau_old, hypers.tau_rate);
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
  beta = 2.0;
  gamma = 0.95;
}

Hypers::Hypers(Rcpp::List hypers) {
  beta = hypers["beta"];
  gamma = hypers["gamma"];
  sigma = hypers["sigma"];
  sigma_mu = hypers["sigma_mu"];
  sigma_mu_hat = sigma_mu;
  shape = hypers["shape"];
  width = hypers["width"];
  num_tree = hypers["num_tree"];
  sigma_hat = hypers["sigma_hat"];
  tau_rate = hypers["tau_rate"];
  temperature = hypers["temperature"];

  // Deal with group and num_group
  arma::uvec group = as<arma::uvec>(hypers["group"]);
  int num_groups = group.max() + 1;
  split_hypers.LoadGroups(group);

  // Gibbs Prior
  split_hypers.log_V.clear();
  split_hypers.use_counts = false;
  double alpha = hypers["alpha"];
  split_hypers.dirichlet_mass = alpha;
  split_hypers.log_mass = log(alpha);
  split_hypers.log_prior = as<arma::vec>(hypers["log_prior"]);
}


// Global tau stuff
double logprior_tau(double tau, double tau_rate) {
  int DO_LOG = 1;
  return Rf_dexp(tau, 1.0 / tau_rate, DO_LOG);
}

double tau_proposal(double tau) {
  double U = 2.0 * unif_rand() - 1;
  return pow(5.0, U) * tau;
}

// double Hypers::loglik_tau(double tau,
//                           const std::vector<Node*>& forest,
//                           const arma::mat& X, const arma::vec& Y) {

//   double tau_old = width;
//   width = tau;
//   vec Y_hat = predict(forest, X, *this);
//   double SSE = dot(Y - Y_hat, Y - Y_hat);
//   double sigma_sq = pow(sigma, 2);

//   double loglik = -0.5 * Y.size() * log(sigma_sq) - 0.5 * SSE / sigma_sq;

//   width = tau_old;
//   return loglik;

// }

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


ProbHypers::ProbHypers(double d_mass,
                       arma::vec log_p,
                       bool use_c,
                       arma::uvec group)
  : use_counts(use_c), dirichlet_mass(d_mass), log_prior(log_p)
{
  log_V.clear();
  log_mass = log(d_mass);
  LoadGroups(group);
}

void ProbHypers::LoadGroups(arma::uvec group) {
  this->group = group;
  int num_groups = group.max() + 1;
  counts = zeros<sp_umat>(num_groups,1);
  int P = group.size();
  group_to_vars.resize(num_groups);
  for(int i = 0; i < num_groups; i++) {
    group_to_vars[i].resize(0);
  }
  for(int j = 0; j < P; j++) {
    int idx = group(j);
    group_to_vars[idx].push_back(j);
  }
}

ProbHypers::ProbHypers() {
}

double ProbHypers::calc_log_v(int n, int t) {
  std::pair<int,int> nt(n,t);

  std::map<std::pair<int,int>,double>::iterator iter = log_V.find(nt);
  if(iter != log_V.end()) {
    return iter->second;
  }

  int D = log_prior.size();
  std::vector<double> log_terms;
  for(int k = t; k <= D; k++) {
    double log_term = log_prior(k-1);
    log_term += R::lgammafn(k + 1) - R::lgammafn(k - t + 1);
    log_term += R::lgammafn(dirichlet_mass * k) 
      - R::lgammafn(dirichlet_mass * k + n);
    log_terms.push_back(log_term);
  }
  arma::vec log_terms_arma(log_terms);
  log_V[nt] = log_sum_exp(log_terms_arma);
  return log_V[nt];

}

void AddTreeCounts(ProbHypers& split_hypers, Node* node) {
  if(!(node->is_leaf)) {
    split_hypers.counts(split_hypers.group(node->var))
      = split_hypers.counts(split_hypers.group(node->var)) + 1;
    AddTreeCounts(split_hypers, node->left);
    AddTreeCounts(split_hypers, node->right);
  }
}

void SubtractTreeCounts(ProbHypers& split_hypers, Node* node) {
  if(!(node->is_leaf)) {
    split_hypers.counts(split_hypers.group(node->var))
      = split_hypers.counts(split_hypers.group(node->var)) - 1;
    SubtractTreeCounts(split_hypers, node->left);
    SubtractTreeCounts(split_hypers, node->right);
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
  arma::vec weights = arma::ones<arma::vec>(Y.n_elem);

  int num_warmup = floor(opts.num_burn / 2.0);

  for(int i = 0; i < num_iter; i++) {
    if(opts.update_s && (num_gibbs > num_warmup)) {
      hypers.split_hypers.use_counts = true;
    }
    IterateGibbsNoS(trees, Y_hat, weights, hypers, X, Y, opts);
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

arma::mat Forest::do_gibbs_weighted(const arma::mat& X, const arma::vec& Y,
                                    const arma::vec& weights,
                                    const arma::mat& X_test, int num_iter) {

  vec Y_hat = predict(trees, X, hypers);
  mat Y_out = zeros<mat>(num_iter, X_test.n_rows);

  int num_warmup = floor(opts.num_burn / 2.0);

  for(int i = 0; i < num_iter; i++) {
    if(opts.update_s && (num_gibbs > num_warmup)) {
      hypers.split_hypers.use_counts = true;
    }
    IterateGibbsNoS(trees, Y_hat, weights, hypers, X, Y, opts);
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

void Forest::set_sigma(double sigma) {
  hypers.sigma = sigma;
}

double Forest::get_sigma() {
  return hypers.sigma;
}

arma::vec Forest::do_predict(arma::mat& X) {
  return(predict(trees, X, hypers));
}

RCPP_MODULE(mod_forest) {

  class_<Forest>("Forest")

    // .constructor<Rcpp::List>()
    .constructor<Rcpp::List, Rcpp::List>()
    .method("do_gibbs", &Forest::do_gibbs)
    .method("do_gibbs_weighted", &Forest::do_gibbs_weighted)
    .method("get_counts", &Forest::get_counts)
    .method("get_tree_counts", &Forest::get_tree_counts)
    .method("set_sigma", &Forest::set_sigma)
    .method("do_predict", &Forest::do_predict)
    .field("num_gibbs", &Forest::num_gibbs);

}
