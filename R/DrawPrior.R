draw_prior <- function(X,Y,alpha,omega,num_tree,num_clust) {

  opts <- Opts()
  hypers <- Hypers(X,Y,num_tree=num_tree)

  rdirichlet <-  function (n, alpha) 
  {
    l <- length(alpha)
    x <- matrix(rgamma(l * n, alpha), ncol = l, byrow = TRUE)
    sm <- x %*% rep(1, l)
    return(x/as.vector(sm))
  }

  pi <- rdirichlet(1, omega / num_clust * rep(1,num_clust))
  s_0 <- 1/ncol(X) * rep(1,ncol(X))
  s <- rdirichlet(num_clust, alpha * s_0)
  z <- sample(1:num_clust, num_tree, replace = TRUE, prob = pi)

  DrawFromPrior(X,Y,X,
                hypers$group,
                hypers$alpha,
                hypers$omega,
                hypers$beta,
                hypers$gamma,
                hypers$sigma,
                hypers$shape,
                hypers$width,
                hypers$num_tree,
                hypers$sigma_hat,
                hypers$k,
                hypers$alpha_scale,
                hypers$alpha_shape_1,
                hypers$alpha_shape_2,
                hypers$tau_rate,
                hypers$num_tree_prob,
                hypers$alpha_rate,
                hypers$temperature,
                hypers$s_0,
                hypers$num_clust,
                opts$num_burn,
                opts$num_thin,
                opts$num_save,
                opts$num_print,
                opts$update_sigma_mu,
                opts$update_s,
                opts$update_alpha,
                opts$update_beta,
                opts$update_gamma,
                opts$update_tau,
                opts$update_tau_mean,
                opts$update_num_tree,
                opts$split_merge,
                opts$mh_bd, opts$mh_prior,
                opts$do_interaction,
                pi, z, s
                )
}
