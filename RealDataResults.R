load_results <- function() {
  load("./cache/results_ais.RData")
  load("./cache/results_bbb.RData")
  load("./cache/results_diamonds.RData")
  load("./cache/results_hatco.RData")
  load("./cache/results_tec.RData")
  load("./cache/results_tri.RData")
  load("./cache/results_wipp.RData")
}

make_df <- function() {

  g <- function(res) {
    rrmse_results(res, "softbart")$normed_results
  }

  method_list <- c("RF", "Lasso", "XGB", "BART")
  idx_list <- c(1, 2, 3, 6)

  dfit <- function(res, dataset) {
    out <- data.frame(RRMSE = res[,1], method = method_list[1], dataset = dataset)
    for(i in 2:length(method_list)) {
      out <- rbind(out, data.frame(RRMSE = res[,i],
                                   method = method_list[i],
                                   dataset = dataset))
    }
    return(out)
  }

  out <- dfit(g(results_ais), "AIS")
  out <- rbind(out, dfit(g(results_bbb), "BBB"))
  out <- rbind(out, dfit(g(results_diamonds), "Diamonds"))
  out <- rbind(out, dfit(g(results_hatco), "Hatco"))
  out <- rbind(out, dfit(g(results_tec), "Tecator"))
  out <- rbind(out, dfit(g(results_tri), "Triazines"))
  out <- rbind(out, dfit(g(results_wipp), "WIPP"))

  return(as_tibble(out))

}

load_results()

testing <- make_df()
ggplot(testing, aes(y = RRMSE, x = method, fill = method)) +
  geom_boxplot() +
  facet_wrap(~dataset) + scale_y_log10() + xlab("")
