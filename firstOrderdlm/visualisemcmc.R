## install and load packages
packages = c("dplyr","ggplot2","gridExtra", "ggmcmc", "coda")
newPackages = packages[!(packages %in% as.character(installed.packages()[,"Package"]))]
if(length(newPackages)) install.packages(newPackages)
lapply(packages, require, character.only = T)

## thinning reduces the autocorrelation between successive iterations
plotIters = function(iters, variable, burnin, thin) {
  mcmcObject = mcmc(iters[seq(from = burnin, to = nrow(iters), by = thin), variable]) %>% ggs()
  
  p1 = ggs_histogram(mcmcObject)
  p2 = ggs_traceplot(mcmcObject)
  p3 = ggs_autocorrelation(mcmcObject)
  p4 = ggs_running(mcmcObject)
  
  grid.arrange(p1, p2, p3, p4)
}

##################
# Visualise MCMC #
##################

iters = read.csv("mcmcOut.csv")
colnames(iters) <- c("V", "W", "m0", "c0")

plotIters(iters, variable = 1:2, burnin = nrow(iters)*0.1, thin = 1)
