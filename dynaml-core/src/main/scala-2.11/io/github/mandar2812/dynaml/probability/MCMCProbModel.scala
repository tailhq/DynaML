package io.github.mandar2812.dynaml.probability

import spire.algebra.Field
import breeze.stats.distributions.{Density, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.mcmc.AbstractMetropolisHastings

/**
  * @author mandar2812 date 05/01/2017.
  *
  * A Bayesian inference model using Metropolis-Hastings
  * to sample from the posterior distribution.
  */
class MCMCProbModel[
ConditioningSet, Domain,
Dist <: Density[ConditioningSet] with Rand[ConditioningSet],
DistL <: Density[Domain]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]],
  proposalDist: RandomVarWithDistr[ConditioningSet, Dist])(
  implicit vectorSpace: Field[ConditioningSet])
  extends ProbabilityModel[ConditioningSet, Domain, Dist, DistL](p, c) {


  override val posterior = DataPipe((data: Domain) => {

    val logLikelihoodFunc = (candidate: ConditioningSet) => {
      prior.underlyingDist.logApply(candidate)+
        likelihood(candidate).underlyingDist.logApply(data)
    }

    //Initialize an MCMC sampler
    val sampler = new AbstractMetropolisHastings[ConditioningSet](
      logLikelihoodFunc, proposalDist.underlyingDist,
      prior.sample(), burnIn = 1000L)

    RandomVariable(() => sampler.draw())
  })
}

object MCMCProbModel {
  def apply[
  ConditioningSet, Domain,
  Dist <: Density[ConditioningSet] with Rand[ConditioningSet],
  DistL <: Density[Domain]
  ](p: RandomVarWithDistr[ConditioningSet, Dist],
    c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]],
    proposalDist: RandomVarWithDistr[ConditioningSet, Dist])(
    implicit vectorSpace: Field[ConditioningSet]) = new MCMCProbModel(p, c, proposalDist)
}
