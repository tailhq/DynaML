package io.github.mandar2812.dynaml.probability

import breeze.numerics.log1p
import spire.algebra.Field
import breeze.stats.distributions.{ContinuousDistr, Density, Rand}
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
DistL <: Density[Domain] with Rand[Domain]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]],
  proposalDist: RandomVarWithDistr[ConditioningSet, Dist],
  burnIn: Long = 1000L)(
  implicit vectorSpace: Field[ConditioningSet])
  extends ProbabilityModel[ConditioningSet, Domain, Dist, DistL](p, c) {


  override val posterior = DataPipe((data: Domain) => {

    val logLikelihoodFunc = (candidate: ConditioningSet) => {
      prior.underlyingDist.logApply(candidate)+
        likelihood(candidate).underlyingDist.logApply(data)
    }

    //Initialize an MCMC sampler
    val sampler = new AbstractMetropolisHastings[ConditioningSet, Dist](
      logLikelihoodFunc, proposalDist.underlyingDist,
      prior.sample(), burnIn)

    RandomVariable(() => sampler.draw())
  })
}

object MCMCProbModel {
  def apply[
  ConditioningSet, Domain,
  Dist <: Density[ConditioningSet] with Rand[ConditioningSet],
  DistL <: Density[Domain] with Rand[Domain]
  ](p: RandomVarWithDistr[ConditioningSet, Dist],
    c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]],
    proposalDist: RandomVarWithDistr[ConditioningSet, Dist])(
    implicit vectorSpace: Field[ConditioningSet]) = new MCMCProbModel(p, c, proposalDist)
}

class ContinuousMCMCModel[
ConditioningSet, Domain](
  p: ContinuousDistrRV[ConditioningSet],
  c: DataPipe[ConditioningSet, ContinuousDistrRV[Domain]],
  proposalDist: RandomVarWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]],
  burnIn: Long = 1000L)(
  implicit vectorSpace: Field[ConditioningSet])
  extends MCMCProbModel[ConditioningSet, Domain,
    ContinuousDistr[ConditioningSet], ContinuousDistr[Domain]](p, c, proposalDist, burnIn) {


  override val posterior = DataPipe((data: Domain) => {

    val logLikelihoodFunc = (candidate: ConditioningSet) => {
      prior.underlyingDist.logPdf(candidate) +
        likelihood(candidate).underlyingDist.logPdf(data)
    }

    //Initialize an MCMC sampler
    val sampler = AbstractMetropolisHastings(
      logLikelihoodFunc, proposalDist.underlyingDist,
      prior.sample(), burnIn)

    RandomVariable(() => sampler.draw())
  })
}

