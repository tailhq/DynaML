package io.github.mandar2812.dynaml.probability

import spire.algebra.Field
import breeze.stats.distributions.{ContinuousDistr, Density, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.{AbstractContinuousDistr, GenericDistribution}
import io.github.mandar2812.dynaml.probability.mcmc.GeneralMetropolisHastings

/**
  * @author mandar2812 date 05/01/2017.
  *
  * A Bayesian inference model using Metropolis-Hastings
  * to sample from the posterior distribution.
  */
abstract class MCMCProbModel[
ConditioningSet, Domain,
Dist <: Density[ConditioningSet] with Rand[ConditioningSet],
DistL <: Density[Domain] with Rand[Domain],
JointDist <: GenericDistribution[(ConditioningSet, Domain)]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]],
  proposalDist: RandomVarWithDistr[ConditioningSet, Dist],
  burnIn: Long = 1000L)(
  implicit vectorSpace: Field[ConditioningSet])
  extends ProbabilityModel[ConditioningSet, Domain, Dist, DistL, JointDist](p, c) {

}

class ContinuousMCMCModel[ConditioningSet, Domain](
  p: ContinuousDistrRV[ConditioningSet],
  c: DataPipe[ConditioningSet, ContinuousDistrRV[Domain]],
  proposalDist: RandomVarWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]],
  burnIn: Long = 1000L)(implicit vectorSpace: Field[ConditioningSet])
  extends MCMCProbModel[
    ConditioningSet, Domain,
    ContinuousDistr[ConditioningSet],
    ContinuousDistr[Domain],
    AbstractContinuousDistr[(ConditioningSet, Domain)]
    ](p, c, proposalDist, burnIn) {


  override val underlyingDist = new AbstractContinuousDistr[(ConditioningSet, Domain)] {

    private val priorSample = prior.sample()

    override def unnormalizedLogPdf(x: (ConditioningSet, Domain)) =
      prior.underlyingDist.unnormalizedLogPdf(x._1) +
        likelihood(x._1).underlyingDist.unnormalizedLogPdf(x._2)

    override def logNormalizer = prior.underlyingDist.logNormalizer +
      likelihood(priorSample).underlyingDist.logNormalizer

    override def draw() = {
      val ps = prior.underlyingDist.draw()
      (ps, likelihood(ps).underlyingDist.draw())
    }
  }

  override val posterior: DataPipe[Domain, RandomVariable[ConditioningSet]] = DataPipe((data: Domain) => {

    val logLikelihoodFunc = (candidate: ConditioningSet) => {
      prior.underlyingDist.logPdf(candidate) + likelihood(candidate).underlyingDist.logPdf(data)
    }

    //Initialize an MCMC sampler
    val sampler = GeneralMetropolisHastings(
      LikelihoodModel(logLikelihoodFunc), proposalDist.underlyingDist,
      prior.sample(), burnIn)

    RandomVariable(() => sampler.draw())
  })
}

