package io.github.mandar2812.dynaml.probability

import spire.algebra.Field
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.AbstractContinuousDistr
import io.github.mandar2812.dynaml.probability.mcmc.GeneralMetropolisHastings

/**
  * @author mandar2812 date 06/01/2017
  *
  * Monte Carlo based bayesian inference model where the parameter
  * space is known to be continuous and hence represented via
  * a [[ContinuousDistrRV]] instance.
  *
  * @tparam ConditioningSet The type representing the model parameters
  * @tparam Domain The type representing the observed data.
  *
  * @param p The prior distribution on model parameters
  * @param c The likelihood of the data given a particular value of parameters
  * */
class ContinuousMCMC[ConditioningSet, Domain](
  p: ContinuousDistrRV[ConditioningSet],
  c: DataPipe[ConditioningSet, ContinuousDistrRV[Domain]],
  proposalDist: RandomVarWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]],
  burnIn: Long = 1000L)(implicit vectorSpace: Field[ConditioningSet])
  extends RejectionSamplingScheme[ConditioningSet, Domain,
    ContinuousDistr[ConditioningSet],
    ContinuousDistr[Domain],
    AbstractContinuousDistr[(ConditioningSet, Domain)]](p, c) {


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

