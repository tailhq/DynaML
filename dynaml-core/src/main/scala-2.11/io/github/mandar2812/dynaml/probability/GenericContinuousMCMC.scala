package io.github.mandar2812.dynaml.probability

import breeze.linalg.DenseVector
import spire.algebra.Field
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.probability.distributions.AbstractContinuousDistr
import io.github.mandar2812.dynaml.probability.mcmc.GeneralMetropolisHastings

/**
  * Monte Carlo based bayesian inference model where the parameter
  * space is known to be continuous and hence represented via
  * a [[ContinuousDistrRV]] instance.
  *
  * @tparam ConditioningSet The type representing the model parameters
  * @tparam Domain The type representing the observed data.
  *
  * @param p The prior distribution on model parameters
  * @param c The likelihood of the data given a particular value of parameters
  *
  * @author mandar2812 date 06/01/2017
  *
  * */
class GenericContinuousMCMC[ConditioningSet, Domain](
  p: ContinuousDistrRV[ConditioningSet],
  c: DataPipe[ConditioningSet, ContinuousDistrRV[Domain]],
  proposalDist: RandomVarWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]],
  burnIn: Long = 1000L, dropCount: Int = 100)(implicit vectorSpace: Field[ConditioningSet])
  extends RejectionSamplingScheme[ConditioningSet, Domain,
    ContinuousDistr[ConditioningSet],
    ContinuousDistr[Domain],
    AbstractContinuousDistr[(ConditioningSet, Domain)]](p, c) {


  override val underlyingDist = new AbstractContinuousDistr[(ConditioningSet, Domain)] {

    private val priorSample = prior.sample.run()

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
      prior.sample.run(), burnIn, dropCount)

    RandomVariable(sampler.draw _)
  })
}

class EnergyMCMC[
Model <: GloballyOptimizable,
Distr <: ContinuousDistr[Double]](
  system: Model, hyper_prior: Map[String, Distr],
  proposal: RandomVarWithDistr[DenseVector[Double], ContinuousDistr[DenseVector[Double]]])
  extends RandomVariable[Map[String, Double]] {

  val encoder: ConfigEncoding = ConfigEncoding(hyper_prior.keys.toList)

  implicit val vector_field = VectorField(hyper_prior.size)

  val processed_prior = EncodedContDistrRV(getPriorMapDistr(hyper_prior), encoder)

  private val logLikelihoodFunction = (candidate: DenseVector[Double]) => {
    processed_prior.underlyingDist.logPdf(candidate) - system.energy(encoder.i(candidate))
  }

  var burnIn = 0
  var dropCount = 0

  val metropolisHastings = GeneralMetropolisHastings(
    LikelihoodModel(logLikelihoodFunction), proposal.underlyingDist,
    processed_prior.sample.run(), burnIn, dropCount)

  val base_draw = DataPipe(() => metropolisHastings.draw())

  override val sample = base_draw > encoder.i
}