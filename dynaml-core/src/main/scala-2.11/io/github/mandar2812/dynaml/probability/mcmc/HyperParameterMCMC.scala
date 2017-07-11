package io.github.mandar2812.dynaml.probability.mcmc

import breeze.linalg.DenseVector
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.utils.{ConfigEncoding, getPriorMapDistr}

/**
  * @author mandar2812 date 24/05/2017.
  * */
class HyperParameterMCMC[
Model <: GloballyOptimizable,
Distr <: ContinuousDistr[Double]](
  system: Model, hyper_prior: Map[String, Distr],
  proposal: ContinuousRVWithDistr[DenseVector[Double], ContinuousDistr[DenseVector[Double]]],
  val burnIn: Long) extends
  RandomVariable[Map[String, Double]] {

  val encoder: ConfigEncoding = ConfigEncoding(hyper_prior.keys.toList)

  val initialState = encoder(system._current_state)

  implicit private val vector_field = VectorField(hyper_prior.size)

  private val processed_prior = EncodedContDistrRV(getPriorMapDistr(hyper_prior), encoder)

  private val logLikelihoodFunction = (candidate: DenseVector[Double]) => {
    processed_prior.underlyingDist.logPdf(candidate) - system.energy(encoder.i(candidate))
  }

  val dropCount = 0

  val metropolisHastings = GeneralMetropolisHastings(
    LikelihoodModel(logLikelihoodFunction), proposal.underlyingDist,
    initialState, burnIn, dropCount)

  val base_draw = DataPipe(() => metropolisHastings.draw())

  override val sample = base_draw > encoder.i
}

object HyperParameterMCMC {

  def apply[
  Model <: GloballyOptimizable,
  Distr <: ContinuousDistr[Double]](
    system: Model, hyper_prior: Map[String, Distr],
    proposal: ContinuousRVWithDistr[DenseVector[Double], ContinuousDistr[DenseVector[Double]]],
    burnIn: Long): HyperParameterMCMC[Model, Distr] =
    new HyperParameterMCMC(system, hyper_prior, proposal, burnIn)

}