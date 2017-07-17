package io.github.mandar2812.dynaml.probability.mcmc

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.optimization.{GlobalOptimizer, GloballyOptimizable}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.utils.{ConfigEncoding, getPriorMapDistr}
import org.apache.log4j.Logger
import spire.implicits.cfor

import scala.util.Random

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

class AdaptiveHyperParameterMCMC[
Model <: GloballyOptimizable, Distr <: ContinuousDistr[Double]](
  val system: Model, val hyper_prior: Map[String, Distr],
  val burnIn: Long) extends RandomVariable[Map[String, Double]] {

  val logger = Logger.getLogger(this.getClass)

  val encoder: ConfigEncoding = ConfigEncoding(hyper_prior.keys.toList)

  val initialState = encoder(system._current_state)

  private val dimensions = hyper_prior.size

  implicit private val vector_field = VectorField(dimensions)

  private val processed_prior = EncodedContDistrRV(getPriorMapDistr(hyper_prior), encoder)

  private val logLikelihood = (candidate: DenseVector[Double]) => {
    processed_prior.underlyingDist.logPdf(candidate) - system.energy(encoder.i(candidate))
  }

  val candidateDistributionPipe = DataPipe2(
    (mean: DenseVector[Double], covariance: DenseMatrix[Double]) => MultGaussianRV(mean, covariance)
  )

  protected val beta = 0.05

  private var count: Long = 1L

  private var sigma: DenseMatrix[Double] = initialState * initialState.t

  private var acceptedSamples: Long = 0L

  private var mean: DenseVector[Double] = initialState

  private var previous_sample: DenseVector[Double] = initialState

  private var previous_log_likelihood = logLikelihood(initialState)

  private val eye = DenseMatrix.eye[Double](dimensions)

  cfor(0)(i => i< burnIn, i => i+1)(i => {
    getNext()
  })

  def _previous_sample = encoder.i(previous_sample)

  def _previous_log_likelihood = previous_log_likelihood

  def sampleAcceptenceRate: Double = acceptedSamples.toDouble/count

  def _count = count

  logger.info("--------------------------------------")
  logger.info("\tAdaptive MCMC")
  logger.info("Initial Sample: \n")
  pprint.pprintln(_previous_sample)
  logger.info("Energy = "+previous_log_likelihood)
  logger.info("--------------------------------------")

  protected def getNext(): DenseVector[Double] = {

    val adj = count.toDouble/(count-1)
    val explorationCov =
      if(count <= 2*dimensions) eye*0.01/dimensions.toDouble
      else (sigma*math.pow((1-beta)*2.38, 2)*adj/dimensions.toDouble) + (eye*math.pow(0.1*beta, 2)/dimensions.toDouble)

    count += 1


    val candidate = candidateDistributionPipe(encoder(_previous_sample), explorationCov).draw

    val likelihood = logLikelihood(candidate)

    logger.info("Candidate: \n")
    pprint.pprintln(encoder.i(candidate))
    logger.info("Log-Likelihood = "+likelihood)

    val acceptanceRatio = likelihood - previous_log_likelihood

    val x = if(acceptanceRatio > 0.0) {
      previous_sample = candidate
      previous_log_likelihood = likelihood
      logger.info("Status: Accepted\n")
      acceptedSamples += 1
      candidate
    } else if(Random.nextDouble() < math.exp(acceptanceRatio)) {
      previous_sample = candidate
      previous_log_likelihood = likelihood
      logger.info("Status: Accepted\n")
      acceptedSamples += 1
      candidate
    } else {
      logger.info("Status: Rejected\n")
      previous_sample
    }

    val m = mean
    val s = sigma
    val mnew = m + (x - m)/(count+1).toDouble
    val snew = s + (m*m.t) - (mnew*mnew.t) + ((x*x.t) - s - (m*m.t))/(count+1).toDouble

    mean = mnew
    sigma = snew

    x
  }

  /**
    * Generate a sample from
    * the random variable
    *
    **/
  override val sample = DataPipe(getNext _) > encoder.i
}