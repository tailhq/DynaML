/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 * */
package io.github.mandar2812.dynaml.probability.mcmc

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.optimization.{
  GloballyOptimizable,
  GloballyOptWithGrad
}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.utils.{ConfigEncoding, getPriorMapDistr, ProgressBar}
import org.apache.log4j.Logger
import spire.implicits.cfor

import scala.util.Random

/**
  * Implementation of the classic Markov Chain Monte Carlo algorithm,
  * equipped with reversible Metropolis Hastings updates.
  *
  * @tparam Model Any type which implements the [[GloballyOptimizable]] trait
  *               i.e. the [[GloballyOptimizable.energy()]] method.
  * @tparam Distr A breeze distribution type for hyper-parameter priors.
  * @param system The model whose hyper-parameters are to be sampled.
  * @param hyper_prior A mean field prior measure over the [[system]] hyper-parameters.
  *                    Represented as a [[Map]].
  * @param proposal A distribution which is used to propose hyper-parameter updates.
  * @author mandar2812 date 24/05/2017.
  * */
class HyperParameterMCMC[
  Model <: GloballyOptimizable,
  Distr <: ContinuousDistr[Double]
](system: Model,
  hyper_prior: Map[String, Distr],
  val proposal: ContinuousRVWithDistr[DenseVector[Double], ContinuousDistr[
    DenseVector[Double]
  ]],
  burnIn: Long)
    extends AdaptiveHyperParameterMCMC[Model, Distr](
      system,
      hyper_prior,
      burnIn,
      "Vanilla MCMC"
    ) {

  override def candidateDistribution(
    mean: DenseVector[Double],
    covariance: DenseMatrix[Double]
  ) = proposal

  override protected def getExplorationVar: DenseMatrix[Double] = {
    sigma
  }

  override protected def updateMoments(x: DenseVector[Double]): Unit = {}
}

object HyperParameterMCMC {

  def apply[Model <: GloballyOptimizable, Distr <: ContinuousDistr[Double]](
    system: Model,
    hyper_prior: Map[String, Distr],
    proposal: ContinuousRVWithDistr[DenseVector[Double], ContinuousDistr[
      DenseVector[Double]
    ]],
    burnIn: Long
  ): HyperParameterMCMC[Model, Distr] =
    new HyperParameterMCMC(system, hyper_prior, proposal, burnIn)

}

/**
  * Hamiltonian Markov Chain Monte Carlo algorithm,
  * equipped with reversible Metropolis Hastings updates.
  *
  * @tparam Model Any type which implements the [[GloballyOptimizable]] trait
  *               i.e. the [[GloballyOptimizable.energy()]] method.
  * @tparam Distr A breeze distribution type for hyper-parameter priors.
  * @param system The model whose hyper-parameters are to be sampled.
  * @param hyper_prior A mean field prior measure over the [[system]] hyper-parameters.
  *                    Represented as a [[Map]].
  * @author mandar2812 date 31/07/2017.
  * */
abstract class HyperParameterHMC[
  Model <: GloballyOptWithGrad,
  Distr <: ContinuousDistr[Double]
](system: Model,
  hyper_prior: Map[String, Distr],
  proposal: ContinuousRVWithDistr[DenseVector[Double], ContinuousDistr[
    DenseVector[Double]
  ]],
  val burnIn: Long)
    extends RandomVariable[Map[String, Double]] {}

/**
  * <h3>Adaptive Metropolis Sampler</h3>
  *
  * Adaptive Markov Chain Monte Carlo algorithm,
  * equipped with reversible Metropolis Hastings updates.
  *
  * The proposal distribution is a multivariate gaussian whose
  * variance is adjusted every iteration.
  *
  * @tparam Model Any type which implements the [[GloballyOptimizable]] trait
  *               i.e. the [[GloballyOptimizable.energy()]] method.
  * @tparam Distr A breeze distribution type for hyper-parameter priors.
  * @param system The model whose hyper-parameters are to be sampled.
  * @param hyper_prior A mean field prior measure over the [[system]] hyper-parameters.
  *                    Represented as a [[Map]].
  * @author mandar2812 date 19/07/2017.
  * */
class AdaptiveHyperParameterMCMC[
  Model <: GloballyOptimizable,
  Distr <: ContinuousDistr[Double]
](val system: Model,
  val hyper_prior: Map[String, Distr],
  val burnIn: Long,
  algoName: String = "Adaptive MCMC")
    extends RandomVariable[Map[String, Double]] {

  self =>

  val logger: Logger = Logger.getLogger(this.getClass)

  val encoder: ConfigEncoding = ConfigEncoding(hyper_prior.keys.toList)

  val initialState = encoder(system._current_state)

  protected val dimensions: Int = hyper_prior.size

  implicit protected val vector_field: VectorField = VectorField(dimensions)

  protected val processed_prior =
    EncodedContDistrRV(getPriorMapDistr(hyper_prior), encoder)

  protected val logLikelihood: (DenseVector[Double]) => Double =
    (candidate: DenseVector[Double]) => {
      processed_prior.underlyingDist.logPdf(candidate) - system.energy(
        encoder.i(candidate)
      )
    }

  def candidateDistribution(
    mean: DenseVector[Double],
    covariance: DenseMatrix[Double]
  ): ContinuousRVWithDistr[DenseVector[Double], ContinuousDistr[
    DenseVector[Double]
  ]] =
    MultGaussianRV(mean, covariance)

  protected val beta = 0.05

  protected var count: Long = 1L

  protected var sigma: DenseMatrix[Double] = initialState * initialState.t

  protected var acceptedSamples: Long = 0L

  protected var mean: DenseVector[Double] = initialState

  protected var previous_sample: DenseVector[Double] = initialState

  protected var previous_log_likelihood = logLikelihood(initialState)

  protected val eye: DenseMatrix[Double] = DenseMatrix.eye[Double](dimensions)

  def _previous_sample = encoder.i(previous_sample)

  def _previous_log_likelihood = previous_log_likelihood

  def sampleAcceptenceRate: Double = acceptedSamples.toDouble / count

  def _count = count

  protected def initMessage(): Unit = {
    println("--------------------------------------")
    println("\t" + algoName)
    println("--------------------------------------")

    println("Initial Sample: \n")
    pprint.pprintln(_previous_sample)
    print("Energy = ")
    pprint.pprintln(previous_log_likelihood)
    println("--------------------------------------")
    //println("\n\n************************************")
    println("\nBurn in period for " + burnIn + " samples\n")
    //println("\n************************************")
  }

  initMessage()
  val pb_burn_in = new ProgressBar(burnIn.toInt)
  cfor(0)(i => i < burnIn, i => i + 1)(i => {
    next()
    pb_burn_in += 1
  })

  //println("\n\n************************************")
  println("\nBurn in period complete\n")
  //println("\n************************************")

  protected def getExplorationVar: DenseMatrix[Double] = {
    val adj = count.toDouble / (count - 1)
    if (count <= 2 * dimensions) eye * 0.01 / dimensions.toDouble
    else
      (sigma * math.pow((1 - beta) * 2.38, 2) * adj / dimensions.toDouble) +
        (eye * math.pow(0.1 * beta, 2) / dimensions.toDouble)
  }

  protected def next(): DenseVector[Double] = {

    val explorationCov      = getExplorationVar
    var isAccepted: Boolean = false

    var acceptedSample = previous_sample

    while (!isAccepted) {
      count += 1
      val candidate =
        candidateDistribution(encoder(_previous_sample), explorationCov).draw

      val likelihood = logLikelihood(candidate)

      //println("Candidate: \n")
      //pprint.pprintln(encoder.i(candidate))
      //print("Log-Likelihood = ")
      //pprint.pprintln(likelihood)

      val acceptanceRatio = likelihood - previous_log_likelihood

      val x = if (acceptanceRatio > 0.0) {
        previous_sample = candidate
        previous_log_likelihood = likelihood
        //println("Status: Accepted\n")
        acceptedSamples += 1
        acceptedSample = candidate
        isAccepted = true
        candidate
      } else if (Random.nextDouble() < math.exp(acceptanceRatio)) {
        previous_sample = candidate
        previous_log_likelihood = likelihood
        //println("Status: Accepted\n")
        acceptedSamples += 1
        acceptedSample = candidate
        isAccepted = true
        candidate
      } else {
        //println("Status: Rejected\n")
        previous_sample
      }

      updateMoments(x)
    }

    acceptedSample
  }

  protected def updateMoments(x: DenseVector[Double]): Unit = {

    val m    = mean
    val s    = sigma
    val mnew = m + (x - m) / (count + 1).toDouble
    val snew = s + (m * m.t) - (mnew * mnew.t) + ((x * x.t) - s - (m * m.t)) / (count + 1).toDouble

    mean = mnew
    sigma = snew
  }

  /**
    * Generate a sample from
    * the random variable
    *
    **/
  override val sample
    : DataPipe[Unit, Map[String, Double]] = DataPipe(next _) > encoder.i
    
  override def iid(n: Int): IIDRandomVariable[Map[String,Double],RandomVariable[Map[String,Double]]] = 
    new IIDRandomVariable[Map[String,Double], RandomVariable[Map[String,Double]]]{
      override val baseRandomVariable: RandomVariable[Map[String,Double]] = self 

      override val num: Int = n

      override val sample: DataPipe[Unit,Stream[Map[String,Double]]] = DataPipe(() => {
        val pb = new ProgressBar(num)

        (1 to num).map(_ => {
          pb += 1
          encoder.i(self.next())
        }).toStream
      })
    }
}

/**
  * Implementation of the <i>Single Component Adaptive Metropolis</i> (SCAM) Algorithm
  *
  * @tparam Model Any type which implements the [[GloballyOptimizable]] trait
  *               i.e. the [[GloballyOptimizable.energy()]] method.
  * @tparam Distr A breeze distribution type for hyper-parameter priors.
  * @param system The model whose hyper-parameters are to be sampled.
  * @param hyper_prior A mean field prior measure over the [[system]] hyper-parameters.
  *                    Represented as a [[Map]].
  * @author mandar2812 date 20/05/2017.
  * */
class HyperParameterSCAM[
  Model <: GloballyOptimizable,
  Distr <: ContinuousDistr[Double]
](override val system: Model,
  override val hyper_prior: Map[String, Distr],
  override val burnIn: Long)
    extends AdaptiveHyperParameterMCMC[Model, Distr](
      system,
      hyper_prior,
      burnIn,
      "Hyper-Parameter SCAM"
    ) {

  override protected def getExplorationVar: DenseMatrix[Double] = {
    val adj = count.toDouble / (count - 1)
    if (count <= 10) eye * 25d
    else
      sigma.mapPairs((k, v) => {
        val (i, j) = k
        if (i == j) (v * adj + 0.05) * 2.4 * 2.4 else 0d
      })
  }
}
