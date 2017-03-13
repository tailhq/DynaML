/*
Copyright 2015 Mandar Chandorkar

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
package io.github.mandar2812.dynaml.models.sgp

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import breeze.stats.distributions.Gaussian
import spire.implicits._
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.probability
import io.github.mandar2812.dynaml.probability.{BlockedMESNRV, RandomVariable}
import io.github.mandar2812.dynaml.probability.distributions.UESN
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
  * @author mandar2812 date: 28/02/2017.
  *
  * Implementation of Extended Skew-Gaussian Process regression model.
  * This is represented with a finite dimensional [[BlockedMESNRV]]
  * distribution of Adcock and Schutes.
  */
abstract class ESGPModel[T, I: ClassTag](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  data: T, num: Int, lambda: Double, tau: Double,
  meanFunc: DataPipe[I, Double] = DataPipe((_:I) => 0.0))
  extends ContinuousProcessModel[T, I, Double, BlockedMESNRV]
    with SecondOrderProcessModel[T, I, Double, Double, DenseMatrix[Double], BlockedMESNRV]
    with GloballyOptimizable {

  private val logger = Logger.getLogger(this.getClass)

  /**
    * The training data
    **/
  override protected val g: T = data

  /**
    * Mean Function: Takes a member of the index set (input)
    * and returns the corresponding mean of the distribution
    * corresponding to input.
    **/
  override val mean = meanFunc
  /**
    * Underlying covariance function of the
    * Gaussian Processes.
    **/
  override val covariance = cov

  val noiseModel = n


  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters ++ List("skewness", "cutoff")


  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state ++ Map("skewness" -> lambda, "cutoff" -> tau)


  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {

    val (covHyp, noiseHyp) = (
      s.filterKeys(covariance.hyper_parameters.contains),
      s.filterKeys(noiseModel.hyper_parameters.contains)
    )

    covariance.setHyperParameters(covHyp)
    noiseModel.setHyperParameters(noiseHyp)

    current_state = covariance.state ++ noiseModel.state
    current_state += ("skewness" -> s("skewness"), "cutoff" -> s("cutoff"))
    this
  }

  val npoints = num

  protected var blockSize = 1000

  protected lazy val trainingData: Seq[I] = dataAsIndexSeq(g)

  protected lazy val trainingDataLabels = PartitionedVector(
    dataAsSeq(g).toStream.map(_._2),
    trainingData.length.toLong, _blockSize
  )

  def blockSize_(b: Int): Unit = {
    blockSize = b
  }

  def _blockSize: Int = blockSize

  protected var caching: Boolean = false

  protected var partitionedKernelMatrixCache: PartitionedPSDMatrix = _


  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U) = {
    logger.info("Calculating posterior predictive distribution")

    //Calculate the prior distribution parameters
    //Skewness and cutoff
    val (l,t) = (current_state("skewness"), current_state("cutoff"))

    //Mean
    //Test:
    val priorMeanTest = PartitionedVector(
      test.map(mean(_))
        .grouped(_blockSize)
        .zipWithIndex.map(c => (c._2.toLong, DenseVector(c._1.toArray)))
        .toStream,
      test.length.toLong)

    //Training
    val trainingMean = PartitionedVector(
      trainingData.map(mean(_)).toStream,
      trainingData.length.toLong, _blockSize
    )

    //Calculate the skewness as a partitioned vector
    //Test
    val priorSkewnessTest = priorMeanTest.map(b => (b._1, DenseVector.fill[Double](b._2.length)(l)))
    //Training
    val skewnessTraining = trainingMean.map(b => (b._1, DenseVector.fill[Double](b._2.length)(l)))

    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    //Calculate the kernel + noise matrix on the training data
    val smoothingMat = if(!caching) {
      logger.info("---------------------------------------------------------------")
      logger.info("Calculating covariance matrix for training points")
      SVMKernel.buildPartitionedKernelMatrix(trainingData,
        trainingData.length, _blockSize, _blockSize,
        effectiveTrainingKernel.evaluate)
    } else {
      logger.info("** Using cached training matrix **")
      partitionedKernelMatrixCache
    }

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix for test points")
    val kernelTest = SVMKernel.buildPartitionedKernelMatrix(
      test, test.length.toLong,
      _blockSize, _blockSize, covariance.evaluate)

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix between training and test points")
    val crossKernel = SVMKernel.crossPartitonedKernelMatrix(
      trainingData, test,
      _blockSize, _blockSize,
      covariance.evaluate)

    //Solve for parameters of the posterior predictive distribution
    val (predMean, predCov, predSkewness, predCutoff) = ESGPModel.solve(
      trainingDataLabels, trainingMean, priorMeanTest,
      smoothingMat, kernelTest, crossKernel,
      skewnessTraining, priorSkewnessTest,
      t)

    BlockedMESNRV(predCutoff, predSkewness, predMean, predCov)
  }


  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    **/
  override def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int) = {
    val stdG = new Gaussian(0.0, 1.0)

    //Calculate the confidence interval alpha, corresponding to the value of sigma
    val alpha = 1.0 - (stdG.cdf(sigma.toDouble) - stdG.cdf(-1.0*sigma.toDouble))

    logger.info("Calculated confidence bound: alpha = "+alpha)
    val BlockedMESNRV(pTau, pLambda, postmean, postcov) = predictiveDistribution(testData)

    val varD: PartitionedVector = bdiag(postcov)

    val stdDev = varD._data.map(c => (c._1, sqrt(c._2))).map(_._2.toArray)
    val mean = postmean._data.map(_._2.toArray)
    val lambda = pLambda._data.map(_._2.toArray)

    logger.info("Generating (marginal) error bars using a buffered approach")

    val zippedBufferedParams = mean.zip(stdDev).zip(lambda).map(c => (c._1._1, c._1._2, c._2))

    val predictions = zippedBufferedParams.flatMap(buffer => {
      val (muBuff, sBuff, lBuff) = buffer

      muBuff.zip(sBuff).zip(lBuff).map(c => {
        val (mu, s, l) = (c._1._1, c._1._2, c._2)
        val (_, _, appMode, lower, higher) =
          probability.OrderStats(RandomVariable(UESN(pTau, l, mu, s)), alpha)

        (appMode, lower, higher)
      })
    })

    predictions.zip(testData).map(c => (c._2, c._1._1, c._1._2, c._1._3))
  }

  /**
    * Returns a [[DataPipe2]] which calculates the energy of data: [[T]].
    * See: [[energy]] below.
    * */
  def calculateEnergyPipe(h: Map[String, Double], options: Map[String, String]) =
    DataPipe2((training: Seq[I], trainingLabels: PartitionedVector) => {
      setState(h)

      val (l,t) = (current_state("skewness"), current_state("cutoff"))

      val trainingMean = PartitionedVector(
        training.toStream.map(mean(_)),
        training.length.toLong, _blockSize
      )

      val skewnessTraining = trainingMean.map(b => (b._1, DenseVector.fill[Double](b._2.length)(l)))

      val effectiveTrainingKernel: LocalScalarKernel[I] = this.covariance + this.noiseModel

      effectiveTrainingKernel.setBlockSizes((_blockSize, _blockSize))

      val kernelTraining: PartitionedPSDMatrix =
        effectiveTrainingKernel.buildBlockedKernelMatrix(training, training.length)

     ESGPModel.logLikelihood(
       trainingLabels, t, skewnessTraining,
       trainingMean, kernelTraining)
    })

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: Map[String, Double], options: Map[String, String]) =
    calculateEnergyPipe(h, options)(trainingData, trainingDataLabels)

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: I) = predictionWithErrorBars(Seq(point), 2).head._2

  /**
    * Cache the training kernel and noise matrices
    * for fast access in future predictions.
    * */
  override def persist(state: Map[String, Double]): Unit = {
    //Set the hyperparameters to state
    setState(state)

    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    //Calculate the kernel matrix over the training data.
    partitionedKernelMatrixCache =
      SVMKernel.buildPartitionedKernelMatrix(
        trainingData, trainingData.length,
        _blockSize, _blockSize, effectiveTrainingKernel.evaluate)

    //Set the caching flag to true
    caching = true

  }

  /**
    * Forget the cached kernel & noise matrices.
    * */
  def unpersist(): Unit = {
    partitionedKernelMatrixCache = null
    caching = false
  }


}


object ESGPModel {

  /**
    * Calculate the negative log likelihood of data for a
    * multivariate extended skew normal model.
    * */
  def logLikelihood(
    y: PartitionedVector, tau: Double,
    skewness: PartitionedVector, center: PartitionedVector,
    covarince:PartitionedPSDMatrix): Double = {

    try {
      val distribution = BlockedMESNRV(tau, skewness, center, covarince)
      -1.0*distribution.underlyingDist.logPdf(y)
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }
  }

  /**
    * Calculate the parameters of the posterior predictive distribution
    * for a multivariate extended skew normal model.
    * */
  def solve(
    trainingLabels: PartitionedVector,
    trainingMean: PartitionedVector,
    priorMeanTest: PartitionedVector,
    smoothingMat: PartitionedPSDMatrix,
    kernelTest: PartitionedPSDMatrix,
    crossKernel: PartitionedMatrix,
    skewnessTraining: PartitionedVector,
    priorSkewnessTest: PartitionedVector,
    priorCutoff: Double): (PartitionedVector, PartitionedPSDMatrix, PartitionedVector, Double) = {

    val Lmat: LowerTriPartitionedMatrix = bcholesky(smoothingMat)

    val alpha: PartitionedVector = Lmat.t \\ (Lmat \\ (trainingLabels-trainingMean))

    val beta : PartitionedVector = Lmat.t \\ (Lmat \\ skewnessTraining)

    val delta: Double = 1.0/sqrt(1.0 + (skewnessTraining dot beta))

    val v: PartitionedMatrix = Lmat \\ crossKernel

    val varianceReducer: PartitionedMatrix = v.t * v

    //Ensure that the variance reduction is symmetric
    val adjustedVarReducer: PartitionedMatrix = varianceReducer

    /*(varianceReducer.L + varianceReducer.L.t).map(bm =>
      if(bm._1._1 == bm._1._2) (bm._1, bm._2*(DenseMatrix.eye[Double](bm._2.rows)*0.5))
      else bm)*/

    val reducedVariance: PartitionedPSDMatrix =
      new PartitionedPSDMatrix(
        (kernelTest - adjustedVarReducer).filterBlocks(c => c._1 >= c._2),
        kernelTest.rows, kernelTest.cols)

    (
      priorMeanTest + crossKernel.t * alpha,
      reducedVariance,
      (priorSkewnessTest - crossKernel.t * beta)*delta,
      (priorCutoff + (skewnessTraining dot alpha))*delta)

  }

  /**
    * Create an instance of [[ESGPModel]] for a
    * particular data type [[T]]
    *
    * @tparam T The type of the training data
    * @tparam I The type of the input patterns in the data set of type [[T]]
    * @param cov The covariance function
    * @param noise The noise covariance function
    * @param meanFunc The trend or mean function
    * @param trainingdata The actual data set of type [[T]]
    * @param lambda Skewness parameter
    * @param tau Cut off parameter
    * @param transform An implicit conversion from [[T]] to [[Seq]] represented as a [[DataPipe]]
    * */
  def apply[T, I: ClassTag](
    cov: LocalScalarKernel[I], noise: LocalScalarKernel[I],
    meanFunc: DataPipe[I, Double], lambda: Double, tau: Double)(
    trainingdata: T, num: Int = 0)(
    implicit transform: DataPipe[T, Seq[(I, Double)]]) = {

    val num_points = if(num > 0) num else transform(trainingdata).length

    new ESGPModel[T, I](cov, noise, trainingdata, num_points, lambda, tau, meanFunc) {
      /**
        * Convert from the underlying data structure to
        * Seq[(I, Y)] where I is the index set of the GP
        * and Y is the value/label type.
        **/
      override def dataAsSeq(data: T) = transform(data)
    }

  }

}