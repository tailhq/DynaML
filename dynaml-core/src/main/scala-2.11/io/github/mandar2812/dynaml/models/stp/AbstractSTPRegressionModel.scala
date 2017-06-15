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
package io.github.mandar2812.dynaml.models.stp

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.MultStudentsTPRV
import io.github.mandar2812.dynaml.probability.distributions.{BlockedMultivariateStudentsT, MultivariateStudentsT}
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
  * @author mandar2812 date 26/08/16.
  * Implementation of a Students' T Regression model.
  */
abstract class AbstractSTPRegressionModel[T, I](
  mu: Double, cov: LocalScalarKernel[I],
  n: LocalScalarKernel[I],
  data: T, num: Int,
  meanFunc: DataPipe[I, Double] = DataPipe((_: I) => 0.0))(implicit ev: ClassTag[I])
  extends ContinuousProcessModel[T, I, Double, MultStudentsTPRV]
  with SecondOrderProcessModel[T, I, Double, Double, DenseMatrix[Double], MultStudentsTPRV]
  with GloballyOptimizable {


  private val logger = Logger.getLogger(this.getClass)

  /**
    * The GP is taken to be zero mean, or centered.
    * This is ensured by standardization of the data
    * before being used for further processing.
    *
    * */
  override val mean: DataPipe[I, Double] = meanFunc

  override val covariance = cov

  val noiseModel = n

  override protected val g: T = data

  val npoints = num

  protected var blockSize = 1000

  def blockSize_(b: Int): Unit = {
    blockSize = b
    covariance.setBlockSizes((b,b))
    noiseModel.setBlockSizes((b,b))
  }

  def _blockSize: Int = blockSize

  protected var (caching, kernelMatrixCache)
  : (Boolean, DenseMatrix[Double]) = (false, null)

  protected var partitionedKernelMatrixCache: PartitionedPSDMatrix = _

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
    current_state += ("degrees_of_freedom" -> (s("degrees_of_freedom")+2.0))
    this
  }

  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters ++ List("degrees_of_freedom")


  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state ++ Map("degrees_of_freedom" -> (2.0+mu))

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: I): Double = predictionWithErrorBars(Seq(point), 1).head._2

  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U): MultStudentsTPRV = {
    logger.info("Calculating posterior predictive distribution")
    //Calculate the kernel matrix on the training data
    val training = dataAsIndexSeq(g)
    val trainingLabels = PartitionedVector(
      dataAsSeq(g).toStream.map(_._2),
      training.length.toLong, _blockSize
    )

    val priorMeanTest = PartitionedVector(
      test.map(mean(_))
        .grouped(_blockSize)
        .zipWithIndex.map(c => (c._2.toLong, DenseVector(c._1.toArray)))
        .toStream,
      test.length.toLong)

    val trainingMean = PartitionedVector(
      dataAsSeq(g).toStream.map(_._1).map(mean(_)),
      training.length.toLong, _blockSize
    )

    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    val smoothingMat = if(!caching) {
      logger.info("---------------------------------------------------------------")
      logger.info("Calculating covariance matrix for training points")
      SVMKernel.buildPartitionedKernelMatrix(training,
        training.length, _blockSize, _blockSize,
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
      training, test,
      _blockSize, _blockSize,
      covariance.evaluate)

    //Calculate the predictive mean and co-variance
    val Lmat: LowerTriPartitionedMatrix = bcholesky(smoothingMat)

    val alpha: PartitionedVector = Lmat.t \\ (Lmat \\ (trainingLabels-trainingMean))

    val v: PartitionedMatrix = Lmat \\ crossKernel

    val varianceReducer: PartitionedMatrix = v.t * v

    //Ensure that the variance reduction is symmetric
    val adjustedVarReducer: PartitionedMatrix = (varianceReducer.L + varianceReducer.L.t).map(bm =>
      if(bm._1._1 == bm._1._2) (bm._1, bm._2*(DenseMatrix.eye[Double](bm._2.rows)*0.5))
      else bm)


    val degOfFreedom = current_state("degrees_of_freedom")

    val beta = (trainingLabels-trainingMean) dot alpha

    val varianceAdjustment = (degOfFreedom + beta - 2.0)/(degOfFreedom + training.length - 2.0)

    val reducedVariance: PartitionedPSDMatrix =
      new PartitionedPSDMatrix(
        (kernelTest - adjustedVarReducer)
          .filterBlocks(c => c._1 <= c._2)
          .map(c => (c._1, c._2*varianceAdjustment)),
        kernelTest.rows, kernelTest.cols)


    MultStudentsTPRV(test.length.toLong, _blockSize)(
      training.length+degOfFreedom,
      priorMeanTest + crossKernel.t * alpha,
      reducedVariance)
  }

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    **/
  def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int):
  Seq[(I, Double, Double, Double)] = {

    val posterior = predictiveDistribution(testData)

    val mean = posterior.mean.toStream

    val (lower, upper) = posterior.underlyingDist.confidenceInterval(sigma.toDouble)

    val lowerErrorBars = lower.toStream
    val upperErrorBars = upper.toStream


    logger.info("Generating error bars")

    val preds = mean.zip(lowerErrorBars.zip(upperErrorBars)).map(t => (t._1, t._2._1, t._2._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    *
    * In this particular case E(h) = -log p(Y|X,h)
    * also known as log likelihood.
    **/
  override def energy(h: Map[String, Double], options: Map[String, String]): Double = {

    setState(h)
    val training = dataAsIndexSeq(g)
    val trainingLabels = PartitionedVector(
      dataAsSeq(g).toStream.map(_._2),
      training.length.toLong, _blockSize
    )

    val trainingMean = PartitionedVector(
      dataAsSeq(g).toStream.map(_._1).map(mean(_)),
      training.length.toLong, _blockSize
    )

    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    val kernelTraining: PartitionedPSDMatrix =
      effectiveTrainingKernel.buildBlockedKernelMatrix(training, npoints)

    if(options.contains("persist") && (options("persist") == "true" || options("persist") == "1")) {
      partitionedKernelMatrixCache = kernelTraining
      caching = true
    }

    AbstractSTPRegressionModel.logLikelihood(
      current_state("degrees_of_freedom"),
      trainingLabels-trainingMean, kernelTraining)
  }

  /**
    * Cache the training kernel and noise matrices
    * for fast access in future predictions.
    * */
  def persist(): Unit = {

    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    val training = dataAsIndexSeq(g)
    partitionedKernelMatrixCache = SVMKernel.buildPartitionedKernelMatrix(training,
      training.length, _blockSize, _blockSize,
      effectiveTrainingKernel.evaluate)
    caching = true
  }

  /**
    * Forget the cached kernel & noise matrices.
    * */
  def unpersist(): Unit = {
    kernelMatrixCache = null
    partitionedKernelMatrixCache = null
    caching = false
  }


}

object AbstractSTPRegressionModel {

  /**
    * Calculate the marginal log likelihood
    * of the training data for a pre-initialized
    * kernel and noise matrices.
    *
    * @param trainingData The function values assimilated as a [[DenseVector]]
    *
    * @param kernelMatrix The kernel matrix formed from the data features
    *
    * */
  def logLikelihood(
    mu: Double, trainingData: DenseVector[Double],
    kernelMatrix: DenseMatrix[Double]): Double = {


    try {
      val dist = MultivariateStudentsT(mu, DenseVector.zeros[Double](trainingData.length), kernelMatrix)
      -1.0*dist.logPdf(trainingData)
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }
  }

  def logLikelihood(
    mu: Double, trainingData: PartitionedVector,
    kernelMatrix: PartitionedPSDMatrix): Double = {

    try {
      val nE =
        if(trainingData.rowBlocks > 1L) trainingData(0L to 0L)._data.head._2.length
        else trainingData.rows.toInt

      val dist = BlockedMultivariateStudentsT(
        mu, PartitionedVector.zeros(trainingData.rows, nE),
        kernelMatrix)

      -1.0*dist.logPdf(trainingData)
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }

  }

}