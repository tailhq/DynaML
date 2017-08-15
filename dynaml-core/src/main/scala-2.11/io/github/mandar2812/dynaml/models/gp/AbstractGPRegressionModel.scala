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
package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, trace}
import breeze.numerics.{log, sqrt}
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.optimization.GloballyOptWithGrad
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}
import io.github.mandar2812.dynaml.probability.{MultGaussianPRV, MultGaussianRV}
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
  * <h3>Gaussian Process Regression</h3>
  *
  * Single-Output Gaussian Process Regression Model
  * Performs gp/spline smoothing/regression with
  * vector inputs and a singular scalar output.
  *
  * @tparam T The data structure holding the training data.
  *
  * @tparam I The index set over which the Gaussian Process
  *           is defined.
  *           e.g:
  *
  *           <ul>
  *             <li>I = Double when implementing GP time series</li>
  *             <li>I = DenseVector when implementing GP regression</li>
  *           <ul>
  *
  * @param cov The covariance function/kernel of the GP model,
  *            expressed as a [[LocalScalarKernel]] instance
  *
  * @param n Measurement noise covariance of the GP model.
  *
  * @param data Training data set of generic type [[T]]
  *
  * @param num The number of training data instances.
  *
  * @param meanFunc The mean/trend function of the GP model expressed as
  *                 a [[DataPipe]] instance.
  * */
abstract class AbstractGPRegressionModel[T, I: ClassTag](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  data: T, num: Int, meanFunc: DataPipe[I, Double] = DataPipe((_:I) => 0.0))
  extends ContinuousProcessModel[T, I, Double, MultGaussianPRV]
  with SecondOrderProcessModel[T, I, Double, Double, DenseMatrix[Double], MultGaussianPRV]
  with GloballyOptWithGrad {

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
    this
  }

  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters

  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state

  protected lazy val trainingData: Seq[I] = dataAsIndexSeq(g)

  protected lazy val trainingDataLabels = PartitionedVector(
    dataAsSeq(g).toStream.map(_._2),
    trainingData.length.toLong, _blockSize
  )

  /**
    * Returns a [[DataPipe2]] which calculates the energy of data: [[T]].
    * See: [[energy]] below.
    * */
  def calculateEnergyPipe(h: Map[String, Double], options: Map[String, String]) =
    DataPipe2((training: Seq[I], trainingLabels: PartitionedVector) => {
      setState(h)


      val trainingMean = PartitionedVector(
        training.toStream.map(mean(_)),
        training.length.toLong, _blockSize
      )

      //val effectiveTrainingKernel: LocalScalarKernel[I] = this.covariance + this.noiseModel

      //effectiveTrainingKernel.setBlockSizes((_blockSize, _blockSize))

      val kernelTraining: PartitionedPSDMatrix = getTrainKernelMatrix

      AbstractGPRegressionModel.logLikelihood(trainingLabels - trainingMean, kernelTraining)
    })

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
    * */
  override def energy(h: Map[String, Double], options: Map[String, String]): Double =
    calculateEnergyPipe(h, options)(trainingData, trainingDataLabels)

  /**
    * Returns a [[DataPipe]] which calculates the gradient of the energy, E(.) of data: [[T]]
    * with respect to the model hyper-parameters.
    * See: [[gradEnergy]] below.
    * */
  def calculateGradEnergyPipe(h: Map[String, Double]) =
    DataPipe2((training: Seq[I], trainingLabels: PartitionedVector) => {
      try {
        covariance.setHyperParameters(h)
        noiseModel.setHyperParameters(h)

        val trainingMean = PartitionedVector(
          training.toStream.map(mean(_)),
          training.length.toLong, _blockSize
        )

        val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel

        effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))
        val hParams = effectiveTrainingKernel.effective_hyper_parameters

        val gradMatrices = SVMKernel.buildPartitionedKernelGradMatrix(
          training, training.length, _blockSize, _blockSize,
          hParams, (x: I, y: I) => effectiveTrainingKernel.evaluate(x,y),
          (hy: String) => (x: I, y: I) => effectiveTrainingKernel.gradient(x,y)(hy))

        val kernelTraining: PartitionedPSDMatrix = gradMatrices("kernel-matrix")

        val Lmat = bcholesky(kernelTraining)

        val alpha = Lmat.t \\ (Lmat \\ (trainingLabels-trainingMean))

        hParams.map(h => {
          //build kernel derivative matrix
          val kernelDerivative: PartitionedMatrix = gradMatrices(h)
          //Calculate gradient for the hyper parameter h
          val grad: PartitionedMatrix =
            alpha*alpha.t*kernelDerivative - (Lmat.t \\ (Lmat \\ kernelDerivative))

          (h.split("/").tail.mkString("/"), btrace(grad))
        }).toMap

      } catch {
        case _: breeze.linalg.NotConvergedException =>
          covariance.effective_hyper_parameters.map(h => (h, Double.NaN)).toMap ++
            noiseModel.effective_hyper_parameters.map(h => (h, Double.NaN)).toMap
      }

    })

  /**
    * Calculates the gradient energy of the configuration and
    * subtracts this from the current value of h to yield a new
    * hyper-parameter configuration.
    *
    * Over ride this function if you aim to implement a gradient based
    * hyper-parameter optimization routine like ML-II
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @return Gradient of the objective function (marginal likelihood) as a Map
    * */
  override def gradEnergy(h: Map[String, Double]): Map[String, Double] =
    calculateGradEnergyPipe(h)(trainingData, trainingDataLabels)


  protected def getTrainKernelMatrix[U <: Seq[I]] = {
    SVMKernel.buildPartitionedKernelMatrix(trainingData,
        trainingData.length, _blockSize, _blockSize,
        (x: I, y: I) => {covariance.evaluate(x, y) + noiseModel.evaluate(x, y)})
  }

  protected def getCrossKernelMatrix[U <: Seq[I]](test: U) =
    SVMKernel.crossPartitonedKernelMatrix(
      trainingData, test,
      _blockSize, _blockSize,
      covariance.evaluate)

  protected def getTestKernelMatrix[U <: Seq[I]](test: U) =
    SVMKernel.buildPartitionedKernelMatrix(
      test, test.length.toLong,
      _blockSize, _blockSize,
      covariance.evaluate)

  /**
   * Calculates posterior predictive distribution for
   * a particular set of test data points.
   *
   * @param test A Sequence or Sequence like data structure
   *             storing the values of the input patters.
   * */
  override def predictiveDistribution[U <: Seq[I]](test: U):
  MultGaussianPRV = {

    logger.info("Calculating posterior predictive distribution")
    //Calculate the kernel matrix on the training data


    val priorMeanTest = PartitionedVector(
      test.map(mean(_))
        .grouped(_blockSize)
        .zipWithIndex.map(c => (c._2.toLong, DenseVector(c._1.toArray)))
        .toStream,
      test.length.toLong)

    val trainingMean = PartitionedVector(
      trainingData.map(mean(_)).toStream,
      trainingData.length.toLong, _blockSize
    )

    val smoothingMat = if(!caching) {
      logger.info("---------------------------------------------------------------")
      logger.info("Calculating covariance matrix for training points")
      getTrainKernelMatrix
    } else {
      logger.info("** Using cached training matrix **")
      partitionedKernelMatrixCache
    }

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix for test points")
    val kernelTest = getTestKernelMatrix(test)

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix between training and test points")
    val crossKernel = getCrossKernelMatrix(test)

    //Calculate the predictive mean and co-variance
    val (postPredictiveMean, postPredictiveCovariance) =
      AbstractGPRegressionModel.solve(
        trainingDataLabels, trainingMean, priorMeanTest,
        smoothingMat, kernelTest, crossKernel)

    MultGaussianPRV(test.length.toLong, _blockSize)(
      postPredictiveMean,
      postPredictiveCovariance)
  }

  /**
    * Draw three predictions from the posterior predictive distribution
    *
    * <ol>
    *   <li>Mean or MAP estimate Y</li>
    *   <li>Y- : The lower error bar estimate (mean - sigma*stdDeviation)</li>
    *   <li>Y+ : The upper error bar. (mean + sigma*stdDeviation)</li>
    * </ol>
    *
    **/
  def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int):
  Seq[(I, Double, Double, Double)] = {

    val posterior = predictiveDistribution(testData)

    val mean = posterior.mu.toStream

    val (lower, upper) = posterior.underlyingDist.confidenceInterval(sigma.toDouble)

    val lowerErrorBars = lower.toStream
    val upperErrorBars = upper.toStream

    logger.info("Generating error bars")

    val preds = mean.zip(lowerErrorBars.zip(upperErrorBars)).map(t => (t._1, t._2._1, t._2._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }


  override def predict(point: I): Double = predictionWithErrorBars(Seq(point), 1).head._2


  /**
    * Cache the training kernel and noise matrices
    * for fast access in future predictions.
    * */
  override def persist(state: Map[String, Double]): Unit = {
    setState(state)
    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    effectiveTrainingKernel.setBlockSizes((blockSize, blockSize))

    partitionedKernelMatrixCache = SVMKernel.buildPartitionedKernelMatrix(trainingData,
      trainingData.length, _blockSize, _blockSize,
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

object AbstractGPRegressionModel {

  /**
    * Calculate the parameters of the posterior predictive distribution
    * for a multivariate gaussian model.
    * */
  def solve(
    trainingLabels: PartitionedVector,
    trainingMean: PartitionedVector,
    priorMeanTest: PartitionedVector,
    smoothingMat: PartitionedPSDMatrix,
    kernelTest: PartitionedPSDMatrix,
    crossKernel: PartitionedMatrix):
  (PartitionedVector, PartitionedPSDMatrix) = {

    val Lmat: LowerTriPartitionedMatrix = bcholesky(smoothingMat)

    val alpha: PartitionedVector = Lmat.t \\ (Lmat \\ (trainingLabels-trainingMean))

    val v: PartitionedMatrix = Lmat \\ crossKernel

    val varianceReducer: PartitionedMatrix = v.t * v

    //Ensure that the variance reduction is symmetric
    val adjustedVarReducer: PartitionedMatrix = varianceReducer /*(varianceReducer.L + varianceReducer.L.t).map(bm =>
      if(bm._1._1 == bm._1._2) (bm._1, bm._2*(DenseMatrix.eye[Double](bm._2.rows)*0.5))
      else bm)*/

    val reducedVariance: PartitionedPSDMatrix =
      new PartitionedPSDMatrix(
        (kernelTest - adjustedVarReducer).filterBlocks(c => c._1 >= c._2),
        kernelTest.rows, kernelTest.cols)

    (priorMeanTest + crossKernel.t * alpha, reducedVariance)
  }

  /**
    * Calculate the marginal log likelihood
    * of the training data for a pre-initialized
    * kernel and noise matrices.
    *
    * @param trainingData The function values assimilated as a [[DenseVector]]
    *
    * @param kernelMatrix The kernel matrix of the training features
    *
    * */
  def logLikelihood(trainingData: DenseVector[Double],
                    kernelMatrix: DenseMatrix[Double]): Double = {

    val smoothingMat = kernelMatrix

    try {
      val Lmat = cholesky(smoothingMat)
      val alpha = Lmat.t \ (Lmat \ trainingData)

      0.5*((trainingData dot alpha) +
        trace(log(Lmat)) +
        trainingData.length*math.log(2*math.Pi))
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }
  }


  /**
    * Calculate the marginal log likelihood
    * of the training data for a pre-initialized
    * kernel and noise matrices.
    *
    * @param trainingData The function values assimilated as a [[DenseVector]]
    *
    * @param kernelMatrix The kernel matrix of the training features
    *
    * */
  def logLikelihood(trainingData: PartitionedVector,
                    kernelMatrix: PartitionedPSDMatrix): Double = {

    val smoothingMat = kernelMatrix

    try {
      val Lmat = bcholesky(smoothingMat)
      val alpha: PartitionedVector = Lmat.t \\ (Lmat \\ trainingData)

      val d: Double = trainingData dot alpha

      0.5*(d +
        btrace(blog(Lmat)) +
        trainingData.rows*math.log(2*math.Pi))
    } catch {
      case _: breeze.linalg.NotConvergedException => Double.PositiveInfinity
      case _: breeze.linalg.MatrixNotSymmetricException => Double.PositiveInfinity
    }
  }


  def apply[M <: AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
    DenseVector[Double]]](data: Seq[(DenseVector[Double], Double)],
                          cov: LocalScalarKernel[DenseVector[Double]],
                          noise: LocalScalarKernel[DenseVector[Double]] = new DiracKernel(1.0),
                          order: Int = 0, ex: Int = 0,
                          meanFunc: DataPipe[DenseVector[Double], Double] = DataPipe(_ => 0.0)): M = {
    assert(ex >= 0 && order >= 0, "Non Negative values for order and ex")
    if(order == 0) new GPRegression(cov, noise, data).asInstanceOf[M]
    else if(order > 0 && ex == 0) new GPNarModel(order, cov, noise, data).asInstanceOf[M]
    else new GPNarXModel(order, ex, cov, noise, data).asInstanceOf[M]
  }

  /**
    * Create an instance of [[AbstractGPRegressionModel]] for a
    * particular data type [[T]]
    *
    * @tparam T The type of the training data
    * @tparam I The type of the input patterns in the data set of type [[T]]
    *
    * @param cov The covariance function
    * @param noise The noise covariance function
    * @param meanFunc The trend or mean function
    * @param trainingdata The actual data set of type [[T]]
    * @param transform An implicit conversion from [[T]] to [[Seq]] represented as a [[DataPipe]]
    * */
  def apply[T, I: ClassTag](
    cov: LocalScalarKernel[I],
    noise: LocalScalarKernel[I],
    meanFunc: DataPipe[I, Double])(
    trainingdata: T, num: Int)(
    implicit transform: DataPipe[T, Seq[(I, Double)]]) = {

    val num_points = if(num > 0) num else transform(trainingdata).length

    new AbstractGPRegressionModel[T, I](cov, noise, trainingdata, num_points, meanFunc) {
      /**
        * Convert from the underlying data structure to
        * Seq[(I, Y)] where I is the index set of the GP
        * and Y is the value/label type.
        **/
      override def dataAsSeq(data: T) = transform(data)
    }

  }

  /**
    * Create an instance of [[GPBasisFuncRegressionModel]] for a
    * particular data type [[T]]
    *
    * @tparam T The type of the training data
    * @tparam I The type of the input patterns in the data set of type [[T]]
    *
    * @param cov The covariance function
    * @param noise The noise covariance function
    * @param basisFunc A [[DataPipe]] transforming the input features to basis function components.
    * @param basis_param_prior A [[MultGaussianRV]] which is the prior
    *                          distribution on basis function coefficients
    * @param trainingdata The actual data set of type [[T]]
    * @param transform An implicit conversion from [[T]] to [[Seq]] represented as a [[DataPipe]]
    * */
  def apply[T, I: ClassTag](
    cov: LocalScalarKernel[I],
    noise: LocalScalarKernel[I],
    basisFunc: DataPipe[I, DenseVector[Double]],
    basis_param_prior: MultGaussianRV)(
    trainingdata: T, num: Int)(
    implicit transform: DataPipe[T, Seq[(I, Double)]]) = {

    val num_points = if(num > 0) num else transform(trainingdata).length

    new GPBasisFuncRegressionModel[T, I](cov, noise, trainingdata, num_points, basisFunc, basis_param_prior) {
      /**
        * Convert from the underlying data structure to
        * Seq[(I, Y)] where I is the index set of the GP
        * and Y is the value/label type.
        **/
      override def dataAsSeq(data: T) = transform(data)
    }

  }


}

abstract class KroneckerGPRegressionModel[T, I: ClassTag, J: ClassTag](
  cov: KroneckerProductKernel[I, J], n: KroneckerProductKernel[I, J],
  data: T, num: Int, meanFunc: DataPipe[(I, J), Double] = DataPipe((_:(I, J)) => 0.0))
  extends AbstractGPRegressionModel[T, (I,J)](cov, n, data, num, meanFunc)
