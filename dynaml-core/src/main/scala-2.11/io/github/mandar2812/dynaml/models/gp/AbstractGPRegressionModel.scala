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
package io.github.mandar2812.dynaml.models.gp

import breeze.linalg._
import breeze.numerics.log
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, DiracKernel}
import io.github.mandar2812.dynaml.optimization.{GloballyOptWithGrad, GloballyOptimizable}
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import org.apache.log4j.Logger

/**
  * Single-Output Gaussian Process Regression Model
  * Performs gp/spline smoothing/regression with
  * vector inputs and a singular scalar output.
  *
  * @tparam T The data structure holding the training data.
  *
  * @tparam I The index set over which the Gaussian Process
  *           is defined.
  *           e.g  1) I = Double when implementing GP time series
  *                2) I = DenseVector when implementing GP regression
  *
  */
abstract class AbstractGPRegressionModel[T, I](
  cov: CovarianceFunction[I, Double, DenseMatrix[Double]],
  n: CovarianceFunction[I, Double, DenseMatrix[Double]],
  data: T, num: Int) extends
  GaussianProcessModel[T, I, Double, Double, DenseMatrix[Double],
  MultGaussianRV]
with GloballyOptWithGrad {

  private val logger = Logger.getLogger(this.getClass)

  /**
   * The GP is taken to be zero mean, or centered.
   * This is ensured by standardization of the data
   * before being used for further processing.
   *
   * */
  override val mean: (I) => Double = _ => 0.0

  override val covariance = cov

  val noiseModel: CovarianceFunction[I, Double, DenseMatrix[Double]] = n

  override protected val g: T = data

  val npoints = num

  protected var (caching, kernelMatrixCache, noiseCache)
  : (Boolean, DenseMatrix[Double], DenseMatrix[Double]) = (false, null, null)


  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {
    covariance.setHyperParameters(s)
    noiseModel.setHyperParameters(s)
    current_state = covariance.state ++ noiseModel.state
    this
  }

  override protected var hyper_parameters: List[String] =
    covariance.hyper_parameters ++ noiseModel.hyper_parameters

  override protected var current_state: Map[String, Double] =
    covariance.state ++ noiseModel.state

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
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val kernelTraining: DenseMatrix[Double] =
      covariance.buildKernelMatrix(training, npoints).getKernelMatrix()

    val noiseMat = noiseModel.buildKernelMatrix(training, npoints).getKernelMatrix()

    AbstractGPRegressionModel.logLikelihood(trainingLabels, kernelTraining, noiseMat)
  }

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
    **/
  override def gradEnergy(h: Map[String, Double]): Map[String, Double] = {

    covariance.setHyperParameters(h)
    noiseModel.setHyperParameters(h)

    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val inverse = inv(covariance.buildKernelMatrix(training, npoints).getKernelMatrix() +
      noiseModel.buildKernelMatrix(training, npoints).getKernelMatrix())

    val hParams = covariance.hyper_parameters ++ noiseModel.hyper_parameters
    val alpha = inverse * trainingLabels
    hParams.map(h => {
      //build kernel derivative matrix
      val kernelDerivative =
        if(noiseModel.hyper_parameters.contains(h))
          DenseMatrix.tabulate[Double](npoints, npoints){(i,j) => {
            noiseModel.gradient(training(i), training(j))(h)
          }}
        else
          DenseMatrix.tabulate[Double](npoints, npoints){(i,j) => {
            covariance.gradient(training(i), training(j))(h)
          }}
      //Calculate gradient for the hyper parameter h
      val grad: DenseMatrix[Double] = (alpha*alpha.t - inverse)*kernelDerivative
      (h, trace(grad))
    }).toMap
  }

  /**
   * Calculates posterior predictive distribution for
   * a particular set of test data points.
   *
   * @param test A Sequence or Sequence like data structure
   *             storing the values of the input patters.
   **/
  override def predictiveDistribution[U <: Seq[I]](test: U):
  MultGaussianRV = {

    logger.info("Calculating posterior predictive distribution")
    //Calculate the kernel matrix on the training data
    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val kernelTraining = if(!caching)
      covariance.buildKernelMatrix(training, npoints).getKernelMatrix()
    else
      kernelMatrixCache

    val noiseMat = if(!caching)
      noiseModel.buildKernelMatrix(training, npoints).getKernelMatrix()
    else
      noiseCache

    val kernelTest = covariance.buildKernelMatrix(test, test.length)
      .getKernelMatrix()
    val crossKernel = covariance.buildCrossKernelMatrix(training, test)

    //Calculate the predictive mean and co-variance
    val smoothingMat = kernelTraining + noiseMat
    val Lmat = cholesky(smoothingMat)
    val alpha = Lmat.t \ (Lmat \ trainingLabels)
    val v = Lmat \ crossKernel

    val varianceReducer = v.t * v
    //Ensure that v is symmetric

    val adjustedVarReducer = DenseMatrix.tabulate[Double](varianceReducer.rows, varianceReducer.cols)(
      (i,j) => if(i <= j) varianceReducer(i,j) else varianceReducer(j,i))

    MultGaussianRV(test.length)(
      crossKernel.t * alpha,
      kernelTest - adjustedVarReducer)
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
    val postcov = posterior.covariance
    val postmean = posterior.mu
    val stdDev = (1 to testData.length).map(i => math.sqrt(postcov(i-1, i-1)))
    val mean = postmean.toArray.toSeq

    logger.info("Generating error bars")
    val preds = (mean zip stdDev).map(j => (j._1, j._1 - sigma*j._2, j._1 + sigma*j._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }


  override def predict(point: I): Double = predictionWithErrorBars(Seq(point), 1).head._2

  /**
    * Returns a prediction with error bars for a test set of indexes and labels.
    * (Index, Actual Value, Prediction, Lower Bar, Higher Bar)
    * */
  def test(testData: T): Seq[(I, Double, Double, Double, Double)] = {
    logger.info("Generating predictions for test set")
    //Calculate the posterior predictive distribution for the test points.
    val predictionWithError = this.predictionWithErrorBars(dataAsIndexSeq(testData), 1)
    //Collate the test data with the predictions and error bars
    dataAsSeq(testData).zip(predictionWithError).map(i => (i._1._1, i._1._2,
      i._2._2, i._2._3, i._2._4))
  }

  /**
    * Cache the training kernel and noise matrices
    * for fast access in future predictions.
    * */
  def persist(): Unit = {
    kernelMatrixCache =
      covariance.buildKernelMatrix(dataAsIndexSeq(g), npoints)
        .getKernelMatrix()
    noiseCache = noiseModel.buildKernelMatrix(dataAsIndexSeq(g), npoints)
      .getKernelMatrix()
    caching = true

  }

  /**
    * Forget the cached kernel & noise matrices.
    * */
  def unpersist(): Unit = {
    kernelMatrixCache = null
    noiseCache = null
    caching = false
  }

}

object AbstractGPRegressionModel {

  /**
    * Calculate the marginal log likelihood
    * of the training data for a pre-initialized
    * kernel and noise matrices.
    *
    * @param trainingData The function values assimilated as a [[DenseVector]]
    *
    * @param kernelMatrix The kernel matrix of the training features
    *
    * @param noiseMatrix The noise matrix with respect to the training data features
    * */
  def logLikelihood(trainingData: DenseVector[Double],
                    kernelMatrix: DenseMatrix[Double],
                    noiseMatrix: DenseMatrix[Double]): Double = {

    val smoothingMat = kernelMatrix + noiseMatrix
    val Lmat = cholesky(smoothingMat)
    val alpha = Lmat.t \ (Lmat \ trainingData)

    0.5*((trainingData dot alpha) +
      trace(log(Lmat)) +
      trainingData.length*math.log(2*math.Pi))

  }

  def apply[M <: AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
    DenseVector[Double]]](data: Seq[(DenseVector[Double], Double)],
                          cov: CovarianceFunction[DenseVector[Double],
                            Double, DenseMatrix[Double]],
                          noise: CovarianceFunction[DenseVector[Double],
                            Double, DenseMatrix[Double]] = new DiracKernel(1.0),
                          order: Int = 0, ex: Int = 0): M = {
    assert(ex >= 0 && order >= 0, "Non Negative values for order and ex")
    if(order == 0) new GPRegression(cov, noise, data).asInstanceOf[M]
    else if(order > 0 && ex == 0) new GPNarModel(order, cov, noise, data).asInstanceOf[M]
    else new GPNarXModel(order, ex, cov, noise, data).asInstanceOf[M]
  }
}