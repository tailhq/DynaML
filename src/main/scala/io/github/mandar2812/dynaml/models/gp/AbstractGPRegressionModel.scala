package io.github.mandar2812.dynaml.models.gp

import breeze.linalg._
import io.github.mandar2812.dynaml.kernels.{DiracKernel, CovarianceFunction}
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
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
  n: CovarianceFunction[I, Double, DenseMatrix[Double]] = new DiracKernel(1.0),
  data: T, num: Int) extends
  GaussianProcessModel[T, I, Double, Double, DenseMatrix[Double],
  (DenseVector[Double], DenseMatrix[Double])]
with GloballyOptimizable {

  private val logger = Logger.getLogger(this.getClass)

  /**
   * The GP is taken to be zero mean, or centered.
   * This is ensured by standardization of the data
   * before being used for further processing.
   *
   * */
  override val mean: (I) => Double = _ => 0.0

  override val covariance = cov

  override val noiseModel = n

  override protected val g: T = data

  var noiseLevel: Double = 1.0

  val npoints = num

  protected var (caching, kernelMatrixCache, noiseCache)
  : (Boolean, DenseMatrix[Double], DenseMatrix[Double]) = (false, null, null)

  def setNoiseLevel(n: Double): this.type = {
    noiseLevel = n
    this
  }

  def setState(s: Map[String, Double]): this.type ={
    covariance.setHyperParameters(s)
    noiseModel.setHyperParameters(s)
    //noiseLevel = s("noiseLevel")
    current_state = cov.state ++ noiseModel.state
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

    //this.setNoiseLevel(h("noiseLevel"))
    covariance.setHyperParameters(h)

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

    this.setNoiseLevel(h("noiseLevel"))
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
  (DenseVector[Double], DenseMatrix[Double]) = {

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
    val inverse = inv(kernelTraining + noiseMat)
    (crossKernel.t * (inverse * trainingLabels),
      kernelTest - (crossKernel.t * (inverse * crossKernel)))
  }

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - )
    * 3) Y+ : The upper error bar.
    **/
  override def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int):
  Seq[(I, Double, Double, Double)] = {

    val posterior = predictiveDistribution(testData)
    val stdDev = (1 to testData.length).map(i => math.sqrt(posterior._2(i-1, i-1)))
    val mean = posterior._1.toArray.toSeq

    logger.info("Generating error bars")
    val preds = (mean zip stdDev).map(j => (j._1, j._1 - sigma*j._2, j._1 + sigma*j._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }

  def persist(): Unit = {
    kernelMatrixCache =
      covariance.buildKernelMatrix(dataAsIndexSeq(g), npoints)
        .getKernelMatrix()
    noiseCache = noiseModel.buildKernelMatrix(dataAsIndexSeq(g), npoints)
      .getKernelMatrix()
    caching = true

  }

  def unpersist(): Unit = {
    kernelMatrixCache = null
    noiseCache = null
    caching = false
  }

}

object AbstractGPRegressionModel {

  def logLikelihood(trainingData: DenseVector[Double],
                    kernelMatrix: DenseMatrix[Double],
                    noiseMatrix: DenseMatrix[Double]): Double = {

    val kernelTraining: DenseMatrix[Double] =
      kernelMatrix + noiseMatrix
    val Kinv = inv(kernelTraining)

    0.5*(trainingData.t * (Kinv * trainingData) + math.log(det(kernelTraining)) +
      trainingData.length*math.log(2*math.Pi))
  }
}