package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{det, inv, DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
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

  override val covariance: CovarianceFunction[I, Double, DenseMatrix[Double]] = cov

  override protected val g: T = data

  var noiseLevel: Double = 1.0

  val npoints = num

  def setNoiseLevel(n: Double): this.type = {
    noiseLevel = n
    this
  }

  override protected var hyper_parameters: List[String] =
    cov.hyper_parameters :+ "noiseLevel"

  override protected var current_state: Map[String, Double] =
    cov.state + (("noiseLevel", noiseLevel))


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

    this.setNoiseLevel(h("noiseLevel"))
    covariance.setHyperParameters(h)

    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val kernelTraining: DenseMatrix[Double] =
      covariance.buildKernelMatrix(training, npoints).getKernelMatrix()

    AbstractGPRegressionModel.logLikelihood(trainingLabels, kernelTraining, noiseLevel)
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
    val kernelTraining = covariance.buildKernelMatrix(training, npoints).getKernelMatrix()
    val kernelTest = covariance.buildKernelMatrix(test, test.length).getKernelMatrix()
    val crossKernel = covariance.buildCrossKernelMatrix(training, test)

    //Calculate the predictive mean and co-variance
    val inverse = inv(kernelTraining + DenseMatrix.eye[Double](training.length) * noiseLevel)
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

}

object AbstractGPRegressionModel {

  def logLikelihood(trainingData: DenseVector[Double],
                    kernelMatrix: DenseMatrix[Double],
                    noise: Double): Double = {

    val kernelTraining: DenseMatrix[Double] =
      kernelMatrix + DenseMatrix.eye[Double](trainingData.length) * noise
    val Kinv = inv(kernelTraining)

    0.5*(trainingData.t * (Kinv * trainingData) + math.log(det(kernelTraining)) +
      trainingData.length*math.log(2*math.Pi))
  }
}