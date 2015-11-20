package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{inv, DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.optimization.ConjugateGradient
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
  (DenseVector[Double], DenseMatrix[Double])]{

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

  val npoints = num

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
    /*val meanBuff = ConjugateGradient.runCG(kernelTraining, trainingLabels,
      DenseVector.ones[Double](training.length), 0.0001, 20)

    logger.info("Predictive Mean: \n"+crossKernel.t * meanBuff)

    val CovBuff = ConjugateGradient.runMultiCG(kernelTraining,
      crossKernel, DenseMatrix.ones[Double](training.length, test.length),
      0.0001, 20)

    logger.info("Predictive Co-variance: \n"+(kernelTest - (crossKernel.t * CovBuff)))
    (crossKernel.t * meanBuff, kernelTest - (crossKernel.t * CovBuff))*/
    val inverse = inv(kernelTraining)
    (crossKernel.t * (inverse * trainingLabels), kernelTest - (crossKernel.t * (inverse * crossKernel)))
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
