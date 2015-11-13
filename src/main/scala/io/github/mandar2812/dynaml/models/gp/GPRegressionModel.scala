package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, AbstractKernel}
import io.github.mandar2812.dynaml.optimization.ConjugateGradient

/**
 * Gaussian Process Regression Model
 * Performs gp/spline smoothing/regression
 * with vector inputs and a singular scalar output.
 */
abstract class GPRegressionModel[T, I](
  cov: CovarianceFunction[I, Double, DenseMatrix[Double]],
  data: T, num: Int) extends
GaussianProcessModel[T, I, Double, Double, DenseMatrix[Double], MultivariateGaussian]{

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
  override def predictiveDistribution[U <: Seq[I]](test: U): MultivariateGaussian = {

    //Calculate the kernel matrix on the training data
    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)
    val kernelTraining = covariance.buildKernelMatrix(training, npoints).getKernelMatrix()
    val kernelTest = covariance.buildKernelMatrix(test, test.length).getKernelMatrix()
    val crossKernel = covariance.buildCrossKernelMatrix(training, test)

    //Calculate the predictive mean and co-variance
    val meanBuff = ConjugateGradient.runCG(kernelTraining, trainingLabels,
      DenseVector.ones[Double](test.length), 0.0001, 50)
    val CovBuff = ConjugateGradient.runMultiCG(kernelTraining,
      crossKernel, DenseMatrix.ones[Double](training.length, test.length),
      0.0001, 50)

    new MultivariateGaussian(crossKernel.t * meanBuff,
      kernelTest - (crossKernel.t * CovBuff))
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
    val stdDev = (1 to testData.length).map(i => math.sqrt(posterior.covariance(i-1, i-1)))
    val mean = posterior.mean.toArray.toSeq
    val preds = (mean zip stdDev).map(j => (j._1, j._1 - sigma*j._2, j._1 + sigma*j._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }

}
