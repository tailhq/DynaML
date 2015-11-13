package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, AbstractKernel}

/**
 * Gaussian Process Regression Model
 * Performs gp/spline smoothing/regression
 * with vector inputs and a singular scalar output.
 */
abstract class GPRegressionModel[T](
  cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
  data: T) extends
GaussianProcessModel[T, DenseVector[Double], Double,
  Double, DenseMatrix[Double], MultivariateGaussian]{

  /**
   * The GP is taken to be zero mean, or centered.
   * This is ensured by standardization of the data
   * before being used for further processing.
   *
   * */
  override val mean: (DenseVector[Double]) => Double = _ => 0.0

  override val covariance: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] = cov

  override protected val g: T = data

  /**
   * Calculates posterior predictive distribution for
   * a particular set of test data points.
   *
   * @param test A Sequence or Sequence like data structure
   *             storing the values of the input patters.
   **/
  override def predictiveDistribution[U <: Seq[DenseVector[Double]]](test: U): MultivariateGaussian = {

    //Calculate the kernel matrix on the training data
    //Calculate the predictive mean and co-variance
    new MultivariateGaussian(DenseVector(0.0), DenseMatrix(0.0))
  }

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - )
    * 3) Y+ : The upper error bar.
    **/
  override def predictionWithErrorBars[U <: Seq[DenseVector[Double]]](testData: U, confidence: Double):
  Seq[(DenseVector[Double], Double, Double, Double)] = {

    Seq()
  }

}
