package io.github.mandar2812.dynaml.models.gp

import breeze.stats.distributions.MultivariateGaussian
import io.github.mandar2812.dynaml.kernels.AbstractKernel
import io.github.mandar2812.dynaml.models.Model


/**
 * High Level description of a Gaussian Process.
 * @author mandar2812
 * @tparam T The underlying data structure storing the training data.
 * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
 * @tparam Y The type of the output label
 */
abstract class GaussianProcessModel[T, I, Y] extends Model[T] {

  /**
   * Mean Function: Takes a member of the index set (input)
   * and returns the corresponding mean of the distribution
   * corresponding to input.
   * */
  val mean: (I) => Y

  /**
   * Underlying covariance function of the
   * Gaussian Processes.
   * */
  val covariance: AbstractKernel[I]

  /**
   * Calculates posterior predictive distribution for
   * a particular set of test data points.
   *
   * @param test A Sequence or Sequence like data structure
   *             storing the values of the input patters.
   * */
  def predictiveDistribution[U <: Seq[I]](test: U): MultivariateGaussian

  /**
   * Returns a prediction with error bars for a test set.
   * */
  def test[U <: Seq[(I,Y)], V <: Seq[(I, Y, Y, Y, Y)]](test: U): V

}
