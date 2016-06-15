package io.github.mandar2812.dynaml.models.gp

import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.Model
import org.apache.log4j.Logger


/**
 * High Level description of a Gaussian Process.
 * @author mandar2812
 * @tparam T The underlying data structure storing the training & test data.
 * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
 * @tparam Y The type of the output label
 * @tparam K The type of value returned by the covariance kernel function
 * @tparam M The underlying data structure of the kernel Matrix
 * @tparam W Implementing class of the posterior distribution
 **/
abstract class GaussianProcessModel[T, I, Y, K, M, W] extends Model[T, I, Y] {

  private val logger = Logger.getLogger(this.getClass)

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
  val covariance: CovarianceFunction[I, K, M]

  /** Calculates posterior predictive distribution for
  * a particular set of test data points.
  *
  * @param test A Sequence or Sequence like data structure
  *             storing the values of the input patters.
  * */
  def predictiveDistribution[U <: Seq[I]](test: U): W


  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    * */

  def dataAsSeq(data: T): Seq[(I,Y)]

  /**
    * Convert from the underlying data structure to
    * Seq[I] where I is the index set of the GP
    * */
  def dataAsIndexSeq(data: T): Seq[I] = dataAsSeq(data).map(_._1)


}
