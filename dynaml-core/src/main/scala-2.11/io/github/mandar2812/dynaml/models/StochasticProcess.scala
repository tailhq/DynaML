package io.github.mandar2812.dynaml.models

import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import org.apache.log4j.Logger

/**
  * date 26/08/16.
  * High Level description of a stochastic process.
  *
  * @author mandar2812
  * @tparam T The underlying data structure storing the training & test data.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam W Implementing class of the posterior distribution
  */
trait StochasticProcess[T, I, Y, W] extends Model[T, I, Y] {

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

/**
  * @author mandar2812
  *
  * Processes which can be specified by upto second order statistics i.e. mean and covariance
  * @tparam T The underlying data structure storing the training & test data.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam K The type returned by the kernel function.
  * @tparam M The data structure holding the kernel/covariance matrix
  * @tparam W Implementing class of the posterior distribution
  *
  * */
abstract class SecondOrderProcess[T, I, Y, K, M, W] extends StochasticProcess[T, I, Y, W] {

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


}
