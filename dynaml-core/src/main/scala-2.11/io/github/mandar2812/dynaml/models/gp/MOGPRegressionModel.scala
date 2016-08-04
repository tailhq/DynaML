package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import breeze.stats.distributions.Gaussian
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import org.apache.log4j.Logger

/**
  * @author mandar2812 date: 19/7/16.
  *
  * Abstract implementation of multi-output gaussian process
  * as outlined in Lawrence et. al 2012 on arxiv
  *
  * @tparam I The index set of the GP.
  */
class MOGPRegressionModel[I](
  cov: CovarianceFunction[(I, Int), Double, DenseMatrix[Double]],
  n: CovarianceFunction[(I, Int), Double, DenseMatrix[Double]],
  data: Stream[(I, DenseVector[Double])],
  num: Int, numOutputs: Int) extends
  AbstractGPRegressionModel[
    Stream[(I, DenseVector[Double])],
    (I, Int)](cov, n, data, num*numOutputs) {

  private val logger = Logger.getLogger(this.getClass)

  val noutputs = numOutputs

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Stream[(I, DenseVector[Double])]): Seq[((I, Int), Double)] =
    data.map((patternAndLabel) =>
      patternAndLabel._2.mapPairs((i, label) =>
        ((patternAndLabel._1, i), label)
      ).toArray.toSeq).reduceLeft((s1, s2) => s1 ++ s2)
}
