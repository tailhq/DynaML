package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.{DiracKernel, CovarianceFunction}

/**
  *
  * @author mandar2812 date 17/11/15.
  *
  * Class representing Gaussian Process regression models
  *
  * y = f(x) + e
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(f(x), noise(X,X))
  */
class GPRegression(
  cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
  noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] = new DiracKernel(1.0),
  trainingdata: Seq[(DenseVector[Double], Double)]) extends
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]](cov, noise, trainingdata,
  trainingdata.length){

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(DenseVector[Double], Double)]) = data
}
