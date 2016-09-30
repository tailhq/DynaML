package io.github.mandar2812.dynaml.utils

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * @author mandar2812 date: 17/6/16.
  *
  * Scales attributes of a vector pattern using the sample mean and variance of
  * each dimension. This assumes that there is no covariance between the data
  * dimensions.
  *
  * @param mean Sample mean of the data
  * @param sigma Sample variance of each data dimension
  */
case class GaussianScaler(mean: DenseVector[Double], sigma: DenseVector[Double])
  extends ReversibleScaler[DenseVector[Double]]{
  override val i: Scaler[DenseVector[Double]] =
    Scaler((pattern: DenseVector[Double]) => (pattern :* sigma) + mean)

  override def run(data: DenseVector[Double]): DenseVector[Double] = (data-mean) :/ sigma

  def apply(r: Range): GaussianScaler = GaussianScaler(mean(r), sigma(r))
}


/**
  * Scales the attributes of a data pattern using the sample mean and covariance matrix
  * calculated on the data set. This allows standardization of multivariate data sets
  * where the covariance of individual data dimensions is not negligible.
  *
  * @param mean Sample mean of data
  * @param sigma Sample covariance matrix of data.
  * */
case class MVGaussianScaler(mean: DenseVector[Double], sigma: DenseMatrix[Double])
  extends ReversibleScaler[DenseVector[Double]] {

  val sigmaInverse = cholesky(inv(sigma))

  override val i: Scaler[DenseVector[Double]] =
    Scaler((pattern: DenseVector[Double]) => (inv(sigmaInverse.t) * pattern) + mean)

  override def run(data: DenseVector[Double]): DenseVector[Double] = sigmaInverse.t * (data - mean)

  def apply(r: Range): MVGaussianScaler = MVGaussianScaler(mean(r), sigma(r,r))
}