package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gaussian, MultivariateGaussian}
import io.github.mandar2812.dynaml.analysis.VectorField
import spire.implicits._
import spire.algebra.Field

/**
  * Created by mandar on 26/7/16.
  */
case class GaussianRV(mu: Double, sigma: Double) extends ContinuousDistrRV[Double] {
  override val underlyingDist = new Gaussian(mu, sigma)
}

case class MultGaussianRV(
  mu: DenseVector[Double], covariance: DenseMatrix[Double])(
  implicit ev: Field[DenseVector[Double]])
  extends ContinuousDistrRV[DenseVector[Double]] {

  override val underlyingDist = MultivariateGaussian(mu, covariance)

}

object MultGaussianRV {

  def apply(num_dim: Int)(mu: DenseVector[Double], covariance: DenseMatrix[Double]) = {
    assert(
      num_dim == mu.length,
      "Number of dimensions of vector space must match the number of elements of mean")

    implicit val ev = VectorField(num_dim)

    new MultGaussianRV(mu, covariance)
  }
}