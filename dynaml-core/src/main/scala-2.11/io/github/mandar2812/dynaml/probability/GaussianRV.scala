package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Gaussian, MultivariateGaussian}
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.{PartitionedVectorField, VectorField}
import io.github.mandar2812.dynaml.probability.distributions.BlockedMultiVariateGaussian
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

case class MultGaussianPRV(mu: PartitionedVector, covariance: PartitionedPSDMatrix)(
  implicit ev: Field[PartitionedVector])
  extends ContinuousDistrRV[PartitionedVector] {

  override val underlyingDist: ContinuousDistr[PartitionedVector] = BlockedMultiVariateGaussian(mu, covariance)

}

object MultGaussianPRV {

  def apply(num_dim: Long, nE: Int)(mu: PartitionedVector, covariance: PartitionedPSDMatrix) = {
    assert(
      num_dim == mu.rows,
      "Number of dimensions of vector space must match the number of elements of mean")

    implicit val ev = PartitionedVectorField(num_dim, nE)

    new MultGaussianPRV(mu, covariance)
  }

}