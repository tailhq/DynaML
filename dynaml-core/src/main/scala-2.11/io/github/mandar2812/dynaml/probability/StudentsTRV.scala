package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Moments, StudentsT}
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.{PartitionedVectorField, VectorField}
import io.github.mandar2812.dynaml.probability.distributions._
import spire.implicits._
import spire.algebra.Field

abstract class AbstractStudentsTRandVar[
T, V, Distr <: ContinuousDistr[T] with Moments[T, V] with HasErrorBars[T]](mu: Double) extends
  ContinuousRVWithDistr[T, Distr]

/**
  * @author mandar2812 date 26/08/16.
  * */
case class StudentsTRV(mu: Double, mean: Double, sigma: Double) extends
  AbstractStudentsTRandVar[Double, Double, UnivariateStudentsT](mu) {

  override val underlyingDist = UnivariateStudentsT(mu, mean, sigma)

}

case class MultStudentsTRV(
  mu: Double, mean: DenseVector[Double],
  covariance : DenseMatrix[Double])(
  implicit ev: Field[DenseVector[Double]])
  extends AbstractStudentsTRandVar[DenseVector[Double], DenseMatrix[Double], MultivariateStudentsT](mu) {

  override val underlyingDist: MultivariateStudentsT = MultivariateStudentsT(mu, mean, covariance)
}

object MultStudentsTRV {

  def apply(num_dim: Int)(mu: Double, mean: DenseVector[Double], covariance: DenseMatrix[Double]) = {
    assert(
      num_dim == mean.length,
      "Number of dimensions of vector space must match the number of elements of mean")

    implicit val ev = VectorField(num_dim)

    new MultStudentsTRV(mu, mean, covariance)
  }
}

case class MultStudentsTPRV(
  mu: Double,
  mean: PartitionedVector,
  covariance: PartitionedPSDMatrix)(
  implicit ev: Field[PartitionedVector])
  extends AbstractStudentsTRandVar[PartitionedVector, PartitionedPSDMatrix, BlockedMultivariateStudentsT](mu) {

  override val underlyingDist: BlockedMultivariateStudentsT = BlockedMultivariateStudentsT(mu, mean, covariance)
}

object MultStudentsTPRV {

  def apply(num_dim: Long, nE: Int)(mu: Double, mean: PartitionedVector, covariance: PartitionedPSDMatrix) = {
    assert(
      num_dim == mean.rows,
      "Number of dimensions of vector space must match the number of elements of mean")

    implicit val ev = PartitionedVectorField(num_dim, nE)

    new MultStudentsTPRV(mu, mean, covariance)
  }

}

case class MatrixTRV(
  mu: Double, m: DenseMatrix[Double],
  u: DenseMatrix[Double],
  v: DenseMatrix[Double]) extends
  AbstractStudentsTRandVar[DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double]), MatrixT](mu) {

  override val underlyingDist = MatrixT(mu, m, v, u)
}
