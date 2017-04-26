package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, StudentsT}
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.{PartitionedVectorField, VectorField}
import io.github.mandar2812.dynaml.probability.distributions.{
BlockedMultivariateStudentsT, MatrixT, MultivariateStudentsT}
import spire.implicits._
import spire.algebra.Field

abstract class AbstractStudentsTRandVar[I](mu: Double) extends ContinuousDistrRV[I]

/**
  * @author mandar2812 date 26/08/16.
  * */
case class StudentsTRV(mu: Double) extends AbstractStudentsTRandVar[Double](mu) {
  override val underlyingDist = new StudentsT(mu)

}

case class MultStudentsTRV(mu: Double,
                           mean: DenseVector[Double],
                           covariance : DenseMatrix[Double])(
  implicit ev: Field[DenseVector[Double]])
  extends AbstractStudentsTRandVar[DenseVector[Double]](mu) {

  override val underlyingDist = MultivariateStudentsT(mu, mean, covariance)
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

case class MultStudentsTPRV(mu: Double,
                            mean: PartitionedVector,
                            covariance: PartitionedPSDMatrix)(
  implicit ev: Field[PartitionedVector])
  extends AbstractStudentsTRandVar[PartitionedVector](mu) {

  override val underlyingDist: ContinuousDistr[PartitionedVector] = BlockedMultivariateStudentsT(mu, mean, covariance)
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
  AbstractStudentsTRandVar[DenseMatrix[Double]](mu) {

  override val underlyingDist = MatrixT(mu, m, u, v)
}
