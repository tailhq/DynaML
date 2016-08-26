package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.StudentsT
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.probability.distributions.MultivariateStudentsT
import spire.implicits._
import spire.algebra.Field

/**
  * Created by mandar on 26/08/16.
  */
case class StudentsTRV(mu: Double) extends ContinuousDistrRV[Double] {
  override val underlyingDist = new StudentsT(mu)

}

case class MultStudentsTRV(mu: Double,
                           mean: DenseVector[Double],
                           covariance : DenseMatrix[Double])(implicit ev: Field[DenseVector[Double]])
  extends ContinuousDistrRV[DenseVector[Double]] {

  override val underlyingDist = new MultivariateStudentsT(mu, mean, covariance)
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