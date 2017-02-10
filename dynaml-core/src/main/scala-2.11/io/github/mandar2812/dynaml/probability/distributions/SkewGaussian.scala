package io.github.mandar2812.dynaml.probability.distributions


import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, sqrt}
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils._
import spire.algebra.Field
import spire.implicits._

/**
  * @author mandar2812 date: 02/01/2017.
  *
  * Univariate skew gaussian distribution
  */
case class SkewGaussian(
  alpha: Double, mu: Double = 0.0,
  sigma: Double = 1.0) extends
  SkewSymmDistribution[Double](
    Gaussian(mu, sigma), Gaussian(mu, sigma),
    DataPipe((x: Double) => alpha*x))

/**
  * The univariate extended skew gaussian distribution
  * */
case class ExtendedSkewGaussian(
  alpha0: Double, alpha: Double,
  mu: Double = 0.0, sigma: Double = 1.0)(
  implicit rand: RandBasis = Rand)
  extends SkewSymmDistribution[Double](
    Gaussian(mu, sigma), Gaussian(mu, sigma),
    DataPipe((x: Double) => alpha*x), alpha0) {

  override def logNormalizer =
    log(warpingDistr.cdf(alpha0/sqrt(1 + alpha*alpha))) + basisDistr.logNormalizer

}

case class MultivariateSkewNormal(
  alpha: DenseVector[Double],
  mu: DenseVector[Double],
  sigma: DenseMatrix[Double]) extends
  SkewSymmDistribution[DenseVector[Double]](
    MultivariateGaussian(mu, sigma), Gaussian(0.0, 1.0),
    DataPipe((x: DenseVector[Double]) => alpha.t*(diagonal(sigma)\(x-mu))))(
    VectorField(alpha.length)
  )
