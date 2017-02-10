package io.github.mandar2812.dynaml.probability.distributions


import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, sqrt}
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils._
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

/**
  * Multivariate Skew-Normal distribution as
  * specified in Azzalani et. al.
  *
  * @param alpha A breeze [[DenseVector]] which represents the skewness parameters
  *
  * @param mu The center of the distribution
  *
  * @param sigma The covariance matrix of the base multivariate gaussian.
  *
  * */
case class MultivariateSkewNormal(
  alpha: DenseVector[Double],
  mu: DenseVector[Double],
  sigma: DenseMatrix[Double]) extends
  SkewSymmDistribution[DenseVector[Double]](
    basisDistr = MultivariateGaussian(mu, sigma),
    warpingDistr = Gaussian(0.0, 1.0),
    w = DataPipe((x: DenseVector[Double]) => alpha.t*(sqrt(diagonal(sigma))\(x-mu))))(
    VectorField(alpha.length))

/**
  * Extended Multivariate Skew-Gaussian distribution
  * as specified in Azzalani et.al.
  *
  * @param tau Determines the cutoff of the warping function
  *
  * @param alpha A breeze [[DenseVector]] which represents the skewness parameters
  *
  * @param mu The center of the distribution
  *
  * @param sigma The covariance matrix of the base multivariate gaussian.
  * */
case class ExtendedMultivariateSkewNormal(
  tau: Double, alpha: DenseVector[Double],
  mu: DenseVector[Double], sigma: DenseMatrix[Double]) extends
  SkewSymmDistribution[DenseVector[Double]](
    basisDistr = MultivariateGaussian(mu, sigma),
    warpingDistr = Gaussian(0.0, 1.0),
    w = DataPipe((x: DenseVector[Double]) => alpha.t*(sqrt(diagonal(sigma))\(x-mu))),
    cutoff = tau*sqrt(1.0 + alpha.t*(diagonal(sigma)\sigma)*alpha))(
    VectorField(alpha.length))
