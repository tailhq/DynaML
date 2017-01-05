package io.github.mandar2812.dynaml.probability.distributions


import breeze.numerics.{log, sqrt}
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.pipes.DataPipe
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
