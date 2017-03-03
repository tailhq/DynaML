package io.github.mandar2812.dynaml.probability.distributions

import breeze.stats.distributions._

/**
  * @author mandar2812 date 03/03/2017.
  *
  * Univariate Truncated Gaussian Distribution
  */
case class TruncatedGaussian(mu: Double, sigma: Double, a: Double, b: Double)(
  implicit rand: RandBasis = Rand) extends
  AbstractContinuousDistr[Double] with
  HasCdf with HasInverseCdf {

  require(sigma > 0.0, "Std Dev must be positive.")
  require(a < b, "A must be lower limit, B must be upper limit")

  private val baseGaussian = Gaussian(mu, sigma)

  private val z = baseGaussian.cdf(b) - baseGaussian.cdf(a)

  override def probability(x: Double, y: Double) = ???

  override def cdf(x: Double) =
    if(x <= a) 0.0
    else if(x >= b) 1.0
    else (baseGaussian.cdf(x) - baseGaussian.cdf(b))/z

  override def inverseCdf(p: Double) = baseGaussian.icdf(baseGaussian.cdf(a) + p*z)

  override def unnormalizedLogPdf(x: Double) =
    if(x <= b && x >= a) baseGaussian.logPdf(x)
    else Double.NegativeInfinity

  override def logNormalizer = math.log(z)

  override def draw() = {
    inverseCdf(rand.uniform.draw())
  }
}
