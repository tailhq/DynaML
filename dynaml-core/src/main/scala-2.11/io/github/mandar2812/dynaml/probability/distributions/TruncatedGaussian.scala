package io.github.mandar2812.dynaml.probability.distributions

import breeze.stats.distributions._

/**
  * Univariate Truncated Gaussian Distribution
  *
  * @param mu The mean of the base gaussian.
  * @param sigma Std Deviation of the base gaussian.
  * @param a Lower limit of truncation.
  * @param b Upper limit of truncation.
  * @author mandar2812 date 03/03/2017.
  *
  * */
case class TruncatedGaussian(mu: Double, sigma: Double, a: Double, b: Double)(
  implicit rand: RandBasis = Rand) extends
  AbstractContinuousDistr[Double] with
  HasCdf with HasInverseCdf with Moments[Double, Double] {

  require(sigma > 0.0, "Std Dev must be positive.")
  require(a < b, "A must be lower limit, B must be upper limit")

  private val baseGaussian = Gaussian(mu, sigma)

  private val z = baseGaussian.cdf(b) - baseGaussian.cdf(a)
  private val y = baseGaussian.pdf(b) - baseGaussian.pdf(a)

  private val (alpha, beta) = ((a-mu)/sigma, (b-mu)/sigma)

  override def probability(x: Double, y: Double) = {
    require(
      x <= y,
      "Lower limit x must be actually lesser than upper limit y in P(x <= a <= y)")
    cdf(y) - cdf(x)
  }

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

  override def mean = mu - sigma*(baseGaussian.pdf(b) - baseGaussian.pdf(a))/z

  override def variance =
    sigma*sigma*(1.0 - ((beta*baseGaussian.pdf(b) - alpha*baseGaussian.pdf(a))/z) - math.pow(y/z, 2.0))

  override def entropy =
    (alpha*baseGaussian.pdf(a) - beta*baseGaussian.pdf(b))/(2*z) + math.sqrt(2*math.Pi*math.exp(1.0))*sigma*z

  override def mode = if (mu < a) a else if (mu > b) b else mu
}
