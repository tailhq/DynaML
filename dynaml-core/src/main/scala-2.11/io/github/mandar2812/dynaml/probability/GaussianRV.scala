package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.Gaussian
import spire.implicits._

/**
  * Created by mandar on 26/7/16.
  */
case class GaussianRV(mu: Double, sigma: Double) extends ContinuousDistrRV[Double] {
  override val underlyingDist = new Gaussian(mu, sigma)
}
