package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.Uniform
import spire.implicits._

/**
  * Created by mandar on 26/7/16.
  */
case class UniformRV(min: Double, max: Double) extends ContinuousDistrRV[Double] {
  override val underlyingDist = new Uniform(min, max)
}
