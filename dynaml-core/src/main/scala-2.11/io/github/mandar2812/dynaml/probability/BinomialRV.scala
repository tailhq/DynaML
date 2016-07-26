package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.Binomial

/**
  * Created by mandar on 26/7/16.
  */
case class BinomialRV(n: Int, p: Double) extends DiscreteDistrRV[Int]{

  override val underlyingDist = new Binomial(n, p)

}
