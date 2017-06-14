package io.github.mandar2812.dynaml.probability

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Binomial, Multinomial}

/**
  * A binomial random variable.
  * Simulates the accumulated results of n coin tosses
  * of a loaded coin.
  * @author mandar date 26/7/16.
  */
case class BinomialRV(n: Int, p: Double) extends DiscreteDistrRV[Int] {

  override val underlyingDist = Binomial(n, p)

}

/**
  *
  * A multinomial random variable
  * i.e. draws values between 0 and N-1
  * @author mandar2812 date 14/6/17
  * */
case class MultinomialRV(weights: DenseVector[Double]) extends DiscreteDistrRV[Int] {

  override val underlyingDist = new Multinomial[DenseVector[Double], Int](weights)

}