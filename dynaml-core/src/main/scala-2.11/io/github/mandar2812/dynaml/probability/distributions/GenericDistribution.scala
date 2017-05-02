package io.github.mandar2812.dynaml.probability.distributions

import breeze.stats.distributions.{ContinuousDistr, Density, DiscreteDistr, Rand}

/**
  * Created by mandar on 06/01/2017.
  */
abstract class GenericDistribution[T] extends Density[T] with Rand[T]

abstract class AbstractDiscreteDistr[T] extends GenericDistribution[T] with DiscreteDistr[T]

abstract class AbstractContinuousDistr[T] extends GenericDistribution[T] with ContinuousDistr[T]

trait HasErrorBars[T] {

  def confidenceInterval(s: Double): (T, T)
}