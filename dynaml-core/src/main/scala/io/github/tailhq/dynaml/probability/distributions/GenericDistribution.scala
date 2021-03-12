package io.github.tailhq.dynaml.probability.distributions

import breeze.stats.distributions.{ContinuousDistr, Density, DiscreteDistr, Rand}

/**
  * @author tailhq date 06/01/2017.
  * */
abstract class GenericDistribution[T] extends Density[T] with Rand[T] with Serializable

abstract class AbstractDiscreteDistr[T] extends GenericDistribution[T] with DiscreteDistr[T]

abstract class AbstractContinuousDistr[T] extends GenericDistribution[T] with ContinuousDistr[T]

/**
  * Distributions which can generate confidence intervals around their mean
  * can extend this trait and override the `confidenceInterval` method .
  * */
trait HasErrorBars[T] extends Serializable {

  def confidenceInterval(s: Double): (T, T)
}