package io.github.mandar2812.dynaml.probability.distributions

import breeze.stats.distributions.{Density, Rand}

/**
  * Created by mandar on 06/01/2017.
  */
abstract class GenericDistribution[T] extends Density[T] with Rand[T]
