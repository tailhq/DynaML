package io.github.mandar2812.dynaml.kernels

import spire.algebra.Field

/**
  * An abstract representation of stationary kernel functions,
  * this requires an implicit variable which represents how addition, subtraction etc
  * are carried out for the input domain [[T]]
  * */
abstract class StationaryKernel[T, V, M](implicit ev: Field[T]) extends CovarianceFunction[T, V, M] {

  override def evaluate(x: T, y: T): V = eval(ev.minus(x,y))

  def eval(x: T): V
}
