package io.github.mandar2812.dynaml.kernels

import spire.algebra.Field

/**
  * An abstract representation of stationary kernel functions,
  * this requires an implicit variable which represents how addition, subtraction etc
  * are carried out for the input domain [[T]]
  * */
abstract class StationaryKernel[T, V, M](implicit ev: Field[T]) extends CovarianceFunction[T, V, M] {

  def evalAt(config: Map[String, Double])(x: T): V

  def eval(x: T): V = evalAt(state)(x)

  override def evaluateAt(config: Map[String, Double])(x: T, y: T) = evalAt(config)(ev.minus(x,y))
}
