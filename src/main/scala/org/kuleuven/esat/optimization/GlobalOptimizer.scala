package org.kuleuven.esat.optimization

/**
 * @author mandar2812 datum 24/6/15.
 *
 * High level interface defining the
 * core functions of a global optimizer
 */
trait GlobalOptimizer[T <: GloballyOptimizable] {

  val system: T

  def optimize(initialConfig: Map[String, Double],
               options: Map[String, AnyRef]): Unit

}
