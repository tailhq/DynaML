package io.github.mandar2812.dynaml.optimization

/**
 * @author mandar2812 datum 24/6/15.
 *
 * High level interface defining the
 * core functions of a global optimizer
 */
trait GlobalOptimizer[T <: GloballyOptimizable] {

  val system: T

  def optimize(initialConfig: Map[String, Double],
               options: Map[String, String]): (T, Map[String, Double])

}
