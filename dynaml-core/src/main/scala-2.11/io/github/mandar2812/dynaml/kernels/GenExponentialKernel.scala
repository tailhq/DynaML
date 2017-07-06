package io.github.mandar2812.dynaml.kernels

import spire.algebra.MetricSpace

/**
  * Covariance expressed as exponential of some
  * scaled distance metric between arguments.
  *
  * K(x, y) = &sigma;<sup>2<sup> exp(-1/2 &times; ||x-y||/&theta;)
  *
  * @author mandar2812 date 06/07/2017.
  * */
class GenExponentialKernel[I](
  sigma: Double, theta: Double)(
  implicit f: MetricSpace[I, Double]) extends
  LocalScalarKernel[I] {

  override val hyper_parameters = List("sigma", "lengthScale")

  override def evaluateAt(config: Map[String, Double])(x: I, y: I) = {
    val (sigma, theta) = (config("sigma"), config("lengthScale"))
    val d = f.distance(x, y)
    sigma*sigma*math.exp(-0.5*d/theta)
  }

  override def gradientAt(config: Map[String, Double])(x: I, y: I) = {
    val (sigma, theta) = (config("sigma"), config("lengthScale"))
    val d = f.distance(x, y)
    val c = evaluateAt(config)(x, y)

    Map(
      "sigma" -> 2*c/sigma,
      "lengthScale" -> 0.5*d*c/(theta*theta)
    )
  }
}

object GenExponentialKernel {

  def apply[I](
    sigma: Double, theta: Double)(
    implicit f: MetricSpace[I, Double]): GenExponentialKernel[I] =
    new GenExponentialKernel[I](sigma, theta)

  def apply[I](r: GenericRBFKernel[I]): GenExponentialKernel[I] = {

    val sigma = if(r.state.contains("amplitude")) r.state("amplitude") else 1d

    val theta = r.state("bandwidth")

    implicit val m: MetricSpace[I, Double] = new MetricSpace[I, Double] {
      override def distance(v: I, w: I) = {
        val diff = r.ev.minus(v, w)
        r.ev.dot(diff, diff)
      }
    }

    apply[I](sigma, theta)

  }

}