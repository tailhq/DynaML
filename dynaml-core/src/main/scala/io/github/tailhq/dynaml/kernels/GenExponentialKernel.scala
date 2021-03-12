package io.github.tailhq.dynaml.kernels

import io.github.tailhq.dynaml.pipes.DataPipe2
import spire.algebra.MetricSpace

/**
  * Covariance expressed as exponential of some
  * scaled distance metric between arguments.
  *
  * K(x, y) = &sigma;<sup>2<sup> exp(-1/2 &times; ||x-y||/&theta;)
  *
  * @author tailhq date 06/07/2017.
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

/**
  * An exponential family covariance function between space time coordinates
  * */
class GenExpSpaceTimeKernel[I](
  sigma: Double, theta_space: Double, theta_time: Double)(
  val ds: DataPipe2[I, I, Double],
  val dt: DataPipe2[Double, Double, Double]) extends
  LocalScalarKernel[(I, Double)] {

  override val hyper_parameters = List("sigma", "spaceScale", "timeScale")

  state = Map("sigma" -> sigma, "spaceScale" -> theta_space, "timeScale" -> theta_time)

  override def evaluateAt(config: Map[String, Double])(x: (I, Double), y: (I, Double)) = {
    val (sigma, theta_s, theta_t) = (
      config("sigma"),
      config("spaceScale"),
      config("timeScale"))

    val (xs, xt) = x
    val (ys, yt) = y

    val d = ds(xs, ys)
    val t = dt(xt, yt)

    sigma*sigma*math.exp(-0.5*((d/theta_s) + (t/theta_t)))
  }

  override def gradientAt(config: Map[String, Double])(x: (I, Double), y: (I, Double)) = {
    val (sigma, theta_s, theta_t) = (config("sigma"), config("spaceScale"), config("timeScale"))
    val (xs, xt) = x
    val (ys, yt) = y
    val d = ds(xs, ys)
    val t = dt(xt, yt)
    val c = evaluateAt(config)(x, y)

    Map(
      "sigma" -> 2*c/sigma,
      "spaceScale" -> 0.5*d*c/(theta_s*theta_s),
      "timeScale" -> 0.5*t*c/(theta_t*theta_t)
    )
  }

}