package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseMatrix, DenseVector}

/**
  * @author mandar2812
  *
  * Implementation of the periodic kernel
  * K(x,y) = &sigma;<sup>2</sup>exp(-2*sin<sup>2</sup>(&pi;*&omega;*|x-y|)/l<sup>2</sup>)
  * */
class PeriodicKernel(s: Double = 1d, ls: Double = 1.0, freq: Double = 1.0)
    extends SVMKernel[DenseMatrix[Double]]
    with LocalScalarKernel[DenseVector[Double]]
    with Serializable {

  override val hyper_parameters = List("sigma", "lengthscale", "frequency")

  state = Map("sigma" -> s, "lengthscale" -> ls, "frequency" -> freq)

  def setlengthscale(d: Double): Unit = {
    this.state += ("lengthscale" -> d)
  }

  def setfrequency(f: Double): Unit = {
    this.state += ("frequency" -> f)
  }

  override def evaluateAt(
    config: Map[String, Double]
  )(x: DenseVector[Double],
    y: DenseVector[Double]
  ): Double = {

    val diff = x - y

    math.pow(config("sigma"), 2) * math.exp(
      -2 * math
        .pow(math.sin(norm(diff, 1) * math.Pi * config("frequency")), 2) /
        math.pow(config("lengthscale"), 2)
    )
  }

  override def gradientAt(
    config: Map[String, Double]
  )(x: DenseVector[Double],
    y: DenseVector[Double]
  ): Map[String, Double] = {

    val diff = norm(x - y, 1)
    val s2   = math.pow(config("sigma"), 2)
    val k = math.Pi * config("frequency") * diff / math.pow(
      config("lengthscale"),
      2
    )

    Map(
      "frequency" -> -2.0 * s2 * evaluateAt(config)(x, y) * math
        .sin(2.0 * k) * k / config("frequency"),
      "lengthscale" -> 4.0 * s2 * evaluateAt(config)(x, y) * math
        .sin(2 * k) * k / config("lengthscale"),
      "sigma" -> 2d * evaluateAt(config)(x, y) / config("sigma")
    )
  }

  def lengthscale: Double = this.state("lengthscale")
  def frequency: Double   = this.state("frequency")
  def sigma: Double       = this.state("sigma")
}

class PeriodicCovFunc(s: Double = 1d, ls: Double = 1.0, freq: Double = 1.0)
    extends LocalScalarKernel[Double] {

  override val hyper_parameters = List("sigma", "lengthscale", "frequency")

  state = Map("sigma" -> s, "lengthscale" -> ls, "frequency" -> freq)

  override def evaluateAt(
    config: Map[String, Double]
  )(x: Double,
    y: Double
  ): Double = {

    val diff = math.abs(x - y)

    math.pow(config("sigma"), 2) * math.exp(
      -2 * math.pow(math.sin(diff * math.Pi * config("frequency")), 2) /
        math.pow(config("lengthscale"), 2)
    )
  }

  override def gradientAt(
    config: Map[String, Double]
  )(x: Double,
    y: Double
  ): Map[String, Double] = {

    val diff = math.abs(x - y)
    val s2   = math.pow(config("sigma"), 2)
    val k = math.Pi * config("frequency") * diff / math.pow(
      config("lengthscale"),
      2
    )

    Map(
      "frequency" -> -2.0 * s2 * evaluateAt(config)(x, y) * math
        .sin(2.0 * k) * k / config("frequency"),
      "lengthscale" -> 4.0 * s2 * evaluateAt(config)(x, y) * math
        .sin(2 * k) * k / config("lengthscale"),
      "sigma" -> 2d * evaluateAt(config)(x, y) / config("sigma")
    )
  }
}
