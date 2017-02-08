package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseMatrix, DenseVector}

/**
  * @author mandar2812
  *
  * Implementation of the periodic kernel
  * K(x,y) = exp(-2*sin^2(pi*omega*x-y/l^2))
  */
class PeriodicKernel(private var lengthscale: Double = 1.0,
                     private var freq: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalScalarKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("lengthscale", "frequency")

  state = Map("lengthscale" -> lengthscale, "frequency" -> freq)

  def setlengthscale(d: Double): Unit = {
    this.state += ("lengthscale" -> d)
    this.lengthscale = d
  }

  def setfrequency(f: Double): Unit = {
    this.state += ("frequency" -> f)
    this.freq = f
  }

  override def evaluateAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Double = {

    val diff = x - y

    Math.exp(-2*math.pow(math.sin(norm(diff, 1)*math.Pi*config("frequency")), 2)/
      (2*math.pow(config("lengthscale"), 2)))
  }

  override def gradientAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Map[String, Double] = {

    val diff = norm(x-y, 1)
    val k = math.Pi*config("frequency")*diff/math.pow(config("lengthscale"), 2)

    Map(
      "frequency" -> -2.0*evaluateAt(config)(x,y)*math.sin(2.0*k)*k/config("frequency"),
      "lengthscale" -> 4.0*evaluateAt(config)(x,y)*math.sin(2*k)*k/config("lengthscale")
    )
  }

  def getlengthscale: Double = this.lengthscale
}

class PeriodicCovFunc(private var lengthscale: Double = 1.0,
                      private var freq: Double = 1.0)
  extends LocalScalarKernel[Double] {

  override val hyper_parameters = List("lengthscale", "frequency")

  state = Map("lengthscale" -> lengthscale, "frequency" -> freq)

  override def evaluateAt(
    config: Map[String, Double])(
    x: Double, y: Double): Double = {

    val diff = x - y

    Math.exp(-2*math.pow(math.sin(diff*math.Pi*config("frequency")), 2)/
      (2*math.pow(config("lengthscale"), 2)))
  }

  override def gradientAt(
    config: Map[String, Double])(
    x: Double,
    y: Double): Map[String, Double] = {

    val diff = math.abs(x-y)
    val k = math.Pi*config("frequency")*diff/math.pow(config("lengthscale"), 2)

    Map(
      "frequency" -> -2.0*evaluateAt(config)(x,y)*math.sin(2.0*k)*k/config("frequency"),
      "lengthscale" -> 4.0*evaluateAt(config)(x,y)*math.sin(2*k)*k/config("lengthscale")
    )
  }
}
