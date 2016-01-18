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
  with LocalSVMKernel[DenseVector[Double]]
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

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    Math.exp(-2*math.pow(math.sin(norm(diff, 1)*math.Pi*state("frequency")), 2)/
      (2*math.pow(this.state("lengthscale"), 2)))
  }

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] = {
    val diff = norm(x-y, 1)
    val k = math.Pi*state("frequency")*diff/math.pow(state("lengthscale"),2)

    Map(
      "frequency" -> -2.0*evaluate(x,y)*math.sin(2.0*k)*k/state("frequency"),
      "lengthscale" -> 4.0*evaluate(x,y)*math.sin(2*k)*k/state("lengthscale")
    )
  }

  def getlengthscale: Double = this.lengthscale
}

class PeriodicCovFunc(private var lengthscale: Double = 1.0,
                      private var freq: Double = 1.0)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters = List("lengthscale", "frequency")

  state = Map("lengthscale" -> lengthscale, "frequency" -> freq)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = x - y
    Math.exp(-2*math.pow(math.sin(diff*math.Pi*state("frequency")), 2)/
      (2*math.pow(this.state("lengthscale"), 2)))
  }

  override def gradient(x: Double,
                        y: Double): Map[String, Double] = {
    val diff = math.abs(x-y)
    val k = math.Pi*state("frequency")*diff/math.pow(state("lengthscale"),2)

    Map(
      "frequency" -> -2.0*evaluate(x,y)*math.sin(2.0*k)*k/state("frequency"),
      "lengthscale" -> 4.0*evaluate(x,y)*math.sin(2*k)*k/state("lengthscale")
    )
  }
}
