package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseMatrix, DenseVector}

/**
  * @author mandar2812
  *
  * Implementation of the periodic kernel
  * K(x,y) = exp(-2*sin^2(pi*omega*x-y/l^2))
  */
class PeriodicKernel(private var bandwidth: Double = 1.0,
                     private var freq: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("bandwidth", "frequency")

  state = Map("bandwidth" -> bandwidth, "frequency" -> freq)

  def setbandwidth(d: Double): Unit = {
    this.state += ("bandwidth" -> d)
    this.bandwidth = d
  }

  def setfrequency(f: Double): Unit = {
    this.state += ("frequency" -> f)
    this.freq = f
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    Math.exp(-2*math.pow(math.sin(norm(diff, 1)*math.Pi*state("frequency")), 2)/
      (2*math.pow(this.state("bandwidth"), 2)))
  }

  def getBandwidth: Double = this.bandwidth
}

class PeriodicCovFunc(private var bandwidth: Double = 1.0,
                      private var freq: Double = 1.0)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters = List("bandwidth", "frequency")

  state = Map("bandwidth" -> bandwidth, "frequency" -> freq)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = x - y
    Math.exp(-2*math.pow(math.sin(diff*math.Pi*state("frequency")), 2)/
      (2*math.pow(this.state("bandwidth"), 2)))
  }
}
